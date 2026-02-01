//! Blind acquisition via FFT (Specification §4.B).
//!
//! The receiver does not know the transmit `TimeIndex` (`TI_tx`), so acquisition searches over:
//! - `TI_search` in a time window `[ti_min .. ti_min + n_ti)`
//! - intra-epoch sample offset `off` in `[0 .. iv_samples)`
//! - CFO in a limited band using an FFT peak search
//!
//! Coarse stage:
//! - coherently accumulate the two preamble symbols (data segments only) as one sparse window
//!   ("despread then FFT"), skipping the symbol-0 guard/noise by inserting zeros
//! - zero-pad to `N_FFT` and take an FFT
//! - keep the top-K hypotheses by peak power in the CFO search band
//!
//! Refine/verify stage:
//! - refine CFO on a small grid around the coarse estimate
//! - verify candidates with coherent preamble energy + noncoherent pilot energy
//! - if verified, search for RAKE finger delays around the best preamble offset
//!
//! Coordinate system:
//! ```text
//! rx_window = [ IV(ti_min) | IV(ti_min+1) | ... ]
//! base(ti)  = (ti - ti_min) * iv_samples
//! start     = base(ti) + off
//! ```

use super::{AcqResult, ScBltcModem};
use crate::crypto::{gen_code_structured_aes_ctr, JitterSpec};
use crate::modem::util::{derotate_cfo_in_place, pulse_shape_chips};
use num_complex::Complex32;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct Cand {
    ti: u64,
    off: usize,
    p_max: f32,
    f_hat: f64,
}

fn cand_better(a: Cand, b: Cand) -> bool {
    match a.p_max.partial_cmp(&b.p_max).unwrap_or(Ordering::Equal) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => (a.ti, a.off) < (b.ti, b.off),
    }
}

fn push_topk(topk: &mut Vec<Cand>, cand: Cand, k_keep: usize) {
    if topk.len() < k_keep {
        topk.push(cand);
        return;
    }
    let mut min_i = 0usize;
    for i in 1..topk.len() {
        if cand_better(topk[min_i], topk[i]) {
            min_i = i;
        }
    }
    if cand_better(cand, topk[min_i]) {
        topk[min_i] = cand;
    }
}

struct AcqContext {
    iv_samples: usize,
    rake_search_half_samples: usize,
    l_sym: usize,
    /// Length (samples) of a single preamble data segment (SF * OSF).
    n_ref_pre0: usize,
    /// Coarse/refine preamble window length:
    /// `sym0_data || zeros(max_j0) || sym1_data` (samples).
    n_ref_pre_acq: usize,
    /// Sample offset from symbol-0 start to symbol-1 start for each TI hypothesis.
    pre1_delta_samp_by_ti: Vec<usize>,
    pilot_timing_win: isize,
    fs: f64,
    nfft: usize,
    bin_max: isize,
    scratch_len: usize,
    ref_pre0_conj_by_ti: Vec<Vec<Complex32>>,
    ref_pre1_conj_by_ti: Vec<Vec<Complex32>>,
}

impl ScBltcModem {
    fn estimate_noise_power_mu(&self, x: &[Complex32]) -> f64 {
        // median(|x|^2) = mu * ln(2)
        if x.is_empty() {
            return 0.0;
        }
        let target = 10_000usize;
        let step = (x.len() / target).max(1);
        let mut pwr: Vec<f64> = x
            .iter()
            .step_by(step)
            .map(|v| (v.norm_sqr() as f64).max(0.0))
            .collect();
        if pwr.is_empty() {
            return 0.0;
        }
        let mid = pwr.len() / 2;
        let (_, med, _) =
            pwr.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let med = *med;
        med / std::f64::consts::LN_2
    }

    fn corr_energy_no_cfo(
        &self,
        rx: &[Complex32],
        start: usize,
        ref_conj_rot: &[Complex32],
    ) -> f64 {
        let rx_seg = &rx[start..start + ref_conj_rot.len()];
        let mut acc = Complex32::new(0.0, 0.0);
        for (r, c) in rx_seg.iter().zip(ref_conj_rot.iter()) {
            acc += *r * *c;
        }
        (acc.norm_sqr() as f64).max(0.0)
    }

    fn energy_of_seq_with_cfo(&self, y: &[Complex32], fs_hz: u32, cfo_hz: f64) -> f64 {
        // Computes |sum_k y[k] * exp(-j 2π f k / fs)|^2.
        let fs = fs_hz as f32;
        let dphi = -2.0 * std::f32::consts::PI * (cfo_hz as f32) / fs;
        let w = Complex32::from_polar(1.0, dphi);
        let mut ph = Complex32::new(1.0, 0.0);
        let mut acc = Complex32::new(0.0, 0.0);
        for v in y {
            acc += *v * ph;
            ph *= w;
        }
        (acc.norm_sqr() as f64).max(0.0)
    }

    fn build_symbol_ref_tx_shaped(&self, chips: &[i8], hop_hz: f64) -> Vec<Complex32> {
        let p = &self.p;
        let mut x = pulse_shape_chips(chips, &self.rrc, p.osf() as usize);
        // Apply the hop for this symbol (receiver reference uses the same hop).
        derotate_cfo_in_place(&mut x, p.fs_hz(), -hop_hz);
        x
    }

    fn prepare_acq_context(
        &self,
        rx_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
    ) -> anyhow::Result<AcqContext> {
        let p = &self.p;
        let iv_samples = ((p.fs_hz() as f64) * p.iv_res_s()).round() as usize;
        if iv_samples == 0 {
            anyhow::bail!("invalid iv_samples");
        }
        if p.rake_search_half_s() < 0.0 {
            anyhow::bail!("invalid rake_search_half_s");
        }
        let rake_search_half_samples =
            ((p.fs_hz() as f64) * p.rake_search_half_s()).round() as usize;

        let l_sym = p.chip_samples();
        let n_ref_pre0 = l_sym;
        let max_j0_samp = p.jitter_max_chips() * (p.osf() as usize);
        // Two preamble symbols coherently accumulated, skipping symbol-0 GI:
        // `sym0_data || zeros(j0) || sym1_data`.
        //
        // We size the buffer using the *maximum* possible `j0` so the FFT inputs have
        // a fixed length across TI hypotheses. Each TI places the symbol-1 segment at its
        // actual `j0` offset and leaves the rest as zeros.
        let n_ref_pre_acq = 2 * l_sym + max_j0_samp;

        if p.nfft_acq() < n_ref_pre_acq {
            anyhow::bail!(
                "Params.nfft_acq must be >= (2*SF + jitter_max)*OSF (need >= {}, got {})",
                n_ref_pre_acq,
                p.nfft_acq()
            );
        }

        let pilot_timing_win: isize = 32;
        let frame_max = p.frame_max_samples_with_jitter();
        let need = n_ti
            .checked_mul(iv_samples)
            .and_then(|v| v.checked_add(rake_search_half_samples))
            .and_then(|v| v.checked_add(frame_max))
            .and_then(|v| v.checked_add(32))
            .ok_or_else(|| anyhow::anyhow!("window_size_overflow"))?;
        if rx_window.len() < need {
            anyhow::bail!(
                "insufficient rx_window: need >= {}, got {}",
                need,
                rx_window.len()
            );
        }

        let fs = p.fs_hz() as f64;
        let nfft = p.nfft_acq();

        let search_hz = p.cfo_search_hz().abs().min(0.5 * fs);
        let bin_max = ((search_hz * (nfft as f64)) / fs).floor() as isize;
        let bin_max = bin_max.clamp(1, (nfft as isize) / 2 - 1);

        let scratch_len = self.fft_acq_scratch_len;

        let mut pre1_delta_samp_by_ti: Vec<usize> = Vec::with_capacity(n_ti);
        let mut ref_pre0_conj_by_ti: Vec<Vec<Complex32>> = Vec::with_capacity(n_ti);
        let mut ref_pre1_conj_by_ti: Vec<Vec<Complex32>> = Vec::with_capacity(n_ti);
        for ti_idx in 0..n_ti {
            let ti = ti_min + (ti_idx as u64);
            let g = gen_code_structured_aes_ctr(
                &self.key,
                ti,
                p.n_sym(),
                p.sf(),
                p.domain_u32(),
                p.hop_bw_hz(),
                JitterSpec {
                    min_chips: p.jitter_min_chips(),
                    span_chips: p.jitter_span_chips(),
                },
            );

            let j0 = g.j_seq_chips.first().copied().unwrap_or(0);
            let delta_sym1 = l_sym + j0 * (p.osf() as usize);
            pre1_delta_samp_by_ti.push(delta_sym1);

            // Preamble symbol 0 uses Walsh index 0, so the chips are just C_seq[0].
            let chips0 = &g.c_seq[0..p.sf()];
            let mut ref0 = pulse_shape_chips(chips0, &self.rrc, p.osf() as usize);
            // Apply the hop for symbol 0 (frame-local phase starts at 1 at the frame start).
            derotate_cfo_in_place(&mut ref0, p.fs_hz(), -g.f_seq_hz[0]);
            debug_assert_eq!(ref0.len(), n_ref_pre0);
            ref_pre0_conj_by_ti.push(ref0.iter().map(|v| v.conj()).collect());

            // Preamble symbol 1 is `-W0`, i.e. `-C_seq[sf..2sf]`.
            let seg1 = p.sf();
            let chips1 = &g.c_seq[seg1..seg1 + p.sf()];
            let mut ref1 = pulse_shape_chips(chips1, &self.rrc, p.osf() as usize);
            derotate_cfo_in_place(&mut ref1, p.fs_hz(), -g.f_seq_hz[1]);
            for v in &mut ref1 {
                *v = -*v;
            }
            // Preserve the hop's phase continuity across the (data+GI) span of symbol 0.
            //
            // Acquisition coherently accumulates both preamble symbols, so symbol 1's reference
            // must include the accumulated phase at its start; otherwise the two segments would
            // carry an artificial phase jump.
            let fs_f32 = p.fs_hz() as f32;
            let phi0 = (2.0 * std::f32::consts::PI * (g.f_seq_hz[0] as f32) * (delta_sym1 as f32)
                / fs_f32)
                .rem_euclid(2.0 * std::f32::consts::PI);
            let ph0 = Complex32::from_polar(1.0, phi0);
            for v in &mut ref1 {
                *v *= ph0;
            }
            debug_assert_eq!(ref1.len(), n_ref_pre0);
            ref_pre1_conj_by_ti.push(ref1.iter().map(|v| v.conj()).collect());
        }

        Ok(AcqContext {
            iv_samples,
            rake_search_half_samples,
            l_sym,
            n_ref_pre0,
            n_ref_pre_acq,
            pre1_delta_samp_by_ti,
            pilot_timing_win,
            fs,
            nfft,
            bin_max,
            scratch_len,
            ref_pre0_conj_by_ti,
            ref_pre1_conj_by_ti,
        })
    }

    fn find_coarse_candidates(
        &self,
        rx_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        ctx: &AcqContext,
    ) -> Vec<Cand> {
        let fft = &self.fft_acq;
        debug_assert_eq!(fft.len(), ctx.nfft);

        let k_keep: usize = 50;
        let bin_max_u = ctx.bin_max as usize;
        let idx_of = |b: isize| -> usize {
            if b >= 0 {
                b as usize
            } else {
                ctx.nfft - ((-b) as usize)
            }
        };

        let off_chunk: usize = 4;
        (0..n_ti)
            .into_par_iter()
            .flat_map_iter(|ti_idx| {
                (0..ctx.iv_samples)
                    .step_by(off_chunk)
                    .map(move |off0| (ti_idx, off0, (off0 + off_chunk).min(ctx.iv_samples)))
            })
            .map_init(
                || {
                    (
                        vec![Complex32::new(0.0, 0.0); ctx.nfft],
                        vec![Complex32::new(0.0, 0.0); ctx.scratch_len],
                    )
                },
                |state, (ti_idx, off0, off1)| {
                    let buf = &mut state.0;
                    let scratch = &mut state.1;
                    let ti = ti_min + (ti_idx as u64);
                    let base = ti_idx * ctx.iv_samples;
                    let ref0 = &ctx.ref_pre0_conj_by_ti[ti_idx];
                    let ref1 = &ctx.ref_pre1_conj_by_ti[ti_idx];
                    let delta_sym1 = ctx.pre1_delta_samp_by_ti[ti_idx];
                    let mut local_topk: Vec<Cand> = Vec::with_capacity(k_keep);

                    for off in off0..off1 {
                        let y0 = base + off;
                        let y1 = y0 + delta_sym1;

                        // Two-symbol sparse preamble window (sym0, zeros(GI), sym1).
                        buf[..ctx.n_ref_pre_acq].fill(Complex32::new(0.0, 0.0));

                        for (dst, (r, c)) in buf[..ctx.n_ref_pre0]
                            .iter_mut()
                            .zip(rx_window[y0..y0 + ctx.n_ref_pre0].iter().zip(ref0.iter()))
                        {
                            *dst = *r * *c;
                        }

                        let sym1_off = delta_sym1;
                        let sym1_end = sym1_off + ctx.n_ref_pre0;
                        debug_assert!(sym1_end <= ctx.n_ref_pre_acq);
                        for (dst, (r, c)) in buf[sym1_off..sym1_end]
                            .iter_mut()
                            .zip(rx_window[y1..y1 + ctx.n_ref_pre0].iter().zip(ref1.iter()))
                        {
                            *dst = *r * *c;
                        }
                        buf[ctx.n_ref_pre_acq..].fill(Complex32::new(0.0, 0.0));

                        fft.process_with_scratch(&mut buf[..], &mut scratch[..]);

                        let mut p_max = f32::NEG_INFINITY;
                        let mut best_bin: isize = 0;

                        for (b, v) in buf.iter().enumerate().take(bin_max_u + 1) {
                            let pw = v.re * v.re + v.im * v.im;
                            if pw > p_max {
                                p_max = pw;
                                best_bin = b as isize;
                            }
                        }
                        for b in 1..=bin_max_u {
                            let v = buf[ctx.nfft - b];
                            let pw = v.re * v.re + v.im * v.im;
                            if pw > p_max {
                                p_max = pw;
                                best_bin = -(b as isize);
                            }
                        }

                        let mut bin_f = best_bin as f64;
                        if best_bin > -ctx.bin_max && best_bin < ctx.bin_max {
                            let idx_m1 = idx_of(best_bin - 1);
                            let idx_p1 = idx_of(best_bin + 1);
                            let v_m1 = buf[idx_m1];
                            let v_p1 = buf[idx_p1];
                            let p_m1 = (v_m1.re * v_m1.re + v_m1.im * v_m1.im) as f64;
                            let p_0 = p_max as f64;
                            let p_p1 = (v_p1.re * v_p1.re + v_p1.im * v_p1.im) as f64;
                            let denom = p_m1 - 2.0 * p_0 + p_p1;
                            if denom.abs() > 1e-30 {
                                let delta = 0.5 * (p_m1 - p_p1) / denom;
                                bin_f += delta.clamp(-0.5, 0.5);
                            }
                        }
                        let f_hat = bin_f * ctx.fs / (ctx.nfft as f64);

                        push_topk(
                            &mut local_topk,
                            Cand {
                                ti,
                                off,
                                p_max,
                                f_hat,
                            },
                            k_keep,
                        );
                    }
                    local_topk
                },
            )
            .reduce(
                || Vec::with_capacity(k_keep),
                |mut a, b| {
                    for cand in b {
                        push_topk(&mut a, cand, k_keep);
                    }
                    a
                },
            )
    }

    fn refine_and_verify_candidates(
        &self,
        rx_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        ctx: &AcqContext,
        topk: &[Cand],
    ) -> anyhow::Result<Option<(Cand, f64, f64)>> {
        let p = &self.p;
        if topk.is_empty() {
            return Ok(None);
        }

        let sigma2_hat = self.estimate_noise_power_mu(rx_window);
        if !(sigma2_hat.is_finite() && sigma2_hat > 0.0) {
            return Ok(None);
        }

        let e_pre0: f64 = ctx.ref_pre0_conj_by_ti[0]
            .iter()
            .map(|v| v.norm_sqr() as f64)
            .sum();
        let e_pre1: f64 = ctx.ref_pre1_conj_by_ti[0]
            .iter()
            .map(|v| v.norm_sqr() as f64)
            .sum();
        let e_pre: f64 = e_pre0 + e_pre1;

        let g_tmp = gen_code_structured_aes_ctr(
            &self.key,
            ti_min,
            p.n_sym(),
            p.sf(),
            p.domain_u32(),
            p.hop_bw_hz(),
            JitterSpec {
                min_chips: p.jitter_min_chips(),
                span_chips: p.jitter_span_chips(),
            },
        );
        let ref_sym_tmp = self
            .build_symbol_ref_tx_shaped(&g_tmp.c_seq[2 * p.sf()..3 * p.sf()], g_tmp.f_seq_hz[2]);
        let e_sym: f64 = ref_sym_tmp.iter().map(|v| v.norm_sqr() as f64).sum();
        let e_total = e_pre + (p.n_pilot() as f64) * e_sym;
        let gamma = p.gamma_hybrid_mult() * sigma2_hat * e_total;

        let mut ti_needed: Vec<usize> = topk.iter().map(|c| (c.ti - ti_min) as usize).collect();
        ti_needed.sort_unstable();
        ti_needed.dedup();

        let mut pilot_ref_conj_by_ti: Vec<Option<Vec<Vec<Complex32>>>> = vec![None; n_ti];
        let mut sym_pos_samp_by_ti: Vec<Option<Vec<usize>>> = vec![None; n_ti];
        for ti_idx in ti_needed {
            let ti = ti_min + (ti_idx as u64);
            let g = gen_code_structured_aes_ctr(
                &self.key,
                ti,
                p.n_sym(),
                p.sf(),
                p.domain_u32(),
                p.hop_bw_hz(),
                JitterSpec {
                    min_chips: p.jitter_min_chips(),
                    span_chips: p.jitter_span_chips(),
                },
            );
            let mut pilots_this_ti: Vec<Vec<Complex32>> = Vec::with_capacity(p.n_pilot());
            for r in 0..p.n_pilot() {
                let ell = 2 + 5 * r;
                let seg0 = ell * p.sf();
                let ref_p =
                    self.build_symbol_ref_tx_shaped(&g.c_seq[seg0..seg0 + p.sf()], g.f_seq_hz[ell]);
                pilots_this_ti.push(ref_p.iter().map(|v| v.conj()).collect());
            }
            pilot_ref_conj_by_ti[ti_idx] = Some(pilots_this_ti);

            // Symbol start positions (in samples) relative to the frame start for this TI.
            let mut pos: Vec<usize> = Vec::with_capacity(p.n_sym());
            let mut acc_chips: usize = 0;
            let osf = p.osf() as usize;
            for &j_ell in g.j_seq_chips.iter().take(p.n_sym()) {
                pos.push(acc_chips * osf);
                acc_chips = acc_chips.saturating_add(p.sf() + j_ell);
            }
            sym_pos_samp_by_ti[ti_idx] = Some(pos);
        }

        // Spec §4.B.3
        let fine_span_hz: f64 = 2.0;
        let fine_step_hz: f64 = 0.25;
        let fine_steps: i64 = (fine_span_hz / fine_step_hz).round() as i64;

        let pilot_ell: Vec<usize> = (0..p.n_pilot()).map(|r| 2 + 5 * r).collect();

        let best_final: Option<(Cand, f64, f64)> = topk
            .par_iter()
            .map_init(
                || {
                    (
                        vec![Complex32::new(0.0, 0.0); ctx.n_ref_pre_acq],
                        vec![Complex32::new(0.0, 0.0); ctx.l_sym],
                    )
                },
                |state, &cand| {
                    let y_pre = &mut state.0;
                    let ref_rot = &mut state.1;

                    let ti_idx = (cand.ti - ti_min) as usize;
                    let base = ti_idx * ctx.iv_samples;
                    let y0 = base + cand.off;

                    let ref0 = &ctx.ref_pre0_conj_by_ti[ti_idx];
                    let ref1 = &ctx.ref_pre1_conj_by_ti[ti_idx];
                    let delta_sym1 = ctx.pre1_delta_samp_by_ti[ti_idx];
                    let pilots = pilot_ref_conj_by_ti[ti_idx]
                        .as_ref()
                        .expect("Pilot refs should exist for candidate TI");
                    let sym_pos = sym_pos_samp_by_ti[ti_idx]
                        .as_ref()
                        .expect("Symbol positions should exist for candidate TI");

                    // Two-symbol sparse preamble window (sym0, zeros(GI), sym1).
                    y_pre.fill(Complex32::new(0.0, 0.0));
                    for (dst, (r, c)) in y_pre[..ctx.n_ref_pre0]
                        .iter_mut()
                        .zip(rx_window[y0..y0 + ctx.n_ref_pre0].iter().zip(ref0.iter()))
                    {
                        *dst = *r * *c;
                    }
                    let y1 = y0 + delta_sym1;
                    let sym1_start = delta_sym1;
                    let sym1_end = sym1_start + ctx.n_ref_pre0;
                    debug_assert!(sym1_end <= y_pre.len());
                    for (dst, (r, c)) in y_pre[sym1_start..sym1_end]
                        .iter_mut()
                        .zip(rx_window[y1..y1 + ctx.n_ref_pre0].iter().zip(ref1.iter()))
                    {
                        *dst = *r * *c;
                    }

                    let mut vpre_best = f64::NEG_INFINITY;
                    let mut f_best = cand.f_hat;
                    for i in -fine_steps..=fine_steps {
                        let f = cand.f_hat + (i as f64) * fine_step_hz;
                        let v = self.energy_of_seq_with_cfo(&y_pre[..], p.fs_hz(), f);
                        if v > vpre_best {
                            vpre_best = v;
                            f_best = f;
                        }
                    }

                    let fs_f32 = p.fs_hz() as f32;
                    let dphi = -2.0 * std::f32::consts::PI * (f_best as f32) / fs_f32;
                    let w = Complex32::from_polar(1.0, dphi);

                    let mut vpil: f64 = 0.0;
                    for r in 0..p.n_pilot() {
                        let refc = &pilots[r];
                        debug_assert_eq!(refc.len(), ctx.l_sym);
                        let delta_pos = sym_pos[pilot_ell[r]];

                        let mut ph = Complex32::new(1.0, 0.0);
                        for (dst, c) in ref_rot.iter_mut().zip(refc.iter()) {
                            *dst = *c * ph;
                            ph *= w;
                        }

                        let mut best_e = 0.0f64;
                        for d in -ctx.pilot_timing_win..=ctx.pilot_timing_win {
                            let start = (y0 as isize) + (delta_pos as isize) + d;
                            if start < 0 {
                                continue;
                            }
                            let start = start as usize;
                            let e = self.corr_energy_no_cfo(rx_window, start, &ref_rot[..]);
                            if e > best_e {
                                best_e = e;
                            }
                        }
                        vpil += best_e;
                    }

                    let lambda = (vpre_best + vpil).max(0.0);
                    if lambda > gamma {
                        Some((cand, lambda, f_best))
                    } else {
                        None
                    }
                },
            )
            .filter_map(|v| v)
            .reduce_with(
                |a, b| match a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) {
                    Ordering::Greater => a,
                    Ordering::Less => b,
                    Ordering::Equal => {
                        if (a.0.ti, a.0.off) <= (b.0.ti, b.0.off) {
                            a
                        } else {
                            b
                        }
                    }
                },
            );

        Ok(best_final)
    }

    fn find_rake_fingers(
        &self,
        rx_window: &[Complex32],
        ti_min: u64,
        n_finger: usize,
        ctx: &AcqContext,
        best: Cand,
        f_fine: f64,
    ) -> anyhow::Result<AcqResult> {
        let p = &self.p;
        let best_ti_idx = (best.ti - ti_min) as usize;
        let base = best_ti_idx * ctx.iv_samples;
        let ref_pre0_conj = &ctx.ref_pre0_conj_by_ti[best_ti_idx];
        let mut ref_pre0_rot = vec![Complex32::new(0.0, 0.0); ctx.n_ref_pre0];
        {
            let fs_f32 = p.fs_hz() as f32;
            let dphi = -2.0 * std::f32::consts::PI * (f_fine as f32) / fs_f32;
            let w = Complex32::from_polar(1.0, dphi);
            let mut ph = Complex32::new(1.0, 0.0);
            for (dst, c) in ref_pre0_rot.iter_mut().zip(ref_pre0_conj.iter()) {
                *dst = *c * ph;
                ph *= w;
            }
        }

        // Also build a symbol-1 preamble reference (with the symbol-0 guard/noise gap in between)
        // to improve timing discrimination under multipath.
        let delta_sym1 = ctx.pre1_delta_samp_by_ti[best_ti_idx];
        let ref_pre1_conj = &ctx.ref_pre1_conj_by_ti[best_ti_idx];
        let mut ref_pre1_rot = vec![Complex32::new(0.0, 0.0); ctx.n_ref_pre0];
        {
            let fs_f32 = p.fs_hz() as f32;
            let dphi = -2.0 * std::f32::consts::PI * (f_fine as f32) / fs_f32;
            let w = Complex32::from_polar(1.0, dphi);
            let mut ph = Complex32::new(1.0, 0.0);
            for (dst, c) in ref_pre1_rot.iter_mut().zip(ref_pre1_conj.iter()) {
                *dst = *c * ph;
                ph *= w;
            }
        }

        // Spec §4.B.3
        // Coordinate system note: all `off`/`n0`/`finger_offsets` values are nonnegative sample
        // offsets from the start of the acquired IV epoch (the `ti_hat` boundary). The nominal
        // "symmetric" +/- search window is therefore clipped to `[0, max_off]` at the low end.
        let center = best.off as isize;
        let half = ctx.rake_search_half_samples as isize;
        let mut off_lo: isize = center - half;
        if off_lo < 0 {
            off_lo = 0;
        }
        let mut off_hi: isize = center + half;
        let max_start = rx_window.len().saturating_sub(ctx.n_ref_pre0);
        let max_off = max_start.saturating_sub(base) as isize;
        if off_hi > max_off {
            off_hi = max_off;
        }

        let want = n_finger.clamp(1, 16);
        let mut corr: Vec<(usize, f64)> = Vec::new();
        if off_hi >= off_lo {
            let n_hyp = (off_hi - off_lo + 1) as usize;
            if n_hyp >= 256 {
                corr = (0..n_hyp)
                    .into_par_iter()
                    .map(|i| {
                        let off = (off_lo as usize) + i;
                        let y0 = base + off;
                        let mut e = self.corr_energy_no_cfo(rx_window, y0, &ref_pre0_rot[..]);
                        let y1 = y0.saturating_add(delta_sym1);
                        if y1 + ref_pre1_rot.len() <= rx_window.len() {
                            e += self.corr_energy_no_cfo(rx_window, y1, &ref_pre1_rot[..]);
                        }
                        (off, e)
                    })
                    .collect();
            } else {
                corr = Vec::with_capacity(n_hyp);
                for off in (off_lo as usize)..=(off_hi as usize) {
                    let y0 = base + off;
                    let mut e = self.corr_energy_no_cfo(rx_window, y0, &ref_pre0_rot[..]);
                    let y1 = y0.saturating_add(delta_sym1);
                    if y1 + ref_pre1_rot.len() <= rx_window.len() {
                        e += self.corr_energy_no_cfo(rx_window, y1, &ref_pre1_rot[..]);
                    }
                    corr.push((off, e));
                }
            }
        }
        corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let n0 = corr.first().map(|v| v.0).unwrap_or(best.off);

        let min_sep = (p.osf() as usize).max(1);
        // Keep the best (strongest) finger first; downstream code treats this as the frame's
        // nominal start for de-hopping and symbol scheduling.
        let mut finger_offsets: Vec<usize> = vec![n0];
        for (off, _) in &corr {
            // Multipath delays are nonnegative relative to the strongest preamble peak.
            if *off < n0 {
                continue;
            }
            if finger_offsets.iter().all(|&x| x.abs_diff(*off) >= min_sep) {
                finger_offsets.push(*off);
            }
            if finger_offsets.len() >= want {
                break;
            }
        }
        // Do not sort: preserve `n0` as the first element.

        Ok(AcqResult {
            ti_hat: best.ti,
            n0,
            cfo_hat_hz: f_fine,
            finger_offsets,
            p_max: best.p_max,
        })
    }

    fn acquire_fft_window_impl(
        &self,
        rx_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        n_finger: usize,
    ) -> anyhow::Result<Option<AcqResult>> {
        // Spec §4.B
        let ctx = self.prepare_acq_context(rx_window, ti_min, n_ti)?;
        let topk = self.find_coarse_candidates(rx_window, ti_min, n_ti, &ctx);
        let Some((best, lambda_best, f_fine)) =
            self.refine_and_verify_candidates(rx_window, ti_min, n_ti, &ctx, &topk)?
        else {
            return Ok(None);
        };
        let mut out = self.find_rake_fingers(rx_window, ti_min, n_finger, &ctx, best, f_fine)?;
        out.p_max = lambda_best as f32;
        Ok(Some(out))
    }

    /// Blind acquisition over a raw baseband window (Specification §4.B).
    ///
    /// `rx_raw_window` must contain `n_ti` consecutive IV intervals plus enough guard samples for:
    /// - candidate verification (preamble + pilots)
    /// - RAKE finger search around the best preamble offset
    pub fn acquire_fft_raw_window(
        &self,
        rx_raw_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        n_finger: usize,
    ) -> anyhow::Result<Option<AcqResult>> {
        self.acquire_fft_window_impl(rx_raw_window, ti_min, n_ti, n_finger)
    }
}
