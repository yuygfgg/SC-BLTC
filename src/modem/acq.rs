use super::{AcqResult, ScBltcModem};
use crate::crypto::gen_code_aes_ctr;
use crate::modem::util::pulse_shape_chips;
use num_complex::Complex32;
use rayon::prelude::*;
use std::cmp::Ordering;

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
        let dphi = (-2.0 * std::f32::consts::PI * (cfo_hz as f32) / fs) as f32;
        let w = Complex32::from_polar(1.0, dphi);
        let mut ph = Complex32::new(1.0, 0.0);
        let mut acc = Complex32::new(0.0, 0.0);
        for v in y {
            acc += *v * ph;
            ph *= w;
        }
        (acc.norm_sqr() as f64).max(0.0)
    }

    fn build_pilot_ref(
        &self,
        c_seq: &[i8],
        ell: usize,
        matched: bool,
    ) -> anyhow::Result<Vec<Complex32>> {
        let p = &self.p;
        if (ell + 1) * p.sf > c_seq.len() {
            anyhow::bail!("pilot ref build out of range");
        }
        let chips = &c_seq[ell * p.sf..(ell + 1) * p.sf];
        let x = pulse_shape_chips(chips, &self.rrc, p.osf as usize);
        if matched {
            Ok(self.rrc.filter_same(&x))
        } else {
            Ok(x)
        }
    }

    fn acquire_fft_window_impl(
        &self,
        rx_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        n_finger: usize,
        matched: bool,
    ) -> anyhow::Result<Option<AcqResult>> {
        // Spec §4.B
        let p = &self.p;
        let iv_samples = ((p.fs_hz as f64) * p.iv_res_s).round() as usize;
        if iv_samples == 0 {
            anyhow::bail!("invalid iv_samples");
        }
        if p.rake_search_half_s < 0.0 {
            anyhow::bail!("invalid rake_search_half_s");
        }
        let rake_search_half_samples = ((p.fs_hz as f64) * p.rake_search_half_s).round() as usize;

        let l_sym = p.chip_samples();
        let n_ref_pre = p.n_pre * l_sym;

        if p.nfft_acq < n_ref_pre {
            anyhow::bail!("Params.nfft_acq must be >= Npre*SF*OSF");
        }

        let ell_last_pilot = 2 + 5 * (p.n_pilot.saturating_sub(1));
        let last_pilot_end = (ell_last_pilot + 1) * l_sym;
        let pilot_timing_win: isize = 32;
        let verify_need = last_pilot_end + (pilot_timing_win as usize);

        let need = n_ti
            .checked_mul(iv_samples)
            .and_then(|v| v.checked_add(iv_samples))
            .and_then(|v| v.checked_add(rake_search_half_samples))
            .and_then(|v| v.checked_add(verify_need))
            .ok_or_else(|| anyhow::anyhow!("window_size_overflow"))?;
        if rx_window.len() < need {
            anyhow::bail!(
                "insufficient rx_window: need >= {}, got {}",
                need,
                rx_window.len()
            );
        }

        let fs = p.fs_hz as f64;
        let nfft = p.nfft_acq;

        let search_hz = p.cfo_search_hz.abs().min(0.5 * fs);
        let bin_max = ((search_hz * (nfft as f64)) / fs).floor() as isize;
        let bin_max = bin_max.clamp(1, (nfft as isize) / 2 - 1);

        let fft = &self.fft_acq;
        debug_assert_eq!(fft.len(), nfft);
        let scratch_len = self.fft_acq_scratch_len;

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

        let ref_pre_conj_by_ti: Vec<Vec<Complex32>> = (0..n_ti)
            .map(|ti_idx| {
                let ti = ti_min + (ti_idx as u64);
                let ref_pre = if matched {
                    self.make_ref_preamble_matched(ti)
                } else {
                    self.make_ref_preamble_tx_shaped(ti)
                };
                debug_assert_eq!(ref_pre.len(), n_ref_pre);
                ref_pre.iter().map(|v| v.conj()).collect()
            })
            .collect();

        let k_keep: usize = 50;
        let bin_max_u = bin_max as usize;
        let idx_of = |b: isize| -> usize {
            if b >= 0 {
                b as usize
            } else {
                nfft - ((-b) as usize)
            }
        };

        let off_chunk: usize = 4;
        let topk: Vec<Cand> = (0..n_ti)
            .into_par_iter()
            .flat_map_iter(|ti_idx| {
                (0..iv_samples)
                    .step_by(off_chunk)
                    .map(move |off0| (ti_idx, off0, (off0 + off_chunk).min(iv_samples)))
            })
            .map_init(
                || {
                    (
                        vec![Complex32::new(0.0, 0.0); nfft],
                        vec![Complex32::new(0.0, 0.0); scratch_len],
                    )
                },
                |state, (ti_idx, off0, off1)| {
                    let buf = &mut state.0;
                    let scratch = &mut state.1;
                    let ti = ti_min + (ti_idx as u64);
                    let base = ti_idx * iv_samples;
                    let refc = &ref_pre_conj_by_ti[ti_idx];
                    let mut local_topk: Vec<Cand> = Vec::with_capacity(k_keep);

                    for off in off0..off1 {
                        let y0 = base + off;
                        let rx_slice = &rx_window[y0..y0 + n_ref_pre];
                        for (dst, (r, c)) in buf[..n_ref_pre]
                            .iter_mut()
                            .zip(rx_slice.iter().zip(refc.iter()))
                        {
                            *dst = *r * *c;
                        }
                        buf[n_ref_pre..].fill(Complex32::new(0.0, 0.0));

                        fft.process_with_scratch(&mut buf[..], &mut scratch[..]);

                        let mut p_max = f32::NEG_INFINITY;
                        let mut best_bin: isize = 0;

                        for b in 0..=bin_max_u {
                            let v = buf[b];
                            let pw = v.re * v.re + v.im * v.im;
                            if pw > p_max {
                                p_max = pw;
                                best_bin = b as isize;
                            }
                        }
                        for b in 1..=bin_max_u {
                            let v = buf[nfft - b];
                            let pw = v.re * v.re + v.im * v.im;
                            if pw > p_max {
                                p_max = pw;
                                best_bin = -(b as isize);
                            }
                        }

                        let mut bin_f = best_bin as f64;
                        if best_bin > -bin_max && best_bin < bin_max {
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
                        let f_hat = bin_f * fs / (nfft as f64);

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
            );

        if topk.is_empty() {
            return Ok(None);
        }

        let sigma2_hat = self.estimate_noise_power_mu(rx_window);
        if !(sigma2_hat.is_finite() && sigma2_hat > 0.0) {
            return Ok(None);
        }

        let e_pre: f64 = ref_pre_conj_by_ti[0]
            .iter()
            .map(|v| v.norm_sqr() as f64)
            .sum();

        let c_seq_tmp = gen_code_aes_ctr(&self.key, ti_min, p.frame_chips(), p.domain_u32);
        let ref_sym_tmp = self.build_pilot_ref(&c_seq_tmp, 2, matched)?;
        let e_sym: f64 = ref_sym_tmp.iter().map(|v| v.norm_sqr() as f64).sum();
        let e_total = e_pre + (p.n_pilot as f64) * e_sym;
        let gamma = p.gamma_hybrid_mult * sigma2_hat * e_total;

        let mut ti_needed: Vec<usize> = topk.iter().map(|c| (c.ti - ti_min) as usize).collect();
        ti_needed.sort_unstable();
        ti_needed.dedup();

        let mut pilot_ref_conj_by_ti: Vec<Option<Vec<Vec<Complex32>>>> = vec![None; n_ti];
        for ti_idx in ti_needed {
            let ti = ti_min + (ti_idx as u64);
            let c_seq = gen_code_aes_ctr(&self.key, ti, p.frame_chips(), p.domain_u32);
            let mut pilots_this_ti: Vec<Vec<Complex32>> = Vec::with_capacity(p.n_pilot);
            for r in 0..p.n_pilot {
                let ell = 2 + 5 * r;
                let ref_p = self.build_pilot_ref(&c_seq, ell, matched)?;
                pilots_this_ti.push(ref_p.iter().map(|v| v.conj()).collect());
            }
            pilot_ref_conj_by_ti[ti_idx] = Some(pilots_this_ti);
        }

        // Spec §4.B.3
        let fine_span_hz: f64 = 2.0;
        let fine_step_hz: f64 = 0.25;
        let fine_steps: i64 = (fine_span_hz / fine_step_hz).round() as i64;

        let delta_pos_by_pilot: Vec<usize> = (0..p.n_pilot).map(|r| (2 + 5 * r) * l_sym).collect();

        let best_final: Option<(Cand, f64, f64)> = topk
            .par_iter()
            .map_init(
                || {
                    (
                        vec![Complex32::new(0.0, 0.0); n_ref_pre],
                        vec![Complex32::new(0.0, 0.0); l_sym],
                    )
                },
                |state, &cand| {
                    let y_pre = &mut state.0;
                    let ref_rot = &mut state.1;

                    let ti_idx = (cand.ti - ti_min) as usize;
                    let base = ti_idx * iv_samples;
                    let y0 = base + cand.off;

                    let ref_pre_conj = &ref_pre_conj_by_ti[ti_idx];
                    let pilots = pilot_ref_conj_by_ti[ti_idx]
                        .as_ref()
                        .expect("missing pilot refs for candidate TI");

                    for (dst, (r, c)) in y_pre.iter_mut().zip(
                        rx_window[y0..y0 + n_ref_pre]
                            .iter()
                            .zip(ref_pre_conj.iter()),
                    ) {
                        *dst = *r * *c;
                    }

                    let mut vpre_best = f64::NEG_INFINITY;
                    let mut f_best = cand.f_hat;
                    for i in -fine_steps..=fine_steps {
                        let f = cand.f_hat + (i as f64) * fine_step_hz;
                        let v = self.energy_of_seq_with_cfo(&y_pre[..], p.fs_hz, f);
                        if v > vpre_best {
                            vpre_best = v;
                            f_best = f;
                        }
                    }

                    let fs_f32 = p.fs_hz as f32;
                    let dphi = (-2.0 * std::f32::consts::PI * (f_best as f32) / fs_f32) as f32;
                    let w = Complex32::from_polar(1.0, dphi);

                    let mut vpil: f64 = 0.0;
                    for r in 0..p.n_pilot {
                        let refc = &pilots[r];
                        debug_assert_eq!(refc.len(), l_sym);
                        let delta_pos = delta_pos_by_pilot[r];

                        let mut ph = Complex32::new(1.0, 0.0);
                        for (dst, c) in ref_rot.iter_mut().zip(refc.iter()) {
                            *dst = *c * ph;
                            ph *= w;
                        }

                        let mut best_e = 0.0f64;
                        for d in -pilot_timing_win..=pilot_timing_win {
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

        let Some((best, lambda_best, f_fine)) = best_final else {
            return Ok(None);
        };

        let best_ti_idx = (best.ti - ti_min) as usize;
        let base = best_ti_idx * iv_samples;
        let ref_pre_conj = &ref_pre_conj_by_ti[best_ti_idx];
        let mut ref_pre_rot = vec![Complex32::new(0.0, 0.0); n_ref_pre];
        {
            let fs_f32 = p.fs_hz as f32;
            let dphi = (-2.0 * std::f32::consts::PI * (f_fine as f32) / fs_f32) as f32;
            let w = Complex32::from_polar(1.0, dphi);
            let mut ph = Complex32::new(1.0, 0.0);
            for (dst, c) in ref_pre_rot.iter_mut().zip(ref_pre_conj.iter()) {
                *dst = *c * ph;
                ph *= w;
            }
        }

        // Spec §4.B.3
        // Coordinate system note: all `off`/`n0`/`finger_offsets` values are nonnegative sample
        // offsets from the start of the acquired IV epoch (the `ti_hat` boundary). The nominal
        // "symmetric" +/- search window is therefore clipped to `[0, max_off]` at the low end.
        let center = best.off as isize;
        let half = rake_search_half_samples as isize;
        let mut off_lo: isize = center - half;
        if off_lo < 0 {
            off_lo = 0;
        }
        let mut off_hi: isize = center + half;
        let max_start = rx_window.len().saturating_sub(n_ref_pre);
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
                        let e = self.corr_energy_no_cfo(rx_window, y0, &ref_pre_rot[..]);
                        (off, e)
                    })
                    .collect();
            } else {
                corr = Vec::with_capacity(n_hyp);
                for off in (off_lo as usize)..=(off_hi as usize) {
                    let y0 = base + off;
                    let e = self.corr_energy_no_cfo(rx_window, y0, &ref_pre_rot[..]);
                    corr.push((off, e));
                }
            }
        }
        corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let n0 = corr.first().map(|v| v.0).unwrap_or(best.off);

        let min_sep = (p.osf as usize).max(1);
        let mut finger_offsets: Vec<usize> = vec![n0];
        for (off, _) in &corr {
            if finger_offsets.iter().all(|&x| x.abs_diff(*off) >= min_sep) {
                finger_offsets.push(*off);
            }
            if finger_offsets.len() >= want {
                break;
            }
        }
        finger_offsets.sort_unstable();
        finger_offsets.dedup();

        Ok(Some(AcqResult {
            ti_hat: best.ti,
            n0,
            cfo_hat_hz: f_fine,
            finger_offsets,
            p_max: lambda_best as f32,
        }))
    }

    pub fn acquire_fft_raw_window(
        &self,
        rx_raw_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        p_fa_total: f64,
        n_finger: usize,
    ) -> anyhow::Result<Option<AcqResult>> {
        let _ = p_fa_total;
        self.acquire_fft_window_impl(rx_raw_window, ti_min, n_ti, n_finger, false)
    }

    pub fn acquire_fft_matched_window(
        &self,
        y_matched_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        p_fa_total: f64,
        n_finger: usize,
    ) -> anyhow::Result<Option<AcqResult>> {
        let _ = p_fa_total;
        self.acquire_fft_window_impl(y_matched_window, ti_min, n_ti, n_finger, true)
    }
}
