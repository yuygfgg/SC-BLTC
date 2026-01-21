use super::{AcqResult, ScBltcModem};
use num_complex::Complex32;
use rustfft::FftPlanner;
use std::cmp::Ordering;

impl ScBltcModem {
    pub fn acquire_fft_raw_window(
        &self,
        rx_raw_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        p_fa_total: f64,
        n_finger: usize,
    ) -> anyhow::Result<Option<AcqResult>> {
        // Spec §4.B.
        let p = &self.p;
        if !(p_fa_total > 0.0 && p_fa_total < 1.0) {
            anyhow::bail!("p_fa_total must be in (0,1)");
        }
        let iv_samples = ((p.fs_hz as f64) * p.iv_res_s).round() as usize;
        if iv_samples == 0 {
            anyhow::bail!("invalid iv_samples");
        }
        let n_ref = p.chip_samples();
        if p.nfft_acq < n_ref {
            anyhow::bail!("Params.nfft_acq must be >= SF*OSF");
        }
        let need = n_ti
            .checked_mul(iv_samples)
            .and_then(|v| v.checked_add(n_ref))
            .ok_or_else(|| anyhow::anyhow!("window_size_overflow"))?;
        if rx_raw_window.len() < need {
            anyhow::bail!(
                "insufficient rx_raw_window: need >= {}, got {}",
                need,
                rx_raw_window.len()
            );
        }

        let fs = p.fs_hz as f64;
        let nfft = p.nfft_acq;

        let search_hz = p.cfo_search_hz.abs().min(0.5 * fs);
        let bin_max = ((search_hz * (nfft as f64)) / fs).floor() as isize;
        let bin_max = bin_max.clamp(1, (nfft as isize) / 2 - 1);
        let n_bins = (2 * bin_max + 1) as usize;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);

        #[derive(Clone)]
        struct Cand {
            ti: u64,
            off: usize,
            p_max: f32,
            f_hat: f64,
        }

        let mut best: Option<Cand> = None;
        let mut buf = vec![Complex32::new(0.0, 0.0); nfft];
        let mut pows = vec![0f32; n_bins];

        for ti_idx in 0..n_ti {
            let ti = ti_min + (ti_idx as u64);
            let base = ti_idx * iv_samples;

            let ref0 = self.make_ref_preamble_tx_shaped(ti);
            debug_assert_eq!(ref0.len(), n_ref);
            let refc: Vec<Complex32> = ref0.iter().map(|v| v.conj()).collect();

            for off in 0..iv_samples {
                let y0 = base + off;
                for k in 0..n_ref {
                    buf[k] = rx_raw_window[y0 + k] * refc[k];
                }
                buf[n_ref..].fill(Complex32::new(0.0, 0.0));

                fft.process(&mut buf);

                let mut p_max = f32::NEG_INFINITY;
                let mut best_bin: isize = 0;
                for (i, b) in (-bin_max..=bin_max).enumerate() {
                    let idx = (b.rem_euclid(nfft as isize)) as usize;
                    let pw = buf[idx].norm_sqr();
                    pows[i] = pw;
                    if pw > p_max {
                        p_max = pw;
                        best_bin = b;
                    }
                }

                let mid = n_bins / 2;
                let (_, med, _) = pows.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                let med = *med as f64;
                let mu = med / std::f64::consts::LN_2;
                let gamma = mu * ((n_bins as f64) / p_fa_total).ln();

                let min_psr = 40.0;
                if (p_max as f64) > gamma && (p_max as f64) > (med * min_psr) {
                    let mut bin_f = best_bin as f64;
                    if best_bin > -bin_max && best_bin < bin_max {
                        let idx_m1 = ((best_bin - 1).rem_euclid(nfft as isize)) as usize;
                        let idx_p1 = ((best_bin + 1).rem_euclid(nfft as isize)) as usize;
                        let p_m1 = buf[idx_m1].norm_sqr() as f64;
                        let p_0 = p_max as f64;
                        let p_p1 = buf[idx_p1].norm_sqr() as f64;
                        let denom = p_m1 - 2.0 * p_0 + p_p1;
                        if denom.abs() > 1e-30 {
                            let delta = 0.5 * (p_m1 - p_p1) / denom;
                            bin_f += delta.clamp(-0.5, 0.5);
                        }
                    }
                    let f_hat = bin_f * fs / (nfft as f64);
                    let cand = Cand {
                        ti,
                        off,
                        p_max,
                        f_hat,
                    };
                    if best.as_ref().map(|b| cand.p_max > b.p_max).unwrap_or(true) {
                        best = Some(cand);
                    }
                }
            }
        }

        let Some(best) = best else {
            return Ok(None);
        };

        let best_ti_idx = (best.ti - ti_min) as usize;
        let base = best_ti_idx * iv_samples;
        let ref0 = self.make_ref_preamble_tx_shaped(best.ti);
        debug_assert_eq!(ref0.len(), n_ref);
        let refc: Vec<Complex32> = ref0.iter().map(|v| v.conj()).collect();

        let want = n_finger.clamp(1, 16);
        let mut corr: Vec<(usize, f32)> = Vec::with_capacity(iv_samples);
        let dphi = -2.0 * std::f32::consts::PI * (best.f_hat as f32) / (fs as f32);
        for off in 0..iv_samples {
            let mut acc = Complex32::new(0.0, 0.0);
            let mut phi = 0.0f32;
            for k in 0..n_ref {
                acc += rx_raw_window[base + off + k] * refc[k] * Complex32::from_polar(1.0, phi);
                phi += dphi;
            }
            corr.push((off, acc.norm_sqr()));
        }
        corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let n0 = corr.first().map(|v| v.0).unwrap_or(best.off);
        let p_max_refined = corr.first().map(|v| v.1).unwrap_or(best.p_max);

        let mut finger_offsets: Vec<usize> = Vec::new();
        for (off, _) in corr {
            if !finger_offsets.contains(&off) {
                finger_offsets.push(off);
            }
            if finger_offsets.len() >= want {
                break;
            }
        }
        if !finger_offsets.contains(&n0) {
            finger_offsets.insert(0, n0);
        }
        finger_offsets.sort_unstable();

        Ok(Some(AcqResult {
            ti_hat: best.ti,
            n0,
            cfo_coarse_hz: best.f_hat,
            finger_offsets,
            p_max: p_max_refined,
        }))
    }

    pub fn acquire_fft_matched_window(
        &self,
        y_matched_window: &[Complex32],
        ti_min: u64,
        n_ti: usize,
        p_fa_total: f64,
        n_finger: usize,
    ) -> anyhow::Result<Option<AcqResult>> {
        // Spec §4.B.
        let p = &self.p;
        if !(p_fa_total > 0.0 && p_fa_total < 1.0) {
            anyhow::bail!("p_fa_total must be in (0,1)");
        }
        let iv_samples = ((p.fs_hz as f64) * p.iv_res_s).round() as usize;
        if iv_samples == 0 {
            anyhow::bail!("invalid iv_samples");
        }
        let n_ref = p.chip_samples();
        if p.nfft_acq < n_ref {
            anyhow::bail!("Params.nfft_acq must be >= SF*OSF");
        }
        let need = n_ti
            .checked_mul(iv_samples)
            .and_then(|v| v.checked_add(n_ref))
            .ok_or_else(|| anyhow::anyhow!("window_size_overflow"))?;
        if y_matched_window.len() < need {
            anyhow::bail!(
                "insufficient y_matched_window: need >= {}, got {}",
                need,
                y_matched_window.len()
            );
        }

        let fs = p.fs_hz as f64;
        let nfft = p.nfft_acq;

        let search_hz = p.cfo_search_hz.abs().min(0.5 * fs);
        let bin_max = ((search_hz * (nfft as f64)) / fs).floor() as isize;
        let bin_max = bin_max.clamp(1, (nfft as isize) / 2 - 1);
        let n_bins = (2 * bin_max + 1) as usize;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);

        #[derive(Clone)]
        struct Cand {
            ti: u64,
            off: usize,
            p_max: f32,
            f_hat: f64,
        }

        let mut best: Option<Cand> = None;
        let mut buf = vec![Complex32::new(0.0, 0.0); nfft];
        let mut pows = vec![0f32; n_bins];

        for ti_idx in 0..n_ti {
            let ti = ti_min + (ti_idx as u64);
            let base = ti_idx * iv_samples;

            let ref0 = self.make_ref_preamble_matched(ti);
            debug_assert_eq!(ref0.len(), n_ref);
            let refc: Vec<Complex32> = ref0.iter().map(|v| v.conj()).collect();

            let pow_mean = ref0.iter().map(|v| v.norm_sqr()).sum::<f32>() / (n_ref as f32);
            let denom_min = (pow_mean * 1e-3).max(1e-12);
            let inv_pow: Vec<f32> = ref0
                .iter()
                .map(|v| 1.0 / v.norm_sqr().max(denom_min))
                .collect();

            for off in 0..iv_samples {
                let y0 = base + off;
                for k in 0..n_ref {
                    buf[k] = y_matched_window[y0 + k] * refc[k] * inv_pow[k];
                }
                buf[n_ref..].fill(Complex32::new(0.0, 0.0));

                fft.process(&mut buf);

                let mut p_max = f32::NEG_INFINITY;
                let mut best_bin: isize = 0;
                for (i, b) in (-bin_max..=bin_max).enumerate() {
                    let idx = (b.rem_euclid(nfft as isize)) as usize;
                    let pw = buf[idx].norm_sqr();
                    pows[i] = pw;
                    if pw > p_max {
                        p_max = pw;
                        best_bin = b;
                    }
                }

                let mid = n_bins / 2;
                let (_, med, _) = pows.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                let med = *med as f64;
                let mu = med / std::f64::consts::LN_2;
                let gamma = mu * ((n_bins as f64) / p_fa_total).ln();

                let min_psr = 40.0;
                if (p_max as f64) > gamma
                    && (p_max as f64) > (med * min_psr)
                    && (p_max as f64) > 1e-4
                {
                    let mut bin_f = best_bin as f64;
                    if best_bin > -bin_max && best_bin < bin_max {
                        let idx_m1 = ((best_bin - 1).rem_euclid(nfft as isize)) as usize;
                        let idx_p1 = ((best_bin + 1).rem_euclid(nfft as isize)) as usize;
                        let p_m1 = buf[idx_m1].norm_sqr() as f64;
                        let p_0 = p_max as f64;
                        let p_p1 = buf[idx_p1].norm_sqr() as f64;
                        let denom = p_m1 - 2.0 * p_0 + p_p1;
                        if denom.abs() > 1e-30 {
                            let delta = 0.5 * (p_m1 - p_p1) / denom;
                            bin_f += delta.clamp(-0.5, 0.5);
                        }
                    }
                    let f_hat = bin_f * fs / (nfft as f64);
                    let cand = Cand {
                        ti,
                        off,
                        p_max,
                        f_hat,
                    };
                    if best.as_ref().map(|b| cand.p_max > b.p_max).unwrap_or(true) {
                        best = Some(cand);
                    }
                }
            }
        }

        let Some(best) = best else {
            return Ok(None);
        };

        let best_ti_idx = (best.ti - ti_min) as usize;
        let base = best_ti_idx * iv_samples;

        let ref0 = self.make_ref_preamble_matched(best.ti);
        debug_assert_eq!(ref0.len(), n_ref);
        let refc: Vec<Complex32> = ref0.iter().map(|v| v.conj()).collect();
        let pow_mean = ref0.iter().map(|v| v.norm_sqr()).sum::<f32>() / (n_ref as f32);
        let denom_min = (pow_mean * 1e-3).max(1e-12);
        let inv_pow: Vec<f32> = ref0
            .iter()
            .map(|v| 1.0 / v.norm_sqr().max(denom_min))
            .collect();

        let want = n_finger.clamp(1, 16);
        let mut corr: Vec<(usize, f32)> = Vec::with_capacity(iv_samples);
        let dphi = (-2.0 * std::f32::consts::PI * (best.f_hat as f32) / (fs as f32)) as f32;
        for off in 0..iv_samples {
            let mut acc = Complex32::new(0.0, 0.0);
            let mut phi = 0.0f32;
            for k in 0..n_ref {
                acc += y_matched_window[base + off + k]
                    * refc[k]
                    * inv_pow[k]
                    * Complex32::from_polar(1.0, phi);
                phi += dphi;
            }
            corr.push((off, acc.norm_sqr()));
        }
        corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let n0 = corr.first().map(|v| v.0).unwrap_or(best.off);
        let p_max_refined = corr.first().map(|v| v.1).unwrap_or(best.p_max);

        let mut finger_offsets: Vec<usize> = Vec::new();
        for (off, _) in corr {
            if !finger_offsets.contains(&off) {
                finger_offsets.push(off);
            }
            if finger_offsets.len() >= want {
                break;
            }
        }
        if !finger_offsets.contains(&n0) {
            finger_offsets.insert(0, n0);
        }
        finger_offsets.sort_unstable();

        Ok(Some(AcqResult {
            ti_hat: best.ti,
            n0,
            cfo_coarse_hz: best.f_hat,
            finger_offsets,
            p_max: p_max_refined,
        }))
    }
}
