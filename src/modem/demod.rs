use super::util::{build_carrier_rot_chips, derotate_cfo_in_place, is_pilot, pll_step, wrap_pm_pi};
use super::{DecodeMeta, ScBltcModem};
use crate::crypto::gen_code_aes_ctr;
use crate::frame::parse_u_bits;
use crate::ldpc::ldpc_decode_to_u256_from_llr;
use crate::tracking::{design_2nd_order_loop, EarlyLateDll};
use crate::walsh::{fht1024_in_place, walsh_row};
use num_complex::Complex32;

impl ScBltcModem {
    /// Spec §4.C–§4.D.
    pub fn demod_decode_raw(
        &self,
        rx_samples: &[Complex32],
        ti_tx: u64,
        frame_start_sample: usize,
        n_offset_total: &[usize],
        cfo_hz: f64,
        maxiter: usize,
    ) -> anyhow::Result<(Option<Vec<u8>>, DecodeMeta)> {
        let p = &self.p;
        let mut x = rx_samples.to_vec();
        derotate_cfo_in_place(&mut x, p.fs_hz, cfo_hz);
        let y = self.rrc.filter_same(&x);
        self.demod_decode_matched(&y, ti_tx, frame_start_sample, n_offset_total, 0.0, maxiter)
    }

    /// Spec §4.C–§4.D.
    pub fn demod_decode_matched(
        &self,
        y_matched: &[Complex32],
        ti_tx: u64,
        frame_start_sample: usize,
        n_offset_total: &[usize],
        cfo_hz: f64,
        maxiter: usize,
    ) -> anyhow::Result<(Option<Vec<u8>>, DecodeMeta)> {
        let p = &self.p;
        if n_offset_total.is_empty() {
            return Ok((None, DecodeMeta::error("no_offsets")));
        }

        let mut y = y_matched.to_vec();
        derotate_cfo_in_place(&mut y, p.fs_hz, cfo_hz);

        let c_seq = gen_code_aes_ctr(&self.key, ti_tx, p.frame_chips(), p.domain_u32);

        let cascade_delay = 2 * self.rrc.delay();
        let frame_start = frame_start_sample as f64;

        let sample_linear = |pos: &[f64]| -> Option<Vec<Complex32>> {
            let mut out = Vec::with_capacity(pos.len());
            for &pp in pos {
                if pp < 0.0 {
                    return None;
                }
                let i0 = pp.floor() as isize;
                let a = (pp - (i0 as f64)) as f32;
                let i0u = i0 as usize;
                if (i0u + 1) >= y.len() {
                    return None;
                }
                let v = y[i0u] * (1.0 - a) + y[i0u + 1] * a;
                out.push(v);
            }
            Some(out)
        };

        let n_finger = n_offset_total.len().min(3);
        let mut t_sym0 = vec![0f64; n_finger];
        for (i, &off) in n_offset_total.iter().take(n_finger).enumerate() {
            t_sym0[i] = frame_start + (off as f64) + (cascade_delay as f64);
        }
        let chip_idx: Vec<f64> = (0..p.sf).map(|i| i as f64).collect();

        let demask_symbol = |y_chips: &[Complex32], ell: usize| -> Vec<Complex32> {
            let seg0 = ell * p.sf;
            let mut out = Vec::with_capacity(p.sf);
            for j in 0..p.sf {
                out.push(y_chips[j] * (c_seq[seg0 + j] as f32));
            }
            out
        };

        let sample_symbol =
            |t0_samp: f64, chip_step_samp: f64, shift_samp: f64| -> Option<Vec<Complex32>> {
                let mut pos = Vec::with_capacity(p.sf);
                let base = t0_samp + shift_samp;
                for &ci in &chip_idx {
                    pos.push(base + ci * chip_step_samp);
                }
                sample_linear(&pos)
            };

        let tsym = (p.sf as f64) / (p.rc_chip_sps as f64);
        let dll_g = design_2nd_order_loop(0.6, 0.707, tsym);
        let sym_step_nom = (p.sf as f64) * (p.osf as f64);
        let sym_step_ppm = 2000.0;
        let sym_step_min = sym_step_nom * (1.0 - sym_step_ppm * 1e-6);
        let sym_step_max = sym_step_nom * (1.0 + sym_step_ppm * 1e-6);
        let mut dll = EarlyLateDll {
            sym_step_samp: sym_step_nom,
            kp: dll_g.kp,
            ki: dll_g.ki,
            sym_step_min,
            sym_step_max,
            dd_scale: 0.25,
        };
        let el_spacing_chips = 0.5;
        let dll_enabled = true;

        let chip_step0 = dll.sym_step_samp / (p.sf as f64);
        let mut u0_fingers: Vec<Vec<Complex32>> = Vec::with_capacity(n_finger);
        let mut u1_fingers: Vec<Vec<Complex32>> = Vec::with_capacity(n_finger);
        for i in 0..n_finger {
            let y0 = sample_symbol(t_sym0[i], chip_step0, 0.0)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            u0_fingers.push(demask_symbol(&y0, 0));

            // Spec §4.B / §4.C.
            let y1 = sample_symbol(t_sym0[i] + sym_step_nom, chip_step0, 0.0)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            u1_fingers.push(demask_symbol(&y1, 1));
        }
        let pre_corr: Vec<Complex32> = (0..n_finger)
            .map(|i| {
                let z0: Complex32 = u0_fingers[i].iter().copied().sum();
                let z1: Complex32 = u1_fingers[i].iter().copied().sum();
                z0 - z1
            })
            .collect();

        let i_ref = pre_corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut theta = pre_corr[i_ref].arg() as f64;

        let rot0 = Complex32::from_polar(1.0, -(theta as f32));
        let pre_den = (p.n_pre as f32) * (p.sf as f32);
        let mut g: Vec<Complex32> = pre_corr.iter().map(|&z| (z / pre_den) * rot0).collect();
        if !(g.iter().all(|v| v.re.is_finite() && v.im.is_finite())
            && g.iter().map(|v| v.norm_sqr()).sum::<f32>() > 0.0)
        {
            g = vec![Complex32::new(1.0, 0.0); n_finger];
        }

        let pll_bw_hz = 1.0;
        let zeta = 0.707;
        let pll_g = design_2nd_order_loop(pll_bw_hz, zeta, tsym);
        let kp = pll_g.kp;
        let ki = pll_g.ki;
        let mut omega = 0.0f64;
        let omega_lim = 2.0 * std::f64::consts::PI * 200.0 * tsym;

        let alpha_ch = 1.0 / (p.n_pilot as f64);

        let mut r_data_all: Vec<Vec<Complex32>> = Vec::with_capacity(p.n_data);

        let mrc_weights = |g_est: &[Complex32]| -> Vec<Complex32> {
            let den: f32 = g_est.iter().map(|v| v.norm_sqr()).sum::<f32>() + 1e-18;
            g_est.iter().map(|&v| v.conj() / den).collect()
        };

        let mut w_mrc = mrc_weights(&g);

        let phase_err = |z: Complex32| -> f64 { (z.im as f64).atan2((z.re as f64) + 1e-18) };
        let omega_err_from_halves = |u_prompt: &[Complex32]| -> Option<f64> {
            let n = u_prompt.len();
            if n < 4 {
                return None;
            }
            let mid = n / 2;
            let z0: Complex32 = u_prompt[..mid].iter().copied().sum();
            let z1: Complex32 = u_prompt[mid..].iter().copied().sum();
            if !(z0.norm() > 1e-6 && z1.norm() > 1e-6) {
                return None;
            }
            let dphi = wrap_pm_pi((z1.arg() - z0.arg()) as f64);
            Some(2.0 * dphi)
        };

        let best_and_conf = |r_all: &[Complex32]| -> (usize, f32) {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (i, &v) in r_all[..p.mw].iter().enumerate() {
                let d = v.re;
                if d > best_v {
                    second = best_v;
                    best_v = d;
                    best_i = i;
                } else if d > second {
                    second = d;
                }
            }
            let conf = (best_v - second) / (best_v.abs() + 1e-18);
            (best_i, conf)
        };

        let mut pre_u = vec![Complex32::new(0.0, 0.0); p.sf];
        let rot_theta0 = Complex32::from_polar(1.0, -(theta as f32));
        for i in 0..n_finger {
            for j in 0..p.sf {
                pre_u[j] += w_mrc[i] * (u0_fingers[i][j] * rot_theta0);
            }
        }
        let pre_mag_ref = pre_u.iter().copied().sum::<Complex32>().norm() + 1e-18;

        let mut q_data = 0usize;
        for ell in 0..p.n_sym {
            let chip_step = dll.sym_step_samp / (p.sf as f64);

            let mut u_p_fingers: Vec<Vec<Complex32>> = Vec::with_capacity(n_finger);
            for i in 0..n_finger {
                let y_p = sample_symbol(t_sym0[i], chip_step, 0.0)
                    .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
                u_p_fingers.push(demask_symbol(&y_p, ell));
            }

            // Spec §3.D
            if ell == 1 {
                for u in &mut u_p_fingers {
                    for v in u {
                        *v = -*v;
                    }
                }
            }

            let mut theta_sym = theta;
            let mut rot_chips = build_carrier_rot_chips(theta_sym, omega, p.sf);
            let mut u_p = vec![Complex32::new(0.0, 0.0); p.sf];
            for j in 0..p.sf {
                let rotj = rot_chips[j];
                for i in 0..n_finger {
                    u_p[j] += w_mrc[i] * (u_p_fingers[i][j] * rotj);
                }
            }

            let mut dll_do_update = false;
            let mut dll_dd = false;
            let mut dll_m = 0usize;

            if ell < p.n_pre || is_pilot(ell) {
                let mut z_p: Complex32 = u_p.iter().copied().sum();

                // Cycle-slip guard.
                if (z_p.re as f64) < 0.0 && z_p.norm() > 0.25 * pre_mag_ref {
                    theta = wrap_pm_pi(theta + std::f64::consts::PI);
                    for gi in &mut g {
                        *gi = -*gi;
                    }
                    w_mrc = mrc_weights(&g);

                    theta_sym = theta;
                    rot_chips = build_carrier_rot_chips(theta_sym, omega, p.sf);
                    u_p.fill(Complex32::new(0.0, 0.0));
                    for j in 0..p.sf {
                        let rotj = rot_chips[j];
                        for i in 0..n_finger {
                            u_p[j] += w_mrc[i] * (u_p_fingers[i][j] * rotj);
                        }
                    }
                }

                // Pilot-aided FLL (W0): refine ω using within-symbol phase slope.
                for _ in 0..2 {
                    let Some(omega_err) = omega_err_from_halves(&u_p) else {
                        break;
                    };
                    let fll_gain = 0.8;
                    omega = (omega + fll_gain * omega_err).clamp(-omega_lim, omega_lim);
                    rot_chips = build_carrier_rot_chips(theta_sym, omega, p.sf);
                    u_p.fill(Complex32::new(0.0, 0.0));
                    for j in 0..p.sf {
                        let rotj = rot_chips[j];
                        for i in 0..n_finger {
                            u_p[j] += w_mrc[i] * (u_p_fingers[i][j] * rotj);
                        }
                    }
                }
                z_p = u_p.iter().copied().sum();

                let err = phase_err(z_p);
                if err.is_finite() && z_p.norm() > 1e-6 {
                    pll_step(&mut theta, &mut omega, err, false, kp, ki, omega_lim);
                }

                if is_pilot(ell) {
                    for i in 0..n_finger {
                        let mut z_i = Complex32::new(0.0, 0.0);
                        for j in 0..p.sf {
                            z_i += u_p_fingers[i][j] * rot_chips[j];
                        }
                        let gi_meas = z_i / (p.sf as f32);
                        g[i] = g[i] * (1.0 - alpha_ch as f32) + gi_meas * (alpha_ch as f32);
                    }
                    w_mrc = mrc_weights(&g);
                }

                dll_do_update = true;
                dll_dd = false;
                dll_m = 0;
            } else {
                let mut r_all = u_p.clone();
                fht1024_in_place(&mut r_all);
                let (mut best_i, mut conf) = best_and_conf(&r_all);

                // Decision-directed FLL: refine ω using the decided prompt stream.
                if conf.is_finite() && conf > 0.10 {
                    let wrow = walsh_row(best_i as u16, p.sf);
                    let mid = p.sf / 2;
                    let mut z0 = Complex32::new(0.0, 0.0);
                    let mut z1 = Complex32::new(0.0, 0.0);
                    for j in 0..mid {
                        z0 += u_p[j] * (wrow[j] as f32);
                    }
                    for j in mid..p.sf {
                        z1 += u_p[j] * (wrow[j] as f32);
                    }
                    if z0.norm() > 1e-6 && z1.norm() > 1e-6 {
                        let dphi = wrap_pm_pi((z1.arg() - z0.arg()) as f64);
                        let omega_err = 2.0 * dphi;
                        let fll_gain = 0.5;
                        omega = (omega + fll_gain * omega_err).clamp(-omega_lim, omega_lim);

                        rot_chips = build_carrier_rot_chips(theta_sym, omega, p.sf);
                        u_p.fill(Complex32::new(0.0, 0.0));
                        for j in 0..p.sf {
                            let rotj = rot_chips[j];
                            for i in 0..n_finger {
                                u_p[j] += w_mrc[i] * (u_p_fingers[i][j] * rotj);
                            }
                        }

                        r_all.clone_from(&u_p);
                        fht1024_in_place(&mut r_all);
                        (best_i, conf) = best_and_conf(&r_all);
                    }
                }

                let r256 = r_all[..p.mw].to_vec();
                r_data_all.push(r256);

                if conf.is_finite() && conf > 0.10 {
                    let mut z_dd = r_all[best_i];
                    if z_dd.re < 0.0 {
                        z_dd = -z_dd;
                    }
                    let err = phase_err(z_dd);
                    if err.is_finite() && z_dd.norm() > 1e-6 {
                        pll_step(&mut theta, &mut omega, err, true, kp, ki, omega_lim);
                    }

                    dll_do_update = true;
                    dll_dd = true;
                    dll_m = best_i;
                }
                q_data += 1;
            }

            let mut phase_adj = 0.0f64;
            if dll_enabled && dll_do_update {
                let el_shift = el_spacing_chips * chip_step;
                let mut u_e = vec![Complex32::new(0.0, 0.0); p.sf];
                let mut u_l = vec![Complex32::new(0.0, 0.0); p.sf];
                for i in 0..n_finger {
                    let y_e = sample_symbol(t_sym0[i], chip_step, -el_shift)
                        .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
                    let y_l = sample_symbol(t_sym0[i], chip_step, el_shift)
                        .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
                    let u_e_i = demask_symbol(&y_e, ell);
                    let u_l_i = demask_symbol(&y_l, ell);
                    for j in 0..p.sf {
                        let rotj = rot_chips[j];
                        u_e[j] += w_mrc[i] * (u_e_i[j] * rotj);
                        u_l[j] += w_mrc[i] * (u_l_i[j] * rotj);
                    }
                }

                let (z_e, z_l) = if dll_m == 0 {
                    (
                        u_e.iter().copied().sum::<Complex32>(),
                        u_l.iter().copied().sum::<Complex32>(),
                    )
                } else {
                    let wrow = walsh_row(dll_m as u16, p.sf);
                    let mut se = Complex32::new(0.0, 0.0);
                    let mut sl = Complex32::new(0.0, 0.0);
                    for j in 0..p.sf {
                        let w = wrow[j] as f32;
                        se += u_e[j] * w;
                        sl += u_l[j] * w;
                    }
                    (se, sl)
                };

                let ae = z_e.norm();
                let al = z_l.norm();
                let den = ae + al + 1e-18;
                let e = (ae - al) / den;
                let err_samp = (e as f64) * (el_shift / 2.0);
                if err_samp.is_finite() && den > 1e-6 {
                    phase_adj = dll.update(err_samp, dll_dd);
                }
            }

            for t in &mut t_sym0 {
                *t += dll.sym_step_samp + phase_adj;
            }
        }

        if q_data != p.n_data || r_data_all.len() != p.n_data {
            return Ok((None, DecodeMeta::error("data_symbol_count_mismatch")));
        }

        // Spec §4.D.2–§4.D.3.
        let mut llr = [0f64; 512];
        for q in 0..p.n_data {
            let d = &r_data_all[q];
            for t in 0..p.k_bits_per_sym {
                let mut m0 = f32::NEG_INFINITY;
                let mut m1 = f32::NEG_INFINITY;
                let shift = (p.k_bits_per_sym - 1 - t) as u16;
                for m in 0..p.mw {
                    let bit = ((m as u16) >> shift) & 1;
                    let v = d[m].re;
                    if bit == 0 {
                        if v > m0 {
                            m0 = v;
                        }
                    } else if v > m1 {
                        m1 = v;
                    }
                }
                llr[q * p.k_bits_per_sym + t] = (m0 - m1) as f64;
            }
        }
        for v in &mut llr {
            *v = v.clamp(-1e6, 1e6);
        }

        let u_hat = ldpc_decode_to_u256_from_llr(&llr, maxiter);
        let (hdr, payload, crc_ok) = parse_u_bits(&u_hat)?;
        let meta = DecodeMeta {
            crc_ok,
            ver: hdr.ver,
            typ: hdr.typ,
            len: hdr.length,
            err: None,
        };
        if !crc_ok {
            return Ok((None, meta));
        }
        Ok((Some(payload[..(hdr.length as usize)].to_vec()), meta))
    }
}
