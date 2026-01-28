use super::util::{derotate_cfo_in_place, is_pilot, wrap_pm_pi};
use super::{DecodeMeta, ScBltcModem};
use crate::crypto::gen_code_aes_ctr;
use crate::frame::parse_u_bits;
use crate::interleaver::deinterleave_frame_llr;
use crate::polar::polar_decode_to_u256_from_llr;
use crate::tracking::{design_2nd_order_loop, EarlyLateDll};
use crate::walsh::{fht1024_in_place, walsh_row};
use num_complex::Complex32;

type FingerVec = Vec<Complex32>;
type FingerSamples = Vec<FingerVec>;

struct SymbolBuffers {
    u_p_fingers: Vec<Vec<Complex32>>,
    rot_tmp: Vec<Complex32>,
    rot_best: Vec<Complex32>,
    u_tmp: Vec<Complex32>,
    u_best: Vec<Complex32>,
    r_tmp: Vec<Complex32>,
    r_best: Vec<Complex32>,
    u_e: Vec<Complex32>,
    u_l: Vec<Complex32>,
    tmp_e: Vec<Complex32>,
    tmp_l: Vec<Complex32>,
}

impl SymbolBuffers {
    fn new(n_finger: usize, sf: usize) -> Self {
        Self {
            u_p_fingers: vec![vec![Complex32::new(0.0, 0.0); sf]; n_finger],
            rot_tmp: vec![Complex32::new(0.0, 0.0); sf],
            rot_best: vec![Complex32::new(0.0, 0.0); sf],
            u_tmp: vec![Complex32::new(0.0, 0.0); sf],
            u_best: vec![Complex32::new(0.0, 0.0); sf],
            r_tmp: vec![Complex32::new(0.0, 0.0); sf],
            r_best: vec![Complex32::new(0.0, 0.0); sf],
            u_e: vec![Complex32::new(0.0, 0.0); sf],
            u_l: vec![Complex32::new(0.0, 0.0); sf],
            tmp_e: vec![Complex32::new(0.0, 0.0); sf],
            tmp_l: vec![Complex32::new(0.0, 0.0); sf],
        }
    }
}

struct DllUpdate {
    dd: bool,
    m: usize,
}

struct SymbolProcOutcome {
    freq_conf: f32,
    best_dhz: f64,
    dll_update: Option<DllUpdate>,
}

struct SymbolTracker<'a> {
    p: &'a crate::params::Params,
    y: &'a [Complex32],
    c_seq: &'a [i8],
    n_finger: usize,
    t_sym0: Vec<f64>,
    dll: EarlyLateDll,
    theta: f64,
    omega: f64,
    omega_lim: f64,
    pll_kp: f64,
    pll_ki: f64,
    tsym: f64,
    g: Vec<Complex32>,
    w_mrc: Vec<Complex32>,
    pre_mag_ref: f32,
    bank_dhz: Vec<f64>,
    bank_domega: Vec<f64>,
    bank_step_hz: f64,
    snap_cand_hz: Option<f64>,
    snap_count: usize,
    freq_snap_confirm: usize,
    freq_snap_min_abs_hz: f64,
    alpha_ch: f64,
    el_spacing_chips: f64,
    buffers: SymbolBuffers,
}

impl<'a> SymbolTracker<'a> {
    fn new(
        p: &'a crate::params::Params,
        y: &'a [Complex32],
        c_seq: &'a [i8],
        n_finger: usize,
        n_offset_total: &[usize],
        frame_start: f64,
        cascade_delay: usize,
    ) -> anyhow::Result<Self> {
        let t_sym0 = Self::init_symbol_times(n_offset_total, n_finger, frame_start, cascade_delay);
        let tsym = (p.sf as f64) / (p.rc_chip_sps as f64);
        let dll = Self::init_dll(p, tsym);
        let chip_step0 = dll.sym_step_samp / (p.sf as f64);

        let (u0_fingers, u1_fingers) = Self::sample_initial_fingers(
            y,
            c_seq,
            p,
            n_finger,
            &t_sym0,
            chip_step0,
            dll.sym_step_samp,
        )?;
        let pre_corr = Self::pre_corr(&u0_fingers, &u1_fingers);
        let (theta, g) = Self::init_phase_and_channel(&pre_corr, p.n_pre, p.sf, n_finger);
        let w_mrc = Self::mrc_weights(&g);

        let (pll_kp, pll_ki, omega_lim) = Self::init_pll(tsym);
        let omega = 0.0f64;

        let (bank_dhz, bank_domega, bank_step_hz) = Self::init_freq_bank(tsym);
        let pre_mag_ref = Self::preamble_mag_ref(p, n_finger, &u0_fingers, &w_mrc, theta);
        let alpha_ch = 1.0 / (p.n_pilot as f64);

        Ok(Self {
            p,
            y,
            c_seq,
            n_finger,
            t_sym0,
            dll,
            theta,
            omega,
            omega_lim,
            pll_kp,
            pll_ki,
            tsym,
            g,
            w_mrc,
            pre_mag_ref,
            bank_dhz,
            bank_domega,
            bank_step_hz,
            snap_cand_hz: None,
            snap_count: 0,
            freq_snap_confirm: 3,
            freq_snap_min_abs_hz: 0.75,
            alpha_ch,
            el_spacing_chips: 0.5,
            buffers: SymbolBuffers::new(n_finger, p.sf),
        })
    }

    fn mrc_weights(g_est: &[Complex32]) -> Vec<Complex32> {
        let den: f32 = g_est.iter().map(|v| v.norm_sqr()).sum::<f32>() + 1e-18;
        g_est.iter().map(|&v| v.conj() / den).collect()
    }

    fn init_symbol_times(
        n_offset_total: &[usize],
        n_finger: usize,
        frame_start: f64,
        cascade_delay: usize,
    ) -> Vec<f64> {
        let mut t_sym0 = vec![0f64; n_finger];
        for (i, &off) in n_offset_total.iter().take(n_finger).enumerate() {
            t_sym0[i] = frame_start + (off as f64) + (cascade_delay as f64);
        }
        t_sym0
    }

    fn init_dll(p: &crate::params::Params, tsym: f64) -> EarlyLateDll {
        let dll_g = design_2nd_order_loop(0.6, 0.707, tsym);
        let sym_step_nom = (p.sf as f64) * (p.osf as f64);
        let sym_step_ppm = 2000.0;
        let sym_step_min = sym_step_nom * (1.0 - sym_step_ppm * 1e-6);
        let sym_step_max = sym_step_nom * (1.0 + sym_step_ppm * 1e-6);
        EarlyLateDll {
            sym_step_samp: sym_step_nom,
            kp: dll_g.kp,
            ki: dll_g.ki,
            sym_step_min,
            sym_step_max,
            dd_scale: 0.25,
        }
    }

    fn sample_initial_fingers(
        y: &[Complex32],
        c_seq: &[i8],
        p: &crate::params::Params,
        n_finger: usize,
        t_sym0: &[f64],
        chip_step0: f64,
        sym_step_nom: f64,
    ) -> anyhow::Result<(FingerSamples, FingerSamples)> {
        let mut u0_fingers: FingerSamples = Vec::with_capacity(n_finger);
        let mut u1_fingers: FingerSamples = Vec::with_capacity(n_finger);
        for &t0 in t_sym0.iter().take(n_finger) {
            let mut y0 = vec![Complex32::new(0.0, 0.0); p.sf];
            Self::sample_symbol_into(y, t0, chip_step0, 0.0, &mut y0)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(c_seq, p.sf, 0, &mut y0);
            u0_fingers.push(y0);

            let mut y1 = vec![Complex32::new(0.0, 0.0); p.sf];
            Self::sample_symbol_into(y, t0 + sym_step_nom, chip_step0, 0.0, &mut y1)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(c_seq, p.sf, 1, &mut y1);
            u1_fingers.push(y1);
        }
        Ok((u0_fingers, u1_fingers))
    }

    fn pre_corr(u0_fingers: &[FingerVec], u1_fingers: &[FingerVec]) -> Vec<Complex32> {
        (0..u0_fingers.len())
            .map(|i| {
                let z0: Complex32 = u0_fingers[i].iter().copied().sum();
                let z1: Complex32 = u1_fingers[i].iter().copied().sum();
                z0 - z1
            })
            .collect()
    }

    fn init_phase_and_channel(
        pre_corr: &[Complex32],
        n_pre: usize,
        sf: usize,
        n_finger: usize,
    ) -> (f64, Vec<Complex32>) {
        let i_ref = pre_corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let theta = pre_corr[i_ref].arg() as f64;

        let rot0 = Complex32::from_polar(1.0, -(theta as f32));
        let pre_den = (n_pre as f32) * (sf as f32);
        let mut g: Vec<Complex32> = pre_corr.iter().map(|&z| (z / pre_den) * rot0).collect();
        if !(g.iter().all(|v| v.re.is_finite() && v.im.is_finite())
            && g.iter().map(|v| v.norm_sqr()).sum::<f32>() > 0.0)
        {
            g = vec![Complex32::new(1.0, 0.0); n_finger];
        }
        (theta, g)
    }

    fn init_pll(tsym: f64) -> (f64, f64, f64) {
        let pll_bw_hz = 1.0;
        let zeta = 0.707;
        let pll_g = design_2nd_order_loop(pll_bw_hz, zeta, tsym);
        let omega_lim = 2.0 * std::f64::consts::PI * 200.0 * tsym;
        (pll_g.kp, pll_g.ki, omega_lim)
    }

    fn init_freq_bank(tsym: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let bank_half_hz = 4.0f64;
        let bank_step_hz = 0.25f64;
        let bank_k = (bank_half_hz / bank_step_hz).round() as i32;
        let bank_dhz: Vec<f64> = (-bank_k..=bank_k)
            .map(|k| (k as f64) * bank_step_hz)
            .collect();
        let bank_domega: Vec<f64> = bank_dhz
            .iter()
            .map(|&df_hz| 2.0 * std::f64::consts::PI * df_hz * tsym)
            .collect();
        (bank_dhz, bank_domega, bank_step_hz)
    }

    fn preamble_mag_ref(
        p: &crate::params::Params,
        n_finger: usize,
        u0_fingers: &[Vec<Complex32>],
        w_mrc: &[Complex32],
        theta: f64,
    ) -> f32 {
        let mut pre_u = vec![Complex32::new(0.0, 0.0); p.sf];
        let rot_theta0 = Complex32::from_polar(1.0, -(theta as f32));
        for i in 0..n_finger {
            for j in 0..p.sf {
                pre_u[j] += w_mrc[i] * (u0_fingers[i][j] * rot_theta0);
            }
        }
        pre_u.iter().copied().sum::<Complex32>().norm() + 1e-18
    }

    fn sample_symbol_into(
        y: &[Complex32],
        t0_samp: f64,
        chip_step_samp: f64,
        shift_samp: f64,
        out: &mut [Complex32],
    ) -> Option<()> {
        let base = t0_samp + shift_samp;
        for (j, v) in out.iter_mut().enumerate() {
            let pp = base + (j as f64) * chip_step_samp;
            if pp < 0.0 {
                return None;
            }
            let i0 = pp.floor() as isize;
            let a = (pp - (i0 as f64)) as f32;
            let i0u = i0 as usize;
            if (i0u + 1) >= y.len() {
                return None;
            }
            *v = y[i0u] * (1.0 - a) + y[i0u + 1] * a;
        }
        Some(())
    }

    fn demask_in_place(c_seq: &[i8], sf: usize, ell: usize, chips: &mut [Complex32]) {
        let seg0 = ell * sf;
        for j in 0..sf {
            chips[j] *= c_seq[seg0 + j] as f32;
        }
    }

    fn fill_rot_chips(rot: &mut [Complex32], sf: usize, theta: f64, omega: f64) {
        let dphi = -(omega as f32) / (sf as f32);
        let mut ph = -(theta as f32);
        for v in rot.iter_mut() {
            *v = Complex32::from_polar(1.0, ph);
            ph += dphi;
        }
    }

    fn best_and_conf_mag(r_all: &[Complex32], mw: usize) -> (usize, f32) {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for (i, &v) in r_all[..mw].iter().enumerate() {
            let d = v.norm_sqr();
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
    }

    fn pll_predict(theta: &mut f64, omega_used: f64) {
        *theta = wrap_pm_pi(*theta + omega_used);
    }

    fn pll_correct(
        theta: &mut f64,
        omega: &mut f64,
        err: f64,
        dd: bool,
        kp: f64,
        ki: f64,
        omega_lim: f64,
    ) {
        let scale = if dd { 0.5 } else { 1.0 };
        *omega = (*omega + (ki * scale) * err).clamp(-omega_lim, omega_lim);
        *theta = wrap_pm_pi(*theta + (kp * scale) * err);
    }

    fn process_symbol(
        &mut self,
        ell: usize,
        r_data_all: &mut Vec<Vec<Complex32>>,
        q_data: &mut usize,
    ) -> anyhow::Result<()> {
        let p = self.p;
        let chip_step = self.dll.sym_step_samp / (p.sf as f64);
        for i in 0..self.n_finger {
            let buf = &mut self.buffers.u_p_fingers[i];
            Self::sample_symbol_into(self.y, self.t_sym0[i], chip_step, 0.0, buf)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(self.c_seq, p.sf, ell, buf);
        }

        // Spec §3.D
        if ell == 1 {
            for u in &mut self.buffers.u_p_fingers {
                for v in u {
                    *v = -*v;
                }
            }
        }

        let outcome = if ell < p.n_pre || is_pilot(ell) {
            self.process_pilot_symbol(ell)
        } else {
            self.process_data_symbol(r_data_all, q_data)?
        };

        self.apply_freq_snap(outcome.freq_conf, outcome.best_dhz);

        let phase_adj = if let Some(update) = outcome.dll_update {
            self.update_dll(ell, chip_step, update)?
        } else {
            0.0
        };

        for t in &mut self.t_sym0 {
            *t += self.dll.sym_step_samp + phase_adj;
        }
        Ok(())
    }

    fn process_pilot_symbol(&mut self, ell: usize) -> SymbolProcOutcome {
        let p = self.p;
        let kp = self.pll_kp;
        let ki = self.pll_ki;
        let omega_lim = self.omega_lim;
        let sf = p.sf;

        let mut omega_used = self.omega;
        let mut best_dhz = 0.0f64;

        let mut best_v = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for (h, &domega) in self.bank_domega.iter().enumerate() {
            let omega_h = (self.omega + domega).clamp(-self.omega_lim, self.omega_lim);
            Self::fill_rot_chips(&mut self.buffers.rot_tmp, sf, self.theta, omega_h);

            self.buffers.u_tmp.fill(Complex32::new(0.0, 0.0));
            for j in 0..p.sf {
                let rotj = self.buffers.rot_tmp[j];
                for i in 0..self.n_finger {
                    self.buffers.u_tmp[j] +=
                        self.w_mrc[i] * (self.buffers.u_p_fingers[i][j] * rotj);
                }
            }
            let z: Complex32 = self.buffers.u_tmp.iter().copied().sum();
            let m = z.norm_sqr();
            if m > best_v {
                second = best_v;
                best_v = m;
                omega_used = omega_h;
                best_dhz = self.bank_dhz[h];
                self.buffers.rot_best.clone_from(&self.buffers.rot_tmp);
                self.buffers.u_best.clone_from(&self.buffers.u_tmp);
            } else if m > second {
                second = m;
            }
        }
        let freq_conf = (best_v - second) / (best_v.abs() + 1e-18);

        // Cycle-slip guard.
        let mut z_p: Complex32 = self.buffers.u_best.iter().copied().sum();
        if (z_p.re as f64) < 0.0 && z_p.norm() > 0.25 * self.pre_mag_ref {
            self.theta = wrap_pm_pi(self.theta + std::f64::consts::PI);
            for gi in &mut self.g {
                *gi = -*gi;
            }
            self.w_mrc = Self::mrc_weights(&self.g);

            Self::fill_rot_chips(&mut self.buffers.rot_best, sf, self.theta, omega_used);
            self.buffers.u_best.fill(Complex32::new(0.0, 0.0));
            for j in 0..p.sf {
                let rotj = self.buffers.rot_best[j];
                for i in 0..self.n_finger {
                    self.buffers.u_best[j] +=
                        self.w_mrc[i] * (self.buffers.u_p_fingers[i][j] * rotj);
                }
            }
            z_p = self.buffers.u_best.iter().copied().sum();
        }

        let err = (z_p.im as f64).atan2((z_p.re as f64) + 1e-18);
        Self::pll_predict(&mut self.theta, omega_used);
        if err.is_finite() && z_p.norm() > 1e-6 {
            Self::pll_correct(
                &mut self.theta,
                &mut self.omega,
                err,
                false,
                kp,
                ki,
                omega_lim,
            );
        }

        if is_pilot(ell) {
            for i in 0..self.n_finger {
                let mut z_i = Complex32::new(0.0, 0.0);
                for j in 0..p.sf {
                    z_i += self.buffers.u_p_fingers[i][j] * self.buffers.rot_best[j];
                }
                let gi_meas = z_i / (p.sf as f32);
                self.g[i] =
                    self.g[i] * (1.0 - self.alpha_ch as f32) + gi_meas * (self.alpha_ch as f32);
            }
            self.w_mrc = Self::mrc_weights(&self.g);
        }

        SymbolProcOutcome {
            freq_conf,
            best_dhz,
            dll_update: Some(DllUpdate { dd: false, m: 0 }),
        }
    }

    fn process_data_symbol(
        &mut self,
        r_data_all: &mut Vec<Vec<Complex32>>,
        q_data: &mut usize,
    ) -> anyhow::Result<SymbolProcOutcome> {
        let p = self.p;
        let kp = self.pll_kp;
        let ki = self.pll_ki;
        let omega_lim = self.omega_lim;
        let sf = p.sf;
        let mw = p.mw;

        let mut omega_used = self.omega;
        let mut best_dhz = 0.0f64;

        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        let mut best_code_conf = 0.0f32;
        for (h, &domega) in self.bank_domega.iter().enumerate() {
            let omega_h = (self.omega + domega).clamp(-self.omega_lim, self.omega_lim);
            Self::fill_rot_chips(&mut self.buffers.rot_tmp, sf, self.theta, omega_h);

            self.buffers.u_tmp.fill(Complex32::new(0.0, 0.0));
            for j in 0..p.sf {
                let rotj = self.buffers.rot_tmp[j];
                for i in 0..self.n_finger {
                    self.buffers.u_tmp[j] +=
                        self.w_mrc[i] * (self.buffers.u_p_fingers[i][j] * rotj);
                }
            }
            self.buffers.r_tmp.clone_from(&self.buffers.u_tmp);
            fht1024_in_place(&mut self.buffers.r_tmp);
            let (i_h, conf_h) = Self::best_and_conf_mag(&self.buffers.r_tmp, mw);
            let m = self.buffers.r_tmp[i_h].norm_sqr();
            if m > best_v {
                second = best_v;
                best_v = m;
                omega_used = omega_h;
                best_dhz = self.bank_dhz[h];
                best_i = i_h;
                best_code_conf = conf_h;
                self.buffers.rot_best.clone_from(&self.buffers.rot_tmp);
                self.buffers.r_best.clone_from(&self.buffers.r_tmp);
            } else if m > second {
                second = m;
            }
        }
        let freq_conf = (best_v - second) / (best_v.abs() + 1e-18);
        // Gate PLL/DD only on code confidence; freq_conf can be flat even when code is OK.
        let data_conf = best_code_conf;

        let r256 = self.buffers.r_best[..p.mw].to_vec();
        r_data_all.push(r256);

        Self::pll_predict(&mut self.theta, omega_used);
        let dll_update = if data_conf.is_finite() && data_conf > 0.10 {
            let mut z_dd = self.buffers.r_best[best_i];
            if z_dd.re < 0.0 {
                z_dd = -z_dd;
            }
            let err = (z_dd.im as f64).atan2((z_dd.re as f64) + 1e-18);
            if err.is_finite() && z_dd.norm() > 1e-6 {
                Self::pll_correct(
                    &mut self.theta,
                    &mut self.omega,
                    err,
                    true,
                    kp,
                    ki,
                    omega_lim,
                );
            }

            // Decision-directed per-symbol phase alignment for soft-demapping (LLRs).
            let den = z_dd.norm();
            if den > 1e-6 {
                let rot = z_dd.conj() / den;
                for v in r_data_all.last_mut().unwrap().iter_mut() {
                    *v *= rot;
                }
            }

            Some(DllUpdate {
                dd: true,
                m: best_i,
            })
        } else {
            None
        };
        *q_data += 1;

        Ok(SymbolProcOutcome {
            freq_conf,
            best_dhz,
            dll_update,
        })
    }

    fn apply_freq_snap(&mut self, freq_conf: f32, best_dhz: f64) {
        let snap_conf_min = 0.15f32;
        if freq_conf.is_finite() && freq_conf >= snap_conf_min {
            let df = best_dhz;
            if let Some(cand) = self.snap_cand_hz {
                if (df - cand).abs() <= 0.5 * self.bank_step_hz {
                    self.snap_count += 1;
                } else {
                    self.snap_cand_hz = Some(df);
                    self.snap_count = 1;
                }
            } else {
                self.snap_cand_hz = Some(df);
                self.snap_count = 1;
            }
            if self.snap_count >= self.freq_snap_confirm {
                let df2 = self.snap_cand_hz.unwrap_or(0.0);
                if df2.abs() >= self.freq_snap_min_abs_hz {
                    self.omega = (self.omega + 2.0 * std::f64::consts::PI * df2 * self.tsym)
                        .clamp(-self.omega_lim, self.omega_lim);
                }
                self.snap_cand_hz = None;
                self.snap_count = 0;
            }
        }
    }

    fn update_dll(&mut self, ell: usize, chip_step: f64, update: DllUpdate) -> anyhow::Result<f64> {
        let p = self.p;
        let el_shift = self.el_spacing_chips * chip_step;
        self.buffers.u_e.fill(Complex32::new(0.0, 0.0));
        self.buffers.u_l.fill(Complex32::new(0.0, 0.0));
        for i in 0..self.n_finger {
            Self::sample_symbol_into(
                self.y,
                self.t_sym0[i],
                chip_step,
                -el_shift,
                &mut self.buffers.tmp_e,
            )
            .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::sample_symbol_into(
                self.y,
                self.t_sym0[i],
                chip_step,
                el_shift,
                &mut self.buffers.tmp_l,
            )
            .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(self.c_seq, p.sf, ell, &mut self.buffers.tmp_e);
            Self::demask_in_place(self.c_seq, p.sf, ell, &mut self.buffers.tmp_l);
            for j in 0..p.sf {
                let rotj = self.buffers.rot_best[j];
                self.buffers.u_e[j] += self.w_mrc[i] * (self.buffers.tmp_e[j] * rotj);
                self.buffers.u_l[j] += self.w_mrc[i] * (self.buffers.tmp_l[j] * rotj);
            }
        }

        let (z_e, z_l) = if update.m == 0 {
            (
                self.buffers.u_e.iter().copied().sum::<Complex32>(),
                self.buffers.u_l.iter().copied().sum::<Complex32>(),
            )
        } else {
            let wrow = walsh_row(update.m as u16, p.sf);
            let mut se = Complex32::new(0.0, 0.0);
            let mut sl = Complex32::new(0.0, 0.0);
            for (j, &w) in wrow.iter().enumerate().take(p.sf) {
                let wf = w as f32;
                se += self.buffers.u_e[j] * wf;
                sl += self.buffers.u_l[j] * wf;
            }
            (se, sl)
        };

        let ae = z_e.norm();
        let al = z_l.norm();
        let den = ae + al + 1e-18;
        let e = (ae - al) / den;
        let err_samp = (e as f64) * (el_shift / 2.0);
        let mut phase_adj = 0.0f64;
        if err_samp.is_finite() && den > 1e-6 {
            phase_adj = self.dll.update(err_samp, update.dd);
        }
        Ok(phase_adj)
    }
}

impl ScBltcModem {
    /// Spec §4.C–§4.D.
    pub fn demod_decode_raw(
        &self,
        rx_samples: &[Complex32],
        ti_tx: u64,
        frame_start_sample: usize,
        n_offset_total: &[usize],
        cfo_hz: f64,
        scl_list_size: usize,
    ) -> anyhow::Result<(Option<Vec<u8>>, DecodeMeta)> {
        let p = &self.p;
        let x_buf = if cfo_hz != 0.0 {
            let mut tmp = rx_samples.to_vec();
            derotate_cfo_in_place(&mut tmp, p.fs_hz, cfo_hz);
            Some(tmp)
        } else {
            None
        };
        let x = x_buf.as_deref().unwrap_or(rx_samples);
        let y = self.rrc.filter_same(x);
        self.demod_decode_matched(
            &y,
            ti_tx,
            frame_start_sample,
            n_offset_total,
            0.0,
            scl_list_size,
        )
    }

    /// Spec §4.C–§4.D.
    pub fn demod_decode_matched(
        &self,
        y_matched: &[Complex32],
        ti_tx: u64,
        frame_start_sample: usize,
        n_offset_total: &[usize],
        cfo_hz: f64,
        scl_list_size: usize,
    ) -> anyhow::Result<(Option<Vec<u8>>, DecodeMeta)> {
        let p = &self.p;
        if n_offset_total.is_empty() {
            return Ok((None, DecodeMeta::error("no_offsets")));
        }

        let y_buf = if cfo_hz != 0.0 {
            let mut tmp = y_matched.to_vec();
            derotate_cfo_in_place(&mut tmp, p.fs_hz, cfo_hz);
            Some(tmp)
        } else {
            None
        };
        let y = y_buf.as_deref().unwrap_or(y_matched);

        let c_seq = gen_code_aes_ctr(&self.key, ti_tx, p.frame_chips(), p.domain_u32);

        let cascade_delay = 2 * self.rrc.delay();
        let frame_start = frame_start_sample as f64;

        let n_finger = n_offset_total.len().min(3);
        let mut tracker = SymbolTracker::new(
            p,
            y,
            &c_seq,
            n_finger,
            n_offset_total,
            frame_start,
            cascade_delay,
        )?;

        let mut r_data_all: Vec<Vec<Complex32>> = Vec::with_capacity(p.n_data);
        let mut q_data = 0usize;
        for ell in 0..p.n_sym {
            tracker.process_symbol(ell, &mut r_data_all, &mut q_data)?;
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
                for (m, v) in d.iter().enumerate().take(p.mw) {
                    let bit = ((m as u16) >> shift) & 1;
                    let v = v.re;
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

        let llr = deinterleave_frame_llr(&llr);
        let u_hat = polar_decode_to_u256_from_llr(&llr, scl_list_size);
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
