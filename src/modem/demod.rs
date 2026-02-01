//! Tracking demodulator and decoder (Specification §4.C-§4.D).
//!
//! Given:
//! - an acquired `ti_tx` (spreading seed)
//! - a start sample index for the acquired IV epoch
//! - one or more RAKE finger offsets within that epoch
//!
//! This module:
//! 1) generates the full-frame chip mask `C_seq`
//! 2) samples chips for each finger with a DLL (early/late gate) and linear interpolation
//! 3) runs a symbol-rate PLL/Costas loop with a small frequency bank (anti-slip)
//! 4) combines fingers with MRC and performs Walsh matched filtering (FHT1024)
//! 5) generates soft LLRs, deinterleaves, and CA-SCL decodes the polar code
//!
//! High-level flow:
//! ```text
//! y_matched[n]
//!   -> (per symbol ell) sample chips per finger -> demask with C_seq
//!   -> MRC combine -> PLL/DLL update
//!   -> if data: FHT1024 -> LLRs
//!   -> polar decode -> parse Header/Payload/CRC
//! ```

use super::util::{
    apply_hop_with_start_in_place, derotate_cfo_in_place, is_pilot, wrap_pm_pi, HopParams,
};
use super::{DecodeMeta, ScBltcModem};
use crate::crypto::{gen_code_structured_aes_ctr, JitterSpec};
use crate::frame::parse_u_bits;
use crate::interleaver::deinterleave_frame_llr;
use crate::polar::polar_decode_to_u256_from_llr;
use crate::tracking::{design_2nd_order_loop, EarlyLateDll};
use crate::walsh::{fht1024_in_place, walsh_sign};
use num_complex::Complex32;
use std::cmp::Ordering;

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
    j_seq: &'a [usize],
    n_finger: usize,
    t_sym0: Vec<f64>,
    dll: EarlyLateDll,
    theta: f64,
    omega: f64,
    omega_lim: f64,
    pll_kp: f64,
    pll_ki: f64,
    t_data: f64,
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

struct TrackerTiming<'a> {
    n_offset_total: &'a [usize],
    frame_start: f64,
    cascade_delay: usize,
}

struct FingerInit<'a> {
    n_finger: usize,
    t_sym0: &'a [f64],
    j0_chips: usize,
    chip_step0: f64,
    sym_step_nom: f64,
}

impl<'a> SymbolTracker<'a> {
    fn new(
        p: &'a crate::params::Params,
        y: &'a [Complex32],
        c_seq: &'a [i8],
        j_seq: &'a [usize],
        n_finger: usize,
        timing: TrackerTiming<'a>,
    ) -> anyhow::Result<Self> {
        let t_sym0 = Self::init_symbol_times(
            timing.n_offset_total,
            n_finger,
            timing.frame_start,
            timing.cascade_delay,
        );
        // Data segment duration (fixed, does not include guard/noise).
        let t_data = (p.sf() as f64) / (p.rc_chip_sps() as f64);
        // Symbol update period (includes guard/noise) used for loop-gain design.
        let j_avg = (p.jitter_min_chips() as f64) + 0.5 * (p.jitter_span_chips() as f64);
        let t_update = ((p.sf() as f64) + j_avg) / (p.rc_chip_sps() as f64);
        let dll = Self::init_dll(p, t_update);
        let chip_step0 = dll.sym_step_samp / (p.sf() as f64);

        let (u0_fingers, u1_fingers) = Self::sample_initial_fingers(
            y,
            c_seq,
            p,
            FingerInit {
                n_finger,
                t_sym0: &t_sym0,
                j0_chips: j_seq.first().copied().unwrap_or(0),
                chip_step0,
                sym_step_nom: dll.sym_step_samp,
            },
        )?;
        let pre_corr = Self::pre_corr(&u0_fingers, &u1_fingers);
        let (theta, g) = Self::init_phase_and_channel(&pre_corr, p.n_pre(), p.sf(), n_finger);
        let w_mrc = Self::mrc_weights(&g);

        let (pll_kp, pll_ki, omega_lim) = Self::init_pll(t_update, t_data);
        let omega = 0.0f64;

        let (bank_dhz, bank_domega, bank_step_hz) = Self::init_freq_bank(t_data);
        let pre_mag_ref = Self::preamble_mag_ref(p, n_finger, &u0_fingers, &w_mrc, theta);
        let alpha_ch = 1.0 / (p.n_pilot() as f64);

        Ok(Self {
            p,
            y,
            c_seq,
            j_seq,
            n_finger,
            t_sym0,
            dll,
            theta,
            omega,
            omega_lim,
            pll_kp,
            pll_ki,
            t_data,
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
            buffers: SymbolBuffers::new(n_finger, p.sf()),
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
        let sym_step_nom = (p.sf() as f64) * (p.osf() as f64);
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
        init: FingerInit<'_>,
    ) -> anyhow::Result<(FingerSamples, FingerSamples)> {
        let mut u0_fingers: FingerSamples = Vec::with_capacity(init.n_finger);
        let mut u1_fingers: FingerSamples = Vec::with_capacity(init.n_finger);
        let gi0_samp = (init.j0_chips as f64) * init.chip_step0;
        for &t0 in init.t_sym0.iter().take(init.n_finger) {
            let mut y0 = vec![Complex32::new(0.0, 0.0); p.sf()];
            Self::sample_symbol_into(y, t0, init.chip_step0, 0.0, &mut y0)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(c_seq, p.sf(), 0, &mut y0);
            u0_fingers.push(y0);

            let mut y1 = vec![Complex32::new(0.0, 0.0); p.sf()];
            // Symbol 1 starts after the symbol-0 data segment plus its guard/noise span.
            Self::sample_symbol_into(
                y,
                t0 + init.sym_step_nom + gi0_samp,
                init.chip_step0,
                0.0,
                &mut y1,
            )
            .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(c_seq, p.sf(), 1, &mut y1);
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
            .max_by(|a, b| {
                let an = a.1.norm();
                let bn = b.1.norm();
                match (an.is_finite(), bn.is_finite()) {
                    (true, true) => an.total_cmp(&bn),
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => Ordering::Equal,
                }
            })
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

    fn init_pll(t_update: f64, t_data: f64) -> (f64, f64, f64) {
        let pll_bw_hz = 1.0;
        let zeta = 0.707;
        let pll_g = design_2nd_order_loop(pll_bw_hz, zeta, t_update);
        // `omega` is stored as the phase advance across the data segment.
        let omega_lim = 2.0 * std::f64::consts::PI * 200.0 * t_data;
        (pll_g.kp, pll_g.ki, omega_lim)
    }

    fn init_freq_bank(t_data: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let bank_half_hz = 4.0f64;
        let bank_step_hz = 0.25f64;
        let bank_k = (bank_half_hz / bank_step_hz).round() as i32;
        let bank_dhz: Vec<f64> = (-bank_k..=bank_k)
            .map(|k| (k as f64) * bank_step_hz)
            .collect();
        let bank_domega: Vec<f64> = bank_dhz
            .iter()
            // `omega` is stored as the phase advance across the data segment.
            .map(|&df_hz| 2.0 * std::f64::consts::PI * df_hz * t_data)
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
        let mut pre_u = vec![Complex32::new(0.0, 0.0); p.sf()];
        let rot_theta0 = Complex32::from_polar(1.0, -(theta as f32));
        for i in 0..n_finger {
            for j in 0..p.sf() {
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

    fn pll_predict(theta: &mut f64, omega_used: f64, sym_scale: f64) {
        *theta = wrap_pm_pi(*theta + omega_used * sym_scale);
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
        llr_out: &mut [f64],
        q_data: &mut usize,
    ) -> anyhow::Result<()> {
        let p = self.p;
        let chip_step = self.dll.sym_step_samp / (p.sf() as f64);
        for i in 0..self.n_finger {
            let buf = &mut self.buffers.u_p_fingers[i];
            Self::sample_symbol_into(self.y, self.t_sym0[i], chip_step, 0.0, buf)
                .ok_or_else(|| anyhow::anyhow!("insufficient_samples"))?;
            Self::demask_in_place(self.c_seq, p.sf(), ell, buf);
        }

        // Spec §3.D
        if ell == 1 {
            for u in &mut self.buffers.u_p_fingers {
                for v in u {
                    *v = -*v;
                }
            }
        }

        let outcome = if ell < p.n_pre() || is_pilot(ell) {
            self.process_pilot_symbol(ell)
        } else {
            self.process_data_symbol(ell, llr_out, q_data)?
        };

        self.apply_freq_snap(outcome.freq_conf, outcome.best_dhz);

        let phase_adj = if let Some(update) = outcome.dll_update {
            self.update_dll(ell, chip_step, update)?
        } else {
            0.0
        };

        // Advance by:
        // - the nominal (tracked) data-chips span
        // - the receiver's fractional timing adjustment
        // - the known guard/noise span for this symbol
        let j_ell = *self
            .j_seq
            .get(ell)
            .ok_or_else(|| anyhow::anyhow!("j_seq out of range"))? as f64;
        let gi_samp = j_ell * chip_step;
        for t in &mut self.t_sym0 {
            *t += self.dll.sym_step_samp + phase_adj + gi_samp;
        }
        Ok(())
    }

    fn process_pilot_symbol(&mut self, ell: usize) -> SymbolProcOutcome {
        let p = self.p;
        let kp = self.pll_kp;
        let ki = self.pll_ki;
        let omega_lim = self.omega_lim;
        let sf = p.sf();

        let mut omega_used = self.omega;
        let mut best_dhz = 0.0f64;

        let mut best_v = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for (h, &domega) in self.bank_domega.iter().enumerate() {
            let omega_h = (self.omega + domega).clamp(-self.omega_lim, self.omega_lim);
            Self::fill_rot_chips(&mut self.buffers.rot_tmp, sf, self.theta, omega_h);

            self.buffers.u_tmp.fill(Complex32::new(0.0, 0.0));
            for j in 0..p.sf() {
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
            for j in 0..sf {
                let rotj = self.buffers.rot_best[j];
                for i in 0..self.n_finger {
                    self.buffers.u_best[j] +=
                        self.w_mrc[i] * (self.buffers.u_p_fingers[i][j] * rotj);
                }
            }
            z_p = self.buffers.u_best.iter().copied().sum();
        }

        let err = (z_p.im as f64).atan2((z_p.re as f64) + 1e-18);
        let sym_scale = 1.0 + (self.j_seq[ell] as f64) / (p.sf() as f64);
        Self::pll_predict(&mut self.theta, omega_used, sym_scale);
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
                for j in 0..p.sf() {
                    z_i += self.buffers.u_p_fingers[i][j] * self.buffers.rot_best[j];
                }
                let gi_meas = z_i / (p.sf() as f32);
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
        ell: usize,
        llr_out: &mut [f64],
        q_data: &mut usize,
    ) -> anyhow::Result<SymbolProcOutcome> {
        let p = self.p;
        let kp = self.pll_kp;
        let ki = self.pll_ki;
        let omega_lim = self.omega_lim;
        let sf = p.sf();
        let mw = p.mw();

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
            for j in 0..p.sf() {
                let rotj = self.buffers.rot_tmp[j];
                for i in 0..self.n_finger {
                    self.buffers.u_tmp[j] +=
                        self.w_mrc[i] * (self.buffers.u_p_fingers[i][j] * rotj);
                }
            }
            self.buffers.r_tmp.clone_from(&self.buffers.u_tmp);
            fht1024_in_place(&mut self.buffers.r_tmp)?;
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
        let q = *q_data;
        if q >= p.n_data() {
            anyhow::bail!(
                "internal mapping error: data symbol index out of range (q_data={q}, n_data={})",
                p.n_data()
            );
        }
        let k = p.k_bits_per_sym();
        let llr_base = q * k;
        if llr_out.len() < llr_base + k {
            anyhow::bail!(
                "internal error: llr_out too small (len={}, need={})",
                llr_out.len(),
                llr_base + k
            );
        }

        let sym_scale = 1.0 + (self.j_seq[ell] as f64) / (p.sf() as f64);
        Self::pll_predict(&mut self.theta, omega_used, sym_scale);
        let mut rot_llr: Option<Complex32> = None;
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

            let den = z_dd.norm();
            if den > 1e-6 {
                // Decision-directed per-symbol phase alignment for soft-demapping (LLRs).
                rot_llr = Some(z_dd.conj() / den);
            }

            Some(DllUpdate {
                dd: true,
                m: best_i,
            })
        } else {
            None
        };

        if k > 8 {
            anyhow::bail!("k_bits_per_sym too large (k={k})");
        }
        let mut m0 = [f32::NEG_INFINITY; 8];
        let mut m1 = [f32::NEG_INFINITY; 8];
        for m in 0..mw {
            let v = self.buffers.r_best[m];
            let v_re = match rot_llr {
                Some(rot) => (v * rot).re,
                None => v.re,
            };
            let m_u16 = m as u16;
            for t in 0..k {
                let bit = (m_u16 >> (k - 1 - t)) & 1;
                if bit == 0 {
                    if v_re > m0[t] {
                        m0[t] = v_re;
                    }
                } else if v_re > m1[t] {
                    m1[t] = v_re;
                }
            }
        }
        for t in 0..k {
            llr_out[llr_base + t] = (m0[t] - m1[t]) as f64;
        }

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
                    self.omega = (self.omega + 2.0 * std::f64::consts::PI * df2 * self.t_data)
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
            Self::demask_in_place(self.c_seq, p.sf(), ell, &mut self.buffers.tmp_e);
            Self::demask_in_place(self.c_seq, p.sf(), ell, &mut self.buffers.tmp_l);
            for j in 0..p.sf() {
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
            let mut se = Complex32::new(0.0, 0.0);
            let mut sl = Complex32::new(0.0, 0.0);
            let m = update.m as u16;
            for j in 0..p.sf() {
                let wf = walsh_sign(m, j) as f32;
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
    ///
    /// This variant accepts raw samples, optionally derotates CFO at the sample rate, then applies
    /// the RRC matched filter internally.
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
        if n_offset_total.is_empty() {
            return Ok((None, DecodeMeta::error("no_offsets")));
        }

        let g = gen_code_structured_aes_ctr(
            &self.key,
            ti_tx,
            p.n_sym(),
            p.sf(),
            p.domain_u32(),
            p.hop_bw_hz(),
            JitterSpec {
                min_chips: p.jitter_min_chips(),
                span_chips: p.jitter_span_chips(),
            },
        );

        // Copy so we can apply CFO derotation and de-hopping before matched filtering.
        // We'll reuse the CFO-corrected buffer across small offset refinements.
        let mut x_cfo = rx_samples.to_vec();
        if cfo_hz != 0.0 {
            derotate_cfo_in_place(&mut x_cfo, p.fs_hz(), cfo_hz);
        }
        let mut x_work = x_cfo.clone();

        // With frequency-hopped waveforms, a ±1 sample error in `n0` can introduce large,
        // hop-dependent phase jumps across symbols. Under high noise, acquisition can still
        // "capture" but return `n0` off by a small number of samples. We therefore try a small
        // symmetric refinement window around the provided offsets, returning the first CRC-pass.
        let refine_win: isize = 6;
        let mut deltas: Vec<isize> = Vec::with_capacity(2 * (refine_win as usize) + 1);
        deltas.push(0);
        for d in 1..=refine_win {
            deltas.push(-d);
            deltas.push(d);
        }

        let mut first_fail: Option<(Option<Vec<u8>>, DecodeMeta)> = None;
        for delta in deltas {
            let mut offs_adj: Vec<usize> = Vec::with_capacity(n_offset_total.len());
            let mut ok = true;
            for &o in n_offset_total {
                let v = (o as isize) + delta;
                if v < 0 {
                    ok = false;
                    break;
                }
                offs_adj.push(v as usize);
            }
            if !ok {
                continue;
            }

            let n0 = *offs_adj
                .first()
                .ok_or_else(|| anyhow::anyhow!("no_offsets"))?;
            let start = frame_start_sample
                .checked_add(n0)
                .ok_or_else(|| anyhow::anyhow!("frame_start overflow"))?;
            if start >= x_cfo.len() {
                continue;
            }

            x_work.copy_from_slice(&x_cfo);
            apply_hop_with_start_in_place(
                &mut x_work,
                HopParams {
                    fs_hz: p.fs_hz(),
                    sf: p.sf(),
                    osf: p.osf() as usize,
                    n_tail_syms: p.n_tail(),
                },
                &g.f_seq_hz,
                &g.j_seq_chips,
                start,
                -1.0,
            );

            let y = self.rrc.filter_same(&x_work);
            let (payload, meta) = match self.demod_decode_matched(
                &y,
                ti_tx,
                frame_start_sample,
                &offs_adj,
                scl_list_size,
            ) {
                Ok(v) => v,
                Err(e) => {
                    // Some deltas can push the symbol sampler out of range. Treat that as a
                    // non-fatal miss and keep searching; but don't hide unexpected internal errors.
                    let msg = e.to_string();
                    if msg.contains("insufficient_samples") {
                        continue;
                    }
                    return Err(e);
                }
            };
            if meta.crc_ok {
                return Ok((payload, meta));
            }
            if first_fail.is_none() {
                first_fail = Some((payload, meta));
            }
        }

        Ok(first_fail.unwrap_or((None, DecodeMeta::error("insufficient_samples"))))
    }

    /// Spec §4.C–§4.D.
    ///
    /// This variant expects samples that are already RRC matched-filtered.
    ///
    /// Note: for frequency-hopped waveforms, the de-hopping operation must be applied before the
    /// matched filter. Use [`ScBltcModem::demod_decode_raw`] unless the caller already did the
    /// de-hopping on raw samples.
    pub fn demod_decode_matched(
        &self,
        y_matched: &[Complex32],
        ti_tx: u64,
        frame_start_sample: usize,
        n_offset_total: &[usize],
        scl_list_size: usize,
    ) -> anyhow::Result<(Option<Vec<u8>>, DecodeMeta)> {
        let p = &self.p;
        if n_offset_total.is_empty() {
            return Ok((None, DecodeMeta::error("no_offsets")));
        }
        let y = y_matched;

        let g = gen_code_structured_aes_ctr(
            &self.key,
            ti_tx,
            p.n_sym(),
            p.sf(),
            p.domain_u32(),
            p.hop_bw_hz(),
            JitterSpec {
                min_chips: p.jitter_min_chips(),
                span_chips: p.jitter_span_chips(),
            },
        );
        let c_seq = g.c_seq;
        let j_seq = g.j_seq_chips;

        let cascade_delay = 2 * self.rrc.delay();
        let frame_start = frame_start_sample as f64;

        let n_finger = n_offset_total.len().min(3);
        let mut tracker = SymbolTracker::new(
            p,
            y,
            &c_seq,
            &j_seq,
            n_finger,
            TrackerTiming {
                n_offset_total,
                frame_start,
                cascade_delay,
            },
        )?;

        // Spec §4.D.2–§4.D.3: fill LLRs incrementally while processing each data symbol.
        let mut llr = [0f64; 512];
        let mut q_data = 0usize;
        for ell in 0..p.n_sym() {
            tracker.process_symbol(ell, &mut llr, &mut q_data)?;
        }

        if q_data != p.n_data() {
            return Ok((None, DecodeMeta::error("data_symbol_count_mismatch")));
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
