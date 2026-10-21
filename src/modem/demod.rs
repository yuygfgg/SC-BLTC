//! Tracking demodulator and decoder (Specification §4.C-§4.D).
//!
//! Given:
//! - an acquired `ti_tx` (spreading seed)
//! - the receiver sample index of a selected reference arrival path
//! - one or more RAKE finger delays relative to that path
//!
//! This module:
//! 1) generates the full-frame chip mask `C_seq`
//! 2) samples chips for each finger with a DLL (early/late gate) and linear interpolation
//! 3) performs a local joint timing/carrier search on pilot symbols, then runs the PLL/DLL loops
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

// A 0.25 Hz bank step is deliberately accumulated over symbols. Adjacent candidates
// differ by less than one percent over a single 204.8 ms data window.
const FREQ_EVIDENCE_DECAY: f64 = 0.95;
const FREQ_EVIDENCE_MIN_SYMBOLS: usize = 4;
const FREQ_SNAP_EVIDENCE_MIN: f32 = 0.05;
const MAX_DEMOD_FINGERS: usize = 3;
const JOINT_TIMING_RADIUS_SAMP: f64 = 1.5;
const JOINT_TIMING_STEP_SAMP: f64 = 0.5;
const JOINT_TIMING_MIN_GAIN: f32 = 0.005;
const JOINT_FREQ_BLEND: f64 = 0.1;
const DLL_BW_SINGLE_PATH_HZ: f64 = 2.0;
const DLL_BW_RAKE_HZ: f64 = 0.6;
const DLL_DAMPING: f64 = 0.707;

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
    freq_evidence: f32,
    best_freq_hz: f64,
    dll_update: Option<DllUpdate>,
}

/// Coupled carrier and timing state used by the symbol tracker.
///
/// The carrier loop and the DLL still use their own loop gains, but they now
/// share one state object. Pilot updates can therefore apply a timing
/// correction and a frequency correction from the same matched-filter metric.
struct JointTrackingState {
    dll: EarlyLateDll,
    theta: f64,
    omega: f64,
    // Common timing state. `t_sym0` stores the corresponding absolute times per finger.
    timing_offset_samp: f64,
}

impl JointTrackingState {
    fn new(dll: EarlyLateDll, theta: f64, omega: f64) -> Self {
        Self {
            dll,
            theta,
            omega,
            timing_offset_samp: 0.0,
        }
    }

    fn apply_joint_measurement(&mut self, timing_delta_samp: f64, omega: f64) {
        if timing_delta_samp.is_finite() {
            self.timing_offset_samp += timing_delta_samp;
        }
        if omega.is_finite() {
            self.omega = omega;
        }
    }
}

struct SymbolTracker<'a> {
    p: &'a crate::params::Params,
    y: &'a [Complex32],
    c_seq: &'a [i8],
    j_seq: &'a [usize],
    f_seq_hz: &'a [f64],
    n_finger: usize,
    // Integer sample delay relative to finger 0 (main path).
    d_finger_samp: Vec<i32>,
    t_sym0: Vec<f64>,
    tracking: JointTrackingState,
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
    freq_likelihood: Vec<f64>,
    /// Absolute-frequency center represented by `freq_likelihood`.
    ///
    /// The PLL center can move between symbols. Keeping this center lets us
    /// regrid historical evidence before adding the next symbol's metrics.
    freq_likelihood_center_hz: f64,
    freq_evidence_symbols: usize,
    snap_cand_hz: Option<f64>,
    snap_count: usize,
    freq_snap_confirm: usize,
    freq_snap_min_abs_hz: f64,
    alpha_ch: f64,
    el_spacing_chips: f64,
    // Spec §4.C: verification statistic for arrival-sample refinement (preamble + pilots).
    verify_pre: f64,
    verify_pilot: f64,
    buffers: SymbolBuffers,
}

struct TrackerTiming<'a> {
    n_offset_total: &'a [isize],
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
        f_seq_hz: &'a [f64],
        n_finger: usize,
        timing: TrackerTiming<'a>,
    ) -> anyhow::Result<Self> {
        let d_finger_samp = Self::init_finger_delays_samp(timing.n_offset_total, n_finger);
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
        let dll = Self::init_dll(p, t_update, n_finger);
        let chip_step0 = dll.sym_step_samp / (p.sf() as f64);

        let (mut u0_fingers, mut u1_fingers) = Self::sample_initial_fingers(
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
        let fs_hz = p.fs_hz() as f64;
        if let Some(&f0) = f_seq_hz.first() {
            Self::apply_hop_delay_phase_comp_in_place(&mut u0_fingers, f0, &d_finger_samp, fs_hz);
        }
        if f_seq_hz.len() >= 2 {
            Self::apply_hop_delay_phase_comp_in_place(
                &mut u1_fingers,
                f_seq_hz[1],
                &d_finger_samp,
                fs_hz,
            );
        }
        let pre_corr = Self::pre_corr(&u0_fingers, &u1_fingers);
        let (theta, g) = Self::init_phase_and_channel(&pre_corr, p.n_pre(), p.sf(), n_finger);
        let w_mrc = Self::mrc_weights(&g);

        let (pll_kp, pll_ki, omega_lim) = Self::init_pll(t_update, t_data);
        let tracking = JointTrackingState::new(dll, theta, 0.0);

        let (bank_dhz, bank_domega, bank_step_hz) = Self::init_freq_bank(t_data);
        let bank_len = bank_dhz.len();
        let pre_mag_ref = Self::preamble_mag_ref(p, n_finger, &u0_fingers, &w_mrc, tracking.theta);
        let alpha_ch = 1.0 / (p.n_pilot() as f64);
        let verify_pre: f64 = pre_corr.iter().map(|z| z.norm_sqr() as f64).sum();

        Ok(Self {
            p,
            y,
            c_seq,
            j_seq,
            f_seq_hz,
            n_finger,
            d_finger_samp,
            t_sym0,
            tracking,
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
            freq_likelihood: vec![0.0; bank_len],
            freq_likelihood_center_hz: 0.0,
            freq_evidence_symbols: 0,
            snap_cand_hz: None,
            snap_count: 0,
            freq_snap_confirm: 3,
            freq_snap_min_abs_hz: 0.75,
            alpha_ch,
            el_spacing_chips: 0.5,
            verify_pre,
            verify_pilot: 0.0,
            buffers: SymbolBuffers::new(n_finger, p.sf()),
        })
    }

    fn init_finger_delays_samp(n_offset_total: &[isize], n_finger: usize) -> Vec<i32> {
        let n = n_finger.min(n_offset_total.len());
        if n == 0 {
            return Vec::new();
        }
        let n0 = n_offset_total[0] as i64;
        (0..n)
            .map(|i| (n_offset_total[i] as i64 - n0).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            .collect()
    }

    fn hop_delay_comp_phasor(f_hz: f64, d_samp: i32, fs_hz: f64) -> Complex32 {
        if d_samp == 0 || f_hz == 0.0 || fs_hz == 0.0 {
            return Complex32::new(1.0, 0.0);
        }
        let phi = 2.0 * std::f64::consts::PI * f_hz * (d_samp as f64) / fs_hz;
        Complex32::from_polar(1.0, phi as f32)
    }

    fn apply_hop_delay_phase_comp_in_place(
        fingers: &mut [FingerVec],
        f_hz: f64,
        d_finger_samp: &[i32],
        fs_hz: f64,
    ) {
        for (i, chips) in fingers.iter_mut().enumerate() {
            let d = d_finger_samp.get(i).copied().unwrap_or(0);
            if d == 0 {
                continue;
            }
            let rot = Self::hop_delay_comp_phasor(f_hz, d, fs_hz);
            for v in chips {
                *v *= rot;
            }
        }
    }

    fn verify_score(&self) -> f32 {
        let v = self.verify_pre + self.verify_pilot;
        if v.is_finite() && v >= 0.0 {
            v as f32
        } else {
            0.0
        }
    }

    fn mrc_weights(g_est: &[Complex32]) -> Vec<Complex32> {
        let den: f32 = g_est.iter().map(|v| v.norm_sqr()).sum::<f32>() + 1e-18;
        g_est.iter().map(|&v| v.conj() / den).collect()
    }

    fn init_symbol_times(
        n_offset_total: &[isize],
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

    fn init_dll(p: &crate::params::Params, tsym: f64, n_finger: usize) -> EarlyLateDll {
        // A single-path channel has a clean early/late discriminator and needs more
        // bandwidth to follow a sampling-rate offset. Multipath combines several
        // delayed discriminators, so retain the narrower loop when RAKE combining is active.
        let loop_bw_hz = if n_finger <= 1 {
            DLL_BW_SINGLE_PATH_HZ
        } else {
            DLL_BW_RAKE_HZ
        };
        let dll_g = design_2nd_order_loop(loop_bw_hz, DLL_DAMPING, tsym);
        let sym_step_nom = (p.sf() as f64) * (p.osf() as f64);
        let sym_step_ppm = crate::params::MAX_SRO_PPM;
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
            // Spec §3.D: preamble symbol 1 is `-W0`, so after de-masking with `C_seq` we flip
            // the whole symbol to keep preamble symbols coherent with symbol 0.
            for v in &mut y1 {
                *v = -*v;
            }
            u1_fingers.push(y1);
        }
        Ok((u0_fingers, u1_fingers))
    }

    fn pre_corr(u0_fingers: &[FingerVec], u1_fingers: &[FingerVec]) -> Vec<Complex32> {
        (0..u0_fingers.len())
            .map(|i| {
                let z0: Complex32 = u0_fingers[i].iter().copied().sum();
                let z1: Complex32 = u1_fingers[i].iter().copied().sum();
                z0 + z1
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

    // The adjacent-codeword margin still gates data decisions. Frequency confidence
    // uses the cross-symbol likelihood accumulator below.
    fn best_code_and_conf_mag(r_all: &[Complex32], mw: usize) -> (usize, f32) {
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

    fn accumulate_frequency_likelihood(
        likelihood: &mut [f64],
        evidence_symbols: &mut usize,
        bank_dhz: &[f64],
        metrics: &[f32],
    ) -> (f32, f64) {
        debug_assert_eq!(metrics.len(), likelihood.len());
        debug_assert_eq!(bank_dhz.len(), likelihood.len());
        if metrics.len() != likelihood.len() || bank_dhz.len() != likelihood.len() {
            return (0.0, 0.0);
        }

        let peak = metrics
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .fold(0.0f32, f32::max);
        if peak <= 0.0 {
            return (0.0, 0.0);
        }

        *evidence_symbols = (*evidence_symbols).saturating_add(1);
        for (acc, &metric) in likelihood.iter_mut().zip(metrics) {
            // Normalize within this symbol so the memory tracks relative frequency
            // likelihood instead of absolute channel power.
            let normalized = if metric.is_finite() && metric >= 0.0 {
                (metric / peak).clamp(0.0, 1.0) as f64
            } else {
                0.0
            };
            *acc = FREQ_EVIDENCE_DECAY * *acc + normalized;
        }

        let mut best_i = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        let mut second = f64::NEG_INFINITY;
        for (i, &score) in likelihood.iter().enumerate() {
            if score > best_v {
                second = best_v;
                best_v = score;
                best_i = i;
            } else if score > second {
                second = score;
            }
        }
        let evidence = if *evidence_symbols >= FREQ_EVIDENCE_MIN_SYMBOLS {
            (best_v - second).max(0.0) as f32
        } else {
            0.0
        };
        (evidence, bank_dhz[best_i])
    }

    fn rebase_frequency_likelihood(&mut self) {
        let center_hz = self.tracking.omega / (2.0 * std::f64::consts::PI * self.t_data);
        let old_center_hz = self.freq_likelihood_center_hz;
        if !center_hz.is_finite()
            || !old_center_hz.is_finite()
            || (center_hz - old_center_hz).abs() <= f64::EPSILON
            || self.bank_step_hz <= 0.0
            || self.bank_dhz.is_empty()
        {
            return;
        }

        let center_shift_hz = (center_hz - old_center_hz).abs();
        let bank_span_hz = self.bank_step_hz * (self.bank_dhz.len().saturating_sub(1) as f64);
        if center_shift_hz > bank_span_hz {
            self.freq_likelihood.fill(0.0);
            self.freq_evidence_symbols = 0;
            self.snap_cand_hz = None;
            self.snap_count = 0;
            self.freq_likelihood_center_hz = center_hz;
            return;
        }

        // Preserve evidence by absolute frequency. Linear interpolation is
        // sufficient because the bank spacing is uniform and the PLL moves
        // the center by much less than one bank step in normal operation.
        let old = self.freq_likelihood.clone();
        let first_dhz = self.bank_dhz[0];
        for (i, score) in self.freq_likelihood.iter_mut().enumerate() {
            let target_hz = center_hz + self.bank_dhz[i];
            let old_pos = (target_hz - old_center_hz - first_dhz) / self.bank_step_hz;
            if old_pos < 0.0 || old_pos > (old.len().saturating_sub(1) as f64) {
                *score = 0.0;
                continue;
            }
            let lo = old_pos.floor() as usize;
            let hi = (lo + 1).min(old.len() - 1);
            let frac = (old_pos - lo as f64).clamp(0.0, 1.0);
            *score = old[lo] * (1.0 - frac) + old[hi] * frac;
        }
        self.freq_likelihood_center_hz = center_hz;
    }

    fn accumulate_current_frequency_likelihood(&mut self, metrics: &[f32]) -> (f32, f64) {
        self.rebase_frequency_likelihood();
        let (evidence, best_dhz) = Self::accumulate_frequency_likelihood(
            &mut self.freq_likelihood,
            &mut self.freq_evidence_symbols,
            &self.bank_dhz,
            metrics,
        );
        // `best_dhz` is relative to the center used for this symbol. Return
        // an absolute frequency so a later PLL update cannot shift the snap.
        (evidence, self.freq_likelihood_center_hz + best_dhz)
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

    /// Search timing and carrier hypotheses from the same pilot correlation.
    ///
    /// The existing DLL and PLL correct one dimension at a time. A sample-rate
    /// error changes both dimensions over the frame, so a pilot can be stronger
    /// at a nearby timing and frequency pair than at either independent loop's
    /// current point. This small local search supplies one coupled measurement;
    /// the regular DLL and PLL then continue with their normal loop updates.
    fn apply_joint_pilot_search(&mut self, ell: usize, chip_step: f64) -> anyhow::Result<()> {
        if !is_pilot(ell) || self.n_finger == 0 {
            return Ok(());
        }

        let p = self.p;
        let sf = p.sf();
        let f_hz = *self.f_seq_hz.get(ell).unwrap_or(&0.0);
        let fs_hz = p.fs_hz() as f64;
        let n_steps = (JOINT_TIMING_RADIUS_SAMP / JOINT_TIMING_STEP_SAMP).round() as i32;
        let mut best_score = f32::NEG_INFINITY;
        let mut center_score = f32::NEG_INFINITY;
        let mut best_shift = 0.0f64;
        let mut best_omega = self.tracking.omega;

        for shift_i in -n_steps..=n_steps {
            let shift = (shift_i as f64) * JOINT_TIMING_STEP_SAMP;
            self.buffers.u_tmp.fill(Complex32::new(0.0, 0.0));
            let mut valid = true;
            for i in 0..self.n_finger {
                if Self::sample_symbol_into(
                    self.y,
                    self.t_sym0[i],
                    chip_step,
                    shift,
                    &mut self.buffers.tmp_e,
                )
                .is_none()
                {
                    valid = false;
                    break;
                }
                Self::demask_in_place(self.c_seq, sf, ell, &mut self.buffers.tmp_e);
                let d = self.d_finger_samp.get(i).copied().unwrap_or(0);
                let rot_d = Self::hop_delay_comp_phasor(f_hz, d, fs_hz);
                for j in 0..sf {
                    self.buffers.u_tmp[j] += self.w_mrc[i] * (self.buffers.tmp_e[j] * rot_d);
                }
            }
            if !valid {
                continue;
            }

            for &domega in &self.bank_domega {
                let omega_h = (self.tracking.omega + domega).clamp(-self.omega_lim, self.omega_lim);
                Self::fill_rot_chips(&mut self.buffers.rot_tmp, sf, self.tracking.theta, omega_h);
                let z: Complex32 = self
                    .buffers
                    .u_tmp
                    .iter()
                    .zip(&self.buffers.rot_tmp)
                    .map(|(&u, &rot)| u * rot)
                    .sum();
                let score = z.norm_sqr();
                if shift_i == 0 {
                    center_score = center_score.max(score);
                }
                if score > best_score {
                    best_score = score;
                    best_shift = shift;
                    best_omega = omega_h;
                }
            }
        }

        if best_score.is_finite() && center_score.is_finite() {
            let gain = (best_score - center_score) / center_score.max(1e-12);
            let accepted_shift = if gain >= JOINT_TIMING_MIN_GAIN {
                best_shift
            } else {
                0.0
            };
            let omega = if accepted_shift != 0.0 {
                self.tracking.omega + JOINT_FREQ_BLEND * (best_omega - self.tracking.omega)
            } else {
                self.tracking.omega
            };
            for t in &mut self.t_sym0 {
                *t += accepted_shift;
            }
            self.tracking.apply_joint_measurement(accepted_shift, omega);
        }
        Ok(())
    }

    fn process_symbol(
        &mut self,
        ell: usize,
        llr_out: &mut [f64],
        q_data: &mut usize,
    ) -> anyhow::Result<()> {
        let p = self.p;
        let chip_step = self.tracking.dll.sym_step_samp / (p.sf() as f64);
        self.apply_joint_pilot_search(ell, chip_step)?;
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

        // Spec §4.C: hop-delay phase compensation per finger:
        // u_{i,ell} *= exp(+j 2π f_ell Δd_i / Fs).
        let f_hz = *self.f_seq_hz.get(ell).unwrap_or(&0.0);
        if f_hz != 0.0 {
            let fs_hz = p.fs_hz() as f64;
            for i in 0..self.n_finger {
                let d = self.d_finger_samp.get(i).copied().unwrap_or(0);
                if d == 0 {
                    continue;
                }
                let rot = Self::hop_delay_comp_phasor(f_hz, d, fs_hz);
                for v in &mut self.buffers.u_p_fingers[i] {
                    *v *= rot;
                }
            }
        }

        let outcome = if ell < p.n_pre() || is_pilot(ell) {
            self.process_pilot_symbol(ell)
        } else {
            self.process_data_symbol(ell, llr_out, q_data)?
        };

        self.apply_freq_snap(outcome.freq_evidence, outcome.best_freq_hz);

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
            *t += self.tracking.dll.sym_step_samp + phase_adj + gi_samp;
        }
        Ok(())
    }

    fn process_pilot_symbol(&mut self, ell: usize) -> SymbolProcOutcome {
        let p = self.p;
        let kp = self.pll_kp;
        let ki = self.pll_ki;
        let omega_lim = self.omega_lim;
        let sf = p.sf();

        let mut omega_used = self.tracking.omega;

        let mut freq_metrics = vec![0.0f32; self.bank_domega.len()];
        let mut best_v = f32::NEG_INFINITY;
        for (h, &domega) in self.bank_domega.iter().enumerate() {
            let omega_h = (self.tracking.omega + domega).clamp(-self.omega_lim, self.omega_lim);
            Self::fill_rot_chips(&mut self.buffers.rot_tmp, sf, self.tracking.theta, omega_h);

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
            freq_metrics[h] = m;
            if m > best_v {
                best_v = m;
                omega_used = omega_h;
                self.buffers.rot_best.clone_from(&self.buffers.rot_tmp);
                self.buffers.u_best.clone_from(&self.buffers.u_tmp);
            }
        }
        let (freq_evidence, best_freq_hz) =
            self.accumulate_current_frequency_likelihood(&freq_metrics);

        // Cycle-slip guard.
        let mut z_p: Complex32 = self.buffers.u_best.iter().copied().sum();
        if (z_p.re as f64) < 0.0 && z_p.norm() > 0.25 * self.pre_mag_ref {
            self.tracking.theta = wrap_pm_pi(self.tracking.theta + std::f64::consts::PI);
            for gi in &mut self.g {
                *gi = -*gi;
            }
            self.w_mrc = Self::mrc_weights(&self.g);

            Self::fill_rot_chips(
                &mut self.buffers.rot_best,
                sf,
                self.tracking.theta,
                omega_used,
            );
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

        // Spec §4.C: accumulate pilot verification energy.
        if is_pilot(ell) {
            self.verify_pilot += z_p.norm_sqr() as f64;
        }

        let err = (z_p.im as f64).atan2((z_p.re as f64) + 1e-18);
        let sym_scale = 1.0 + (self.j_seq[ell] as f64) / (p.sf() as f64);
        Self::pll_predict(&mut self.tracking.theta, omega_used, sym_scale);
        if err.is_finite() && z_p.norm() > 1e-6 {
            Self::pll_correct(
                &mut self.tracking.theta,
                &mut self.tracking.omega,
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
            freq_evidence,
            best_freq_hz,
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

        let mut omega_used = self.tracking.omega;

        let mut best_i = 0usize;
        let mut freq_metrics = vec![0.0f32; self.bank_domega.len()];
        let mut best_v = f32::NEG_INFINITY;
        let mut best_code_conf = 0.0f32;
        for (h, &domega) in self.bank_domega.iter().enumerate() {
            let omega_h = (self.tracking.omega + domega).clamp(-self.omega_lim, self.omega_lim);
            Self::fill_rot_chips(&mut self.buffers.rot_tmp, sf, self.tracking.theta, omega_h);

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
            let (i_h, conf_h) = Self::best_code_and_conf_mag(&self.buffers.r_tmp, mw);
            let m = self.buffers.r_tmp[i_h].norm_sqr();
            freq_metrics[h] = m;
            if m > best_v {
                best_v = m;
                omega_used = omega_h;
                best_i = i_h;
                best_code_conf = conf_h;
                self.buffers.rot_best.clone_from(&self.buffers.rot_tmp);
                self.buffers.r_best.clone_from(&self.buffers.r_tmp);
            }
        }
        let (freq_evidence, best_freq_hz) =
            self.accumulate_current_frequency_likelihood(&freq_metrics);
        // Gate PLL/DD only on code confidence; frequency evidence is accumulated separately.
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
        Self::pll_predict(&mut self.tracking.theta, omega_used, sym_scale);
        let mut rot_llr: Option<Complex32> = None;
        let dll_update = if data_conf.is_finite() && data_conf > 0.10 {
            let mut z_dd = self.buffers.r_best[best_i];
            if z_dd.re < 0.0 {
                z_dd = -z_dd;
            }
            let err = (z_dd.im as f64).atan2((z_dd.re as f64) + 1e-18);
            if err.is_finite() && z_dd.norm() > 1e-6 {
                Self::pll_correct(
                    &mut self.tracking.theta,
                    &mut self.tracking.omega,
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
            freq_evidence,
            best_freq_hz,
            dll_update,
        })
    }

    fn apply_freq_snap(&mut self, freq_evidence: f32, best_freq_hz: f64) {
        if !(freq_evidence.is_finite()
            && freq_evidence >= FREQ_SNAP_EVIDENCE_MIN
            && best_freq_hz.is_finite())
        {
            self.snap_cand_hz = None;
            self.snap_count = 0;
            return;
        }

        let candidate_hz = best_freq_hz;
        if let Some(cand) = self.snap_cand_hz {
            if (candidate_hz - cand).abs() <= 0.5 * self.bank_step_hz {
                self.snap_count += 1;
            } else {
                self.snap_cand_hz = Some(candidate_hz);
                self.snap_count = 1;
            }
        } else {
            self.snap_cand_hz = Some(candidate_hz);
            self.snap_count = 1;
        }
        if self.snap_count >= self.freq_snap_confirm {
            let snap_hz = self.snap_cand_hz.unwrap_or(0.0);
            let center_hz = self.tracking.omega / (2.0 * std::f64::consts::PI * self.t_data);
            let correction_hz = snap_hz - center_hz;
            if snap_hz.abs() >= self.freq_snap_min_abs_hz {
                self.tracking.omega = (self.tracking.omega
                    + 2.0 * std::f64::consts::PI * correction_hz * self.t_data)
                    .clamp(-self.omega_lim, self.omega_lim);
            }
            self.freq_likelihood.fill(0.0);
            self.freq_likelihood_center_hz =
                self.tracking.omega / (2.0 * std::f64::consts::PI * self.t_data);
            self.freq_evidence_symbols = 0;
            self.snap_cand_hz = None;
            self.snap_count = 0;
        }
    }

    fn update_dll(&mut self, ell: usize, chip_step: f64, update: DllUpdate) -> anyhow::Result<f64> {
        let p = self.p;
        let f_hz = *self.f_seq_hz.get(ell).unwrap_or(&0.0);
        let fs_hz = p.fs_hz() as f64;
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
            let d = self.d_finger_samp.get(i).copied().unwrap_or(0);
            let rot_d = Self::hop_delay_comp_phasor(f_hz, d, fs_hz);
            for j in 0..p.sf() {
                let rotj = self.buffers.rot_best[j];
                self.buffers.u_e[j] += self.w_mrc[i] * ((self.buffers.tmp_e[j] * rot_d) * rotj);
                self.buffers.u_l[j] += self.w_mrc[i] * ((self.buffers.tmp_l[j] * rot_d) * rotj);
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
            phase_adj = self.tracking.dll.update(err_samp, update.dd);
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
        n_offset_total: &[isize],
        cfo_hz: f64,
        scl_list_size: usize,
    ) -> anyhow::Result<(Option<Vec<u8>>, DecodeMeta)> {
        if n_offset_total.is_empty() {
            return Ok((None, DecodeMeta::error("no_offsets")));
        }

        // Acquisition orders fingers by descending preamble energy and keeps the first
        // offset as the reference path. Start with that path, then add weaker paths only
        // when the current decode does not pass CRC.
        let n_finger = n_offset_total.len().min(MAX_DEMOD_FINGERS);
        let mut best_fail: Option<(Option<Vec<u8>>, DecodeMeta)> = None;
        let mut best_score = f32::NEG_INFINITY;
        for finger_count in 1..=n_finger {
            let result = self.demod_decode_raw_with_offsets(
                rx_samples,
                ti_tx,
                frame_start_sample,
                &n_offset_total[..finger_count],
                cfo_hz,
                scl_list_size,
            )?;
            if result.1.crc_ok {
                return Ok(result);
            }

            let score = result.1.verify_score;
            let better = score.is_finite()
                && (best_fail.is_none() || score > best_score || !best_score.is_finite());
            if better {
                best_score = score;
                best_fail = Some(result);
            } else if best_fail.is_none() {
                best_fail = Some(result);
            }
        }

        Ok(best_fail.unwrap_or((None, DecodeMeta::error("insufficient_samples"))))
    }

    fn demod_decode_raw_with_offsets(
        &self,
        rx_samples: &[Complex32],
        ti_tx: u64,
        frame_start_sample: usize,
        n_offset_total: &[isize],
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
        // With frequency-hopped waveforms, a ±1 sample error in the reference arrival can introduce large,
        // hop-dependent phase jumps across symbols. Under high noise, acquisition can still
        // "capture" but return the arrival sample off by a small number of samples. We therefore try a small
        // symmetric refinement window around the provided offsets, returning the first CRC-pass.
        let refine_win: isize = 6;
        let mut deltas: Vec<isize> = Vec::with_capacity(2 * (refine_win as usize) + 1);
        deltas.push(0);
        for d in 1..=refine_win {
            deltas.push(-d);
            deltas.push(d);
        }

        // Best-effort fallback when all deltas fail CRC: pick the Δn with the strongest
        // preamble/pilots verification statistic (Spec §4.C).
        let mut best_fail: Option<(Option<Vec<u8>>, DecodeMeta)> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;
        for delta in deltas {
            let candidate_frame_start = if delta >= 0 {
                frame_start_sample
                    .checked_add(delta as usize)
                    .ok_or_else(|| anyhow::anyhow!("frame_start overflow"))?
            } else {
                frame_start_sample
                    .checked_sub(delta.unsigned_abs())
                    .ok_or_else(|| anyhow::anyhow!("frame_start before buffer"))?
            };

            let min_offset = *n_offset_total
                .iter()
                .min()
                .ok_or_else(|| anyhow::anyhow!("no_offsets"))?;
            let sample_start = if min_offset >= 0 {
                candidate_frame_start
                    .checked_add(min_offset as usize)
                    .ok_or_else(|| anyhow::anyhow!("frame_start overflow"))?
            } else {
                candidate_frame_start
                    .checked_sub(min_offset.unsigned_abs())
                    .ok_or_else(|| anyhow::anyhow!("frame_start before buffer"))?
            };
            if sample_start >= x_cfo.len() || candidate_frame_start >= x_cfo.len() {
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
                // The sample buffer may start at an earlier finger, but the hop schedule
                // starts at the candidate reference arrival.
                candidate_frame_start,
                -1.0,
            );

            let y = self.rrc.filter_same(&x_work);
            let (payload, meta) = match self.demod_decode_matched(
                &y,
                ti_tx,
                candidate_frame_start,
                n_offset_total,
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
            let score = meta.verify_score;
            let better = score.is_finite()
                && (best_fail.is_none() || score > best_score || !best_score.is_finite());
            if better {
                best_score = score;
                best_fail = Some((payload, meta));
            } else if best_fail.is_none() {
                best_fail = Some((payload, meta));
            }
        }

        Ok(best_fail.unwrap_or((None, DecodeMeta::error("insufficient_samples"))))
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
        n_offset_total: &[isize],
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
        let f_seq_hz = g.f_seq_hz;
        let j_seq = g.j_seq_chips;

        let cascade_delay = 2 * self.rrc.delay();
        let frame_start = frame_start_sample as f64;

        let n_finger = n_offset_total.len().min(MAX_DEMOD_FINGERS);
        let mut tracker = SymbolTracker::new(
            p,
            y,
            &c_seq,
            &j_seq,
            &f_seq_hz,
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

        let verify_score = tracker.verify_score();

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
            verify_score,
            err: None,
        };
        if !crc_ok {
            return Ok((None, meta));
        }
        Ok((Some(payload[..(hdr.length as usize)].to_vec()), meta))
    }
}

#[cfg(test)]
mod tests {
    use super::{SymbolTracker, FREQ_SNAP_EVIDENCE_MIN};

    #[test]
    fn frequency_evidence_accumulates_flat_adjacent_candidates() {
        let bank = [-0.25, 0.0, 0.25];
        let mut likelihood = [0.0; 3];
        let mut symbols = 0usize;
        let metrics = [0.9914f32, 1.0, 0.9914];
        let mut evidence = 0.0;
        let mut candidate = 0.0;
        for _ in 0..64 {
            (evidence, candidate) = SymbolTracker::accumulate_frequency_likelihood(
                &mut likelihood,
                &mut symbols,
                &bank,
                &metrics,
            );
        }
        assert_eq!(candidate, 0.0);
        assert!(
            evidence >= FREQ_SNAP_EVIDENCE_MIN,
            "flat single-symbol margin did not accumulate: evidence={evidence}"
        );
    }

    #[test]
    fn frequency_evidence_waits_for_multiple_symbols() {
        let bank = [-0.25, 0.0, 0.25];
        let mut likelihood = [0.0; 3];
        let mut symbols = 0usize;
        let metrics = [0.4f32, 1.0, 0.4];
        let (evidence, candidate) = SymbolTracker::accumulate_frequency_likelihood(
            &mut likelihood,
            &mut symbols,
            &bank,
            &metrics,
        );
        assert_eq!(candidate, 0.0);
        assert_eq!(evidence, 0.0);
    }

    #[test]
    fn frequency_evidence_reaches_snap_margin_after_history() {
        let bank = [-0.25, 0.0, 0.25];
        let mut likelihood = [0.0; 3];
        let mut symbols = 0usize;
        let metrics = [0.9914f32, 1.0, 0.9914];
        let mut evidence = 0.0;
        for _ in 0..4 {
            (evidence, _) = SymbolTracker::accumulate_frequency_likelihood(
                &mut likelihood,
                &mut symbols,
                &bank,
                &metrics,
            );
        }
        assert!(evidence < FREQ_SNAP_EVIDENCE_MIN);
        for _ in 0..3 {
            (evidence, _) = SymbolTracker::accumulate_frequency_likelihood(
                &mut likelihood,
                &mut symbols,
                &bank,
                &metrics,
            );
        }
        assert!(evidence >= FREQ_SNAP_EVIDENCE_MIN);
    }
}
