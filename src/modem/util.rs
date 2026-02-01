//! Small modem utilities shared by TX/acquisition/demodulation.
//!
//! These helpers mainly encode the fixed frame schedule and small signal-processing primitives.

use crate::rrc::Fir;
use num_complex::Complex32;

/// Parameters for per-symbol hopping that tend to travel together.
#[derive(Copy, Clone, Debug)]
pub(super) struct HopParams {
    pub fs_hz: u32,
    pub sf: usize,
    pub osf: usize,
    pub n_tail_syms: usize,
}

/// Whether spread symbol index `ell` is a pilot (Specification §3.A0).
///
/// Layout is:
/// - ell=0..1: preamble
/// - then 16 blocks of: `pilot + data + data + data + data`
pub(super) fn is_pilot(ell: usize) -> bool {
    // Spec §3 / §4: pilots start after the 2-symbol preamble.
    ell >= 2 && (ell - 2).is_multiple_of(5)
}

/// Wrap an angle to the range (-pi, +pi].
pub(super) fn wrap_pm_pi(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// Pack a flat `{0,1}` bit slice into `k`-bit symbol indices (MSB first).
pub(super) fn bits_to_symbols(bits: &[u8], k: usize) -> Vec<u16> {
    assert!(bits.len().is_multiple_of(k));
    let mut out = Vec::with_capacity(bits.len() / k);
    for chunk in bits.chunks_exact(k) {
        let mut v = 0u16;
        for &b in chunk {
            v = (v << 1) | (b as u16 & 1);
        }
        out.push(v);
    }
    out
}

/// Upsample chips by `osf` and apply the RRC filter (TX pulse shaping).
pub(super) fn pulse_shape_chips(chips: &[i8], fir: &Fir, osf: usize) -> Vec<Complex32> {
    let mut up = vec![Complex32::new(0.0, 0.0); chips.len() * osf];
    for (i, &c) in chips.iter().enumerate() {
        up[i * osf] = Complex32::new(c as f32, 0.0);
    }
    fir.filter_same(&up)
}

/// Upsample real-valued chips by `osf` and apply the RRC filter (TX pulse shaping).
pub(super) fn pulse_shape_real_chips(chips: &[f32], fir: &Fir, osf: usize) -> Vec<Complex32> {
    let mut up = vec![Complex32::new(0.0, 0.0); chips.len() * osf];
    for (i, &c) in chips.iter().enumerate() {
        up[i * osf] = Complex32::new(c, 0.0);
    }
    fir.filter_same(&up)
}

/// Apply a smooth cosine ramp-down to the tail of a waveform (Specification §3.E2).
pub(super) fn apply_ramp_down(x: &[Complex32], ramp_n: usize) -> Vec<Complex32> {
    if ramp_n <= 1 || x.is_empty() {
        return x.to_vec();
    }
    let n = ramp_n.min(x.len());
    let mut y = x.to_vec();
    for i in 0..n {
        let t = i as f32 / ((n - 1) as f32);
        let w = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        let idx = y.len() - n + i;
        y[idx] *= w;
    }
    y
}

/// Apply per-symbol frequency hopping with phase continuity.
///
/// The input slice `x` is interpreted as a frame that starts at sample 0 and consists of:
/// - `N_sym` symbols of `(SF data chips + j_ell guard chips)` each
/// - then `n_tail_syms * SF` tail chips (optional)
///
/// This function multiplies `x[n]` by `exp(j * sgn * Phi[n])`, where `Phi` is the cumulative
/// phase defined by the per-symbol hop offsets in `f_seq_hz`. Use `sgn=+1` for TX hopping and
/// `sgn=-1` for RX de-hopping.
pub(super) fn apply_hop_in_place(
    x: &mut [Complex32],
    hop: HopParams,
    f_seq_hz: &[f64],
    j_seq_chips: &[usize],
    sgn: f32,
) {
    if x.is_empty() || f_seq_hz.is_empty() || j_seq_chips.is_empty() {
        return;
    }
    let n_sym = f_seq_hz.len().min(j_seq_chips.len());
    if n_sym == 0 {
        return;
    }

    let fs = hop.fs_hz as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut ph = Complex32::new(1.0, 0.0);
    let mut idx = 0usize;

    for ell in 0..n_sym {
        let f = f_seq_hz[ell] as f32;
        let dphi = sgn * two_pi * f / fs;
        let w = Complex32::from_polar(1.0, dphi);

        let seg_chips = hop.sf + j_seq_chips[ell];
        let seg_samp = seg_chips.saturating_mul(hop.osf);
        let end = (idx + seg_samp).min(x.len());
        while idx < end {
            x[idx] *= ph;
            ph *= w;
            if (idx & 4095) == 4095 {
                ph /= ph.norm();
            }
            idx += 1;
        }
        if idx >= x.len() {
            return;
        }
    }

    // Tail uses the last hop frequency to preserve phase continuity at shutdown.
    let f_last = f_seq_hz[n_sym - 1] as f32;
    let dphi = sgn * two_pi * f_last / fs;
    let w = Complex32::from_polar(1.0, dphi);
    let tail_samp = hop
        .n_tail_syms
        .saturating_mul(hop.sf)
        .saturating_mul(hop.osf);
    let end = (idx + tail_samp).min(x.len());
    while idx < end {
        x[idx] *= ph;
        ph *= w;
        if (idx & 4095) == 4095 {
            ph /= ph.norm();
        }
        idx += 1;
    }
}

/// Apply per-symbol hopping starting at an absolute `start_sample` inside `x`.
///
/// This is useful for receiver-side de-hopping before matched filtering: the FIR uses samples
/// prior to the frame start, so we rotate the prefix as well (using `f_seq_hz[0]`) to avoid a
/// discontinuity at `start_sample`.
pub(super) fn apply_hop_with_start_in_place(
    x: &mut [Complex32],
    hop: HopParams,
    f_seq_hz: &[f64],
    j_seq_chips: &[usize],
    start_sample: usize,
    sgn: f32,
) {
    if x.is_empty() || f_seq_hz.is_empty() || j_seq_chips.is_empty() {
        return;
    }
    let n_sym = f_seq_hz.len().min(j_seq_chips.len());
    if n_sym == 0 {
        return;
    }

    let fs = hop.fs_hz as f32;
    let two_pi = 2.0 * std::f32::consts::PI;

    // Prefix rotation uses the first hop frequency.
    let f0 = f_seq_hz[0] as f32;
    let w0 = Complex32::from_polar(1.0, sgn * two_pi * f0 / fs);
    let mut ph = Complex32::new(1.0, 0.0);

    let mut idx = 0usize;
    let pre_end = start_sample.min(x.len());
    while idx < pre_end {
        x[idx] *= ph;
        ph *= w0;
        if (idx & 4095) == 4095 {
            ph /= ph.norm();
        }
        idx += 1;
    }
    if idx >= x.len() {
        return;
    }

    // Frame timeline starts at `start_sample` using the per-symbol schedule.
    for ell in 0..n_sym {
        let f = f_seq_hz[ell] as f32;
        let w = Complex32::from_polar(1.0, sgn * two_pi * f / fs);

        let seg_chips = hop.sf + j_seq_chips[ell];
        let seg_samp = seg_chips.saturating_mul(hop.osf);
        let end = (idx + seg_samp).min(x.len());
        while idx < end {
            x[idx] *= ph;
            ph *= w;
            if (idx & 4095) == 4095 {
                ph /= ph.norm();
            }
            idx += 1;
        }
        if idx >= x.len() {
            return;
        }
    }

    // Tail uses the last hop frequency to preserve phase continuity at shutdown.
    let f_last = f_seq_hz[n_sym - 1] as f32;
    let w_last = Complex32::from_polar(1.0, sgn * two_pi * f_last / fs);
    let tail_samp = hop
        .n_tail_syms
        .saturating_mul(hop.sf)
        .saturating_mul(hop.osf);
    let end = (idx + tail_samp).min(x.len());
    while idx < end {
        x[idx] *= ph;
        ph *= w_last;
        if (idx & 4095) == 4095 {
            ph /= ph.norm();
        }
        idx += 1;
    }
}

/// Multiply `x[n]` by `exp(-j*2*pi*cfo*n/fs)` in-place.
pub(super) fn derotate_cfo_in_place(x: &mut [Complex32], fs_hz: u32, cfo_hz: f64) {
    if cfo_hz == 0.0 {
        return;
    }
    let fs = fs_hz as f32;
    let phase_step = -2.0 * std::f32::consts::PI * (cfo_hz as f32) / fs;
    let w = Complex32::from_polar(1.0, phase_step);
    let mut ph = Complex32::new(1.0, 0.0);
    for (n, v) in x.iter_mut().enumerate() {
        *v *= ph;
        ph *= w;
        // Keep `ph` on the unit circle to avoid long-run magnitude drift.
        if (n & 4095) == 4095 {
            ph /= ph.norm();
        }
    }
}
