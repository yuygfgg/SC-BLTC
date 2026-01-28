use crate::rrc::Fir;
use num_complex::Complex32;

pub(super) fn is_pilot(ell: usize) -> bool {
    // Spec §3 / §4: pilots start after the 2-symbol preamble.
    ell >= 2 && (ell - 2).is_multiple_of(5)
}

pub(super) fn wrap_pm_pi(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

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

pub(super) fn pulse_shape_chips(chips: &[i8], fir: &Fir, osf: usize) -> Vec<Complex32> {
    let mut up = vec![Complex32::new(0.0, 0.0); chips.len() * osf];
    for (i, &c) in chips.iter().enumerate() {
        up[i * osf] = Complex32::new(c as f32, 0.0);
    }
    fir.filter_same(&up)
}

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

pub(super) fn derotate_cfo_in_place(x: &mut [Complex32], fs_hz: u32, cfo_hz: f64) {
    if cfo_hz == 0.0 {
        return;
    }
    let fs = fs_hz as f32;
    for (n, v) in x.iter_mut().enumerate() {
        let ph = -2.0 * std::f32::consts::PI * (cfo_hz as f32) * (n as f32) / fs;
        *v *= Complex32::from_polar(1.0, ph);
    }
}
