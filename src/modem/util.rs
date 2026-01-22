use crate::rrc::Fir;
use num_complex::Complex32;

pub(super) fn is_pilot(ell: usize) -> bool {
    // Spec §3 / §4: pilots start after the 2-symbol preamble.
    ell >= 2 && ((ell - 2) % 5 == 0)
}

pub(super) fn wrap_pm_pi(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

pub(super) fn pll_step(
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
    *theta = wrap_pm_pi(*theta + *omega + (kp * scale) * err);
}

pub(super) fn build_carrier_rot_chips(theta: f64, omega: f64, sf: usize) -> Vec<Complex32> {
    let mut rot = Vec::with_capacity(sf);
    let dphi = -(omega as f32) / (sf as f32);
    let mut ph = -(theta as f32);
    for _ in 0..sf {
        rot.push(Complex32::from_polar(1.0, ph));
        ph += dphi;
    }
    rot
}

pub(super) fn bits_to_symbols(bits: &[u8], k: usize) -> Vec<u16> {
    assert!(bits.len() % k == 0);
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
