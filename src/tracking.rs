//! Tracking-loop helpers (Specification §4.C.2).
//!
//! The receiver runs two coupled control loops at the symbol rate:
//! - a carrier phase/frequency loop (PLL/Costas)
//! - a code timing loop (DLL, early/late gate)
//!
//! This module provides a small discrete-time PI loop designer and a state container for the
//! early/late DLL.

/// Discrete-time PI gains.
#[derive(Clone, Copy, Debug)]
pub struct LoopGains {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
}

/// Design a stable 2nd-order discrete-time PI loop (Specification §4.C.2).
///
/// The returned gains target a given loop bandwidth and damping factor, using pole mapping
/// `z = exp(s*T)` rather than a small-`T` approximation.
pub fn design_2nd_order_loop(loop_bw_hz: f64, damping: f64, update_period_s: f64) -> LoopGains {
    // Discrete-time PI loop design via pole mapping (stable even when ωn·T is not small).
    let bw = loop_bw_hz;
    let zeta = damping;
    let t = update_period_s;
    if bw <= 0.0 || t <= 0.0 || zeta <= 0.0 || !zeta.is_finite() {
        return LoopGains { kp: 0.0, ki: 0.0 };
    }
    let wn = 2.0 * std::f64::consts::PI * bw;
    if !wn.is_finite() {
        return LoopGains { kp: 0.0, ki: 0.0 };
    }

    // Desired discrete poles (sum, product).
    let zeta2 = zeta * zeta;
    let (sum, prod) = if zeta2 < 1.0 {
        // Underdamped: complex-conjugate poles z = r e^{±j ang}.
        let wd = wn * (1.0 - zeta2).sqrt();
        let r = (-zeta * wn * t).exp();
        let ang = wd * t;
        (2.0 * r * ang.cos(), r * r)
    } else {
        // Over/critically damped: real poles.
        let root = (zeta2 - 1.0).sqrt();
        let s1 = -wn * (zeta - root);
        let s2 = -wn * (zeta + root);
        let z1 = (s1 * t).exp();
        let z2 = (s2 * t).exp();
        (z1 + z2, z1 * z2)
    };

    // Closed-loop poly: λ^2 - (2 - kp - ki)λ + (1 - kp) = 0.
    let kp = 1.0 - prod;
    let ki = 1.0 + prod - sum;
    LoopGains { kp, ki }
}

/// Early/late code timing loop state.
#[derive(Clone, Debug)]
pub struct EarlyLateDll {
    /// Current estimate of samples per symbol.
    pub sym_step_samp: f64,
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Clamp for `sym_step_samp` (minimum).
    pub sym_step_min: f64,
    /// Clamp for `sym_step_samp` (maximum).
    pub sym_step_max: f64,
    /// Scale factor applied when the loop is decision-directed.
    pub dd_scale: f64,
}

impl EarlyLateDll {
    /// Spec §4.C.2.
    pub fn update(&mut self, err_samp: f64, dd: bool) -> f64 {
        let scale = if dd { self.dd_scale } else { 1.0 };
        let kps = self.kp * scale;
        let kis = self.ki * scale;

        self.sym_step_samp =
            (self.sym_step_samp - kis * err_samp).clamp(self.sym_step_min, self.sym_step_max);
        -kps * err_samp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn loop_roots(kp: f64, ki: f64) -> (Complex64, Complex64) {
        // λ^2 - (2 - kp - ki) λ + (1 - kp) = 0
        let b = -(2.0 - kp - ki);
        let c = 1.0 - kp;
        let disc = Complex64::new(b * b - 4.0 * c, 0.0).sqrt();
        let r1 = (-Complex64::new(b, 0.0) + disc) * 0.5;
        let r2 = (-Complex64::new(b, 0.0) - disc) * 0.5;
        (r1, r2)
    }

    #[test]
    fn loop_design_is_stable_for_low_symbol_rate_example() {
        let g = design_2nd_order_loop(1.0, 0.707, 0.205);
        assert!(g.kp.is_finite() && g.ki.is_finite());
        let (r1, r2) = loop_roots(g.kp, g.ki);
        assert!(r1.norm() < 1.0 + 1e-12, "unstable root r1={:?}", r1);
        assert!(r2.norm() < 1.0 + 1e-12, "unstable root r2={:?}", r2);
    }
}
