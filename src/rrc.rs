//! Root-raised-cosine (RRC) pulse shaping and a small FIR helper.
//!
//! The transmitter shapes the chip sequence with an RRC filter and the receiver uses the same taps
//! as a matched filter (Specification §3.D2 / §4.B.2 / §4.C).
//!
//! The generated tap sequence is energy-normalized, so it can be used directly for both TX shaping
//! and RX matched filtering.

use num_complex::Complex32;

/// Generate unit-energy RRC taps.
///
/// - `alpha` is the roll-off factor in `(0, 1]`
/// - `sps` is samples per symbol (in this project, symbol == chip, so `sps = OSF`)
/// - `span_symbols` is the filter span in symbols (must be a positive even integer)
pub fn rrc_taps(alpha: f64, sps: u32, span_symbols: u32) -> anyhow::Result<Vec<f32>> {
    if !(0.0 < alpha && alpha <= 1.0) {
        anyhow::bail!("alpha must be in (0,1]");
    }
    if span_symbols == 0 || !span_symbols.is_multiple_of(2) {
        anyhow::bail!("span_symbols must be a positive even integer");
    }
    if sps == 0 {
        anyhow::bail!("sps must be positive");
    }

    let n_taps = (span_symbols * sps + 1) as usize;
    let center = (n_taps as f64 - 1.0) / 2.0;
    let sps_f = sps as f64;

    let mut h = vec![0f64; n_taps];
    for (i, h_i) in h.iter_mut().enumerate() {
        let ti = (i as f64 - center) / sps_f;
        if ti.abs() < 1e-12 {
            *h_i = 1.0 - alpha + 4.0 * alpha / std::f64::consts::PI;
            continue;
        }
        let sing = (1.0 / (4.0 * alpha)).abs();
        if (ti.abs() - sing).abs() < 1e-9 {
            let a = alpha;
            *h_i = (a / 2.0_f64.sqrt())
                * ((1.0 + 2.0 / std::f64::consts::PI) * (std::f64::consts::PI / (4.0 * a)).sin()
                    + (1.0 - 2.0 / std::f64::consts::PI)
                        * (std::f64::consts::PI / (4.0 * a)).cos());
            continue;
        }

        let a = alpha;
        let num = (std::f64::consts::PI * ti * (1.0 - a)).sin()
            + 4.0 * a * ti * (std::f64::consts::PI * ti * (1.0 + a)).cos();
        let den = std::f64::consts::PI * ti * (1.0 - (4.0 * a * ti).powi(2));
        *h_i = num / den;
    }

    let e: f64 = h.iter().map(|v| v * v).sum();
    let scale = e.sqrt();
    for v in &mut h {
        *v /= scale;
    }

    Ok(h.into_iter().map(|v| v as f32).collect())
}

/// A simple complex-valued FIR filter.
///
/// The tap vector is applied with a "current sample + previous samples" convention:
/// `y[n] = sum_{k=0..L-1} taps[k] * x[n-k]`.
#[derive(Clone, Debug)]
pub struct Fir {
    pub taps: Vec<f32>,
}

impl Fir {
    /// Group delay for symmetric taps (`(L-1)/2`).
    pub fn delay(&self) -> usize {
        (self.taps.len() - 1) / 2
    }

    /// Convolve `x` with the FIR and return an output with the same length as `x`.
    pub fn filter_same(&self, x: &[Complex32]) -> Vec<Complex32> {
        let l = self.taps.len();
        let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
        for n in 0..x.len() {
            let mut acc = Complex32::new(0.0, 0.0);
            let kmax = std::cmp::min(l - 1, n);
            for k in 0..=kmax {
                acc += x[n - k] * self.taps[k];
            }
            y[n] = acc;
        }
        y
    }

    /// Create a stateful version of this FIR for streaming processing.
    pub fn state(&self) -> FirState {
        FirState {
            taps: self.taps.clone(),
            z: vec![Complex32::new(0.0, 0.0); self.taps.len().saturating_sub(1)],
        }
    }
}

/// Stateful FIR filter for block-by-block processing.
#[derive(Clone, Debug)]
pub struct FirState {
    taps: Vec<f32>,
    z: Vec<Complex32>,
}

impl FirState {
    /// Filter a block, preserving the internal delay line across calls.
    pub fn process_block(&mut self, x: &[Complex32]) -> Vec<Complex32> {
        let l = self.taps.len();
        let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
        for (i, &s) in x.iter().enumerate() {
            let mut acc = s * self.taps[0];
            for k in 1..l {
                acc += self.z[k - 1] * self.taps[k];
            }
            y[i] = acc;

            if !self.z.is_empty() {
                let z_len = self.z.len();
                self.z.copy_within(..z_len - 1, 1);
                self.z[0] = s;
            }
        }
        y
    }
}
