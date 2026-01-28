use num_complex::Complex32;

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

#[derive(Clone, Debug)]
pub struct Fir {
    pub taps: Vec<f32>,
}

impl Fir {
    pub fn delay(&self) -> usize {
        (self.taps.len() - 1) / 2
    }

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

    pub fn state(&self) -> FirState {
        FirState {
            taps: self.taps.clone(),
            z: vec![Complex32::new(0.0, 0.0); self.taps.len().saturating_sub(1)],
        }
    }
}

#[derive(Clone, Debug)]
pub struct FirState {
    taps: Vec<f32>,
    z: Vec<Complex32>,
}

impl FirState {
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
