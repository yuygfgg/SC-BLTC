use anyhow::Context;
use num_complex::Complex32;

#[derive(Clone)]
pub struct Rng64 {
    st: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self {
            st: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.st;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.st = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        ((u as f32) + 1.0) / ((1u32 << 24) as f32 + 2.0)
    }
}

pub struct Gauss {
    have: bool,
    spare: f32,
}

impl Default for Gauss {
    fn default() -> Self {
        Self::new()
    }
}

impl Gauss {
    pub fn new() -> Self {
        Self {
            have: false,
            spare: 0.0,
        }
    }

    pub fn next(&mut self, rng: &mut Rng64) -> f32 {
        if self.have {
            self.have = false;
            return self.spare;
        }
        let u1 = rng.next_f32().max(1e-12);
        let u2 = rng.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * th.cos();
        let z1 = r * th.sin();
        self.have = true;
        self.spare = z1;
        z0
    }
}

#[derive(Clone)]
pub struct MultipathFir {
    taps: Vec<Complex32>,
    dl: Vec<Complex32>,
    pos: usize,
}

impl MultipathFir {
    pub fn new(taps: Vec<Complex32>) -> anyhow::Result<Self> {
        if taps.is_empty() {
            anyhow::bail!("multipath taps must be non-empty");
        }
        Ok(Self {
            dl: vec![Complex32::new(0.0, 0.0); taps.len()],
            taps,
            pos: 0,
        })
    }

    pub fn filter(&mut self, x: Complex32) -> Complex32 {
        let l = self.taps.len();
        self.dl[self.pos] = x;
        let mut y = Complex32::new(0.0, 0.0);
        for k in 0..l {
            let idx = (self.pos + l - k) % l;
            y += self.taps[k] * self.dl[idx];
        }
        self.pos += 1;
        if self.pos >= l {
            self.pos = 0;
        }
        y
    }
}

pub struct DopplerOu {
    f_hz: f64,
    a: f64,
    b: f64,
    max_abs_hz: f64,
}

impl DopplerOu {
    pub fn new(std_hz: f64, tau_s: f64, fs_hz: f64, max_abs_hz: f64) -> Self {
        let a = if tau_s > 0.0 {
            (-1.0 / (fs_hz * tau_s)).exp()
        } else {
            0.0
        };
        let b = std_hz * (1.0 - a * a).max(0.0).sqrt();
        Self {
            f_hz: 0.0,
            a,
            b,
            max_abs_hz: max_abs_hz.max(0.0),
        }
    }

    pub fn set_f_hz(&mut self, f_hz: f64) {
        self.f_hz = f_hz;
        self.clamp_f_hz();
    }

    pub fn step_hz(&mut self, rng: &mut Rng64, gauss: &mut Gauss) -> f64 {
        let w = gauss.next(rng) as f64;
        self.f_hz = self.a * self.f_hz + self.b * w;
        self.clamp_f_hz();
        self.f_hz
    }

    fn clamp_f_hz(&mut self) {
        if self.max_abs_hz > 0.0 {
            self.f_hz = self.f_hz.clamp(-self.max_abs_hz, self.max_abs_hz);
        }
    }
}

pub struct ChannelState {
    phi: f32,
    mp: Option<MultipathFir>,
    doppler: Option<DopplerOu>,
}

impl ChannelState {
    pub fn new(mp: Option<MultipathFir>, doppler: Option<DopplerOu>) -> Self {
        Self {
            phi: 0.0,
            mp,
            doppler,
        }
    }

    pub fn has_multipath(&self) -> bool {
        self.mp.is_some()
    }
}

pub fn normalize_taps(mut taps: Vec<Complex32>) -> Vec<Complex32> {
    let e: f32 = taps.iter().map(|c| c.norm_sqr()).sum();
    if e > 0.0 {
        let s = 1.0 / e.sqrt();
        for t in &mut taps {
            *t *= s;
        }
    }
    taps
}

pub fn parse_mp_tap(s: &str, rng: &mut Rng64) -> anyhow::Result<(usize, Complex32)> {
    let ss = s.trim().replace(':', ",");
    let parts: Vec<&str> = ss
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .collect();
    if parts.len() < 2 || parts.len() > 3 {
        anyhow::bail!("mp_tap must be `delay_samp,gain_db[,phase_deg]`, got `{s}`");
    }
    let delay: usize = parts[0].parse().context("parse delay_samp")?;
    let gain_db: f32 = parts[1].parse().context("parse gain_db")?;
    let phase_rad: f32 = if parts.len() == 3 {
        let deg: f32 = parts[2].parse().context("parse phase_deg")?;
        deg * (std::f32::consts::PI / 180.0)
    } else {
        2.0 * std::f32::consts::PI * rng.next_f32()
    };
    let gain_lin = 10.0f32.powf(gain_db / 20.0);
    Ok((delay, Complex32::from_polar(gain_lin, phase_rad)))
}

pub fn build_random_multipath(
    paths: usize,
    max_delay: usize,
    decay_db_per_samp: f32,
    rng: &mut Rng64,
    gauss: &mut Gauss,
) -> Vec<Complex32> {
    if paths <= 1 || max_delay == 0 {
        return vec![Complex32::new(1.0, 0.0)];
    }
    let mut entries: Vec<(usize, Complex32)> = vec![(0, Complex32::new(1.0, 0.0))];
    for _ in 1..paths {
        let d = 1 + (rng.next_u64() as usize) % max_delay;
        let pwr = 10.0f32.powf(-decay_db_per_samp * (d as f32) / 10.0);
        let sigma = (pwr / 2.0).sqrt();
        let gr = sigma * gauss.next(rng);
        let gi = sigma * gauss.next(rng);
        entries.push((d, Complex32::new(gr, gi)));
    }
    let max_d = entries.iter().map(|(d, _)| *d).max().unwrap_or(0);
    let mut taps = vec![Complex32::new(0.0, 0.0); max_d + 1];
    for (d, g) in entries {
        taps[d] += g;
    }
    normalize_taps(taps)
}

#[allow(clippy::too_many_arguments)]
pub fn apply_channel_sample(
    x: Complex32,
    amp: f32,
    cfo_hz: f64,
    fs_actual_hz: f64,
    noise_std: f32,
    st: &mut ChannelState,
    rng: &mut Rng64,
    gauss: &mut Gauss,
) -> Complex32 {
    let mut y = if let Some(mp) = &mut st.mp {
        mp.filter(x)
    } else {
        x
    };

    let mut f_hz = cfo_hz;
    if let Some(d) = &mut st.doppler {
        f_hz += d.step_hz(rng, gauss);
    }
    if f_hz != 0.0 {
        y *= Complex32::from_polar(1.0, st.phi);
        let dphi = (2.0 * std::f64::consts::PI * f_hz / fs_actual_hz) as f32;
        st.phi += dphi;
        if st.phi.abs() > 1000.0 {
            st.phi = st.phi.rem_euclid(2.0 * std::f32::consts::PI);
        }
    }

    y *= amp;

    if noise_std > 0.0 {
        let nr = noise_std * gauss.next(rng);
        let ni = noise_std * gauss.next(rng);
        y += Complex32::new(nr, ni);
    }
    y
}
