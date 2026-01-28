use anyhow::Context;

#[derive(Clone, Debug, serde::Deserialize)]
/// Spec §1.
pub struct Params {
    pub fs_hz: u32,
    pub rc_chip_sps: u32,
    pub osf: u32,
    pub sf: usize,
    pub k_bits_per_sym: usize,
    pub mw: usize,

    pub n_pre: usize,
    pub n_data: usize,
    pub n_pilot: usize,
    pub n_sym: usize,

    pub fec_n: usize,
    pub fec_k: usize,

    pub iv_res_s: f64,
    pub domain_u32: u32,

    pub n_tail: usize,
    pub rrc_alpha: f64,
    pub rrc_span_symbols: u32,
    pub tx_ramp_ms: f64,

    pub nfft_acq: usize,
    pub cfo_search_hz: f64,
    pub gamma_hybrid_mult: f64,
    pub rake_search_half_s: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            fs_hz: 25_000,
            rc_chip_sps: 5_000,
            osf: 5,
            sf: 1024,
            k_bits_per_sym: 8,
            mw: 256,
            n_pre: 2,
            n_data: 64,
            n_pilot: 16,
            n_sym: 82,
            fec_n: 512,
            fec_k: 256,
            iv_res_s: 0.001,
            domain_u32: 0x424C_5443,
            n_tail: 8,
            rrc_alpha: 0.25,
            rrc_span_symbols: 6,
            tx_ramp_ms: 20.0,
            nfft_acq: 32768,
            cfo_search_hz: 8000.0,
            gamma_hybrid_mult: 10.0,
            rake_search_half_s: 0.004,
        }
    }
}

impl Params {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("read params file {path}"))?;
        let params: Params = toml::from_str(&content).context("parse params toml")?;
        Ok(params)
    }

    pub fn tc_s(&self) -> f64 {
        1.0 / (self.rc_chip_sps as f64)
    }

    pub fn chip_samples(&self) -> usize {
        self.sf * (self.osf as usize)
    }

    pub fn frame_chips(&self) -> usize {
        self.n_sym * self.sf
    }

    pub fn frame_samples(&self) -> usize {
        self.frame_chips() * (self.osf as usize)
    }

    pub fn frame_samples_with_tail(&self) -> usize {
        (self.n_sym + self.n_tail) * self.sf * (self.osf as usize)
    }
}
