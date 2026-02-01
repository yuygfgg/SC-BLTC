//! Protocol/system parameters (Specification §1).
//!
//! `Params` holds the PHY constants used across the implementation and provides a few derived
//! quantities (chips per frame, samples per symbol, ...).
//!
//! Conventions:
//! - `*_hz`: Hz
//! - `*_s`: seconds
//! - `*_ms`: milliseconds
//! - `*_sps`: rate in "per second" (see the field doc for whether it is samples/s or chips/s)

use anyhow::Context;

/// Spec §1.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Params {
    /// Sampling rate `F_s` (Hz).
    pub fs_hz: u32,
    /// Chip rate `R_c` (chips/s).
    pub rc_chip_sps: u32,
    /// Oversampling factor `OSF = F_s / R_c` (integer).
    pub osf: u32,
    /// Spreading factor `SF` (chips per spread symbol, fixed to 1024 by the spec).
    pub sf: usize,
    /// Bits per Walsh symbol `k` (fixed to 8 by the spec).
    pub k_bits_per_sym: usize,
    /// Walsh orthogonal set size `M_W` (fixed to 256 by the spec).
    pub mw: usize,

    /// Number of preamble symbols `N_pre` (fixed to 2; Barker-2: `+W0, -W0`).
    pub n_pre: usize,
    /// Number of data symbols `N_data = M/k` (fixed to 64).
    pub n_data: usize,
    /// Number of pilot symbols `N_pilot` (fixed to 16).
    pub n_pilot: usize,
    /// Total spread symbols per frame `N_sym = N_pre + N_pilot + N_data` (fixed to 82).
    pub n_sym: usize,

    /// Polar code length `N` (coded bits, fixed to 512).
    pub fec_n: usize,
    /// Polar code information bits `K` (uncoded bits, fixed to 256).
    pub fec_k: usize,

    /// `IV_res`: time resolution used for `TimeIndex` (seconds, fixed to 1ms).
    pub iv_res_s: f64,
    /// Domain-separation constant inside the AES-CTR nonce (fixed to `0x424C_5443`, ASCII "BLTC").
    pub domain_u32: u32,

    /// Tail zero-symbols appended after the frame (lets the RRC filter decay; Specification §3.E2).
    pub n_tail: usize,
    /// RRC roll-off `alpha`.
    pub rrc_alpha: f64,
    /// RRC span (in symbols, must be a positive even integer).
    pub rrc_span_symbols: u32,
    /// TX ramp-down duration (ms, Specification §3.E2).
    pub tx_ramp_ms: f64,

    /// Acquisition FFT size `N_FFT` (Specification §4.B.2).
    pub nfft_acq: usize,
    /// Acquisition CFO search half-bandwidth `f_search` (Hz) (Specification §4.B.1).
    pub cfo_search_hz: f64,
    /// Threshold scale for the "coherent preamble + noncoherent pilots" verification statistic.
    pub gamma_hybrid_mult: f64,
    /// RAKE finger search half window (seconds, Specification §4.B.3 / §4.C.1).
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
    /// Read parameters from a TOML file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("read params file {path}"))?;
        let params: Params = toml::from_str(&content).context("parse params toml")?;
        Ok(params)
    }

    /// Chip duration `T_c = 1/R_c` (seconds).
    pub fn tc_s(&self) -> f64 {
        1.0 / (self.rc_chip_sps as f64)
    }

    /// Samples per spread symbol: `SF * OSF`.
    pub fn chip_samples(&self) -> usize {
        self.sf * (self.osf as usize)
    }

    /// Chips per frame: `N_sym * SF`.
    pub fn frame_chips(&self) -> usize {
        self.n_sym * self.sf
    }

    /// Samples per frame: `frame_chips * OSF`.
    pub fn frame_samples(&self) -> usize {
        self.frame_chips() * (self.osf as usize)
    }

    /// Samples per frame including the tail padding (Specification §3.E2).
    pub fn frame_samples_with_tail(&self) -> usize {
        (self.n_sym + self.n_tail) * self.sf * (self.osf as usize)
    }
}
