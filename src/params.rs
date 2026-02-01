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

/// Spec §1.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Sampling rate `F_s` (Hz).
    fs_hz: u32,
    /// Chip rate `R_c` (chips/s).
    rc_chip_sps: u32,
    /// Oversampling factor `OSF = F_s / R_c` (integer).
    osf: u32,
    /// Spreading factor `SF` (chips per spread symbol, fixed to 1024 by the spec).
    sf: usize,
    /// Bits per Walsh symbol `k` (fixed to 8 by the spec).
    k_bits_per_sym: usize,
    /// Walsh orthogonal set size `M_W` (fixed to 256 by the spec).
    mw: usize,

    /// Number of preamble symbols `N_pre` (fixed to 2; Barker-2: `+W0, -W0`).
    n_pre: usize,
    /// Number of data symbols `N_data = M/k` (fixed to 64).
    n_data: usize,
    /// Number of pilot symbols `N_pilot` (fixed to 16).
    n_pilot: usize,
    /// Total spread symbols per frame `N_sym = N_pre + N_pilot + N_data` (fixed to 82).
    n_sym: usize,

    /// Polar code length `N` (coded bits, fixed to 512).
    fec_n: usize,
    /// Polar code information bits `K` (uncoded bits, fixed to 256).
    fec_k: usize,

    /// `IV_res`: time resolution used for `TimeIndex` (seconds, fixed to 1ms).
    iv_res_s: f64,
    /// Domain-separation constant inside the AES-CTR nonce (fixed to `0x424C_5443`, ASCII "BLTC").
    domain_u32: u32,

    /// Tail zero-symbols appended after the frame (lets the RRC filter decay; Specification §3.E2).
    n_tail: usize,
    /// RRC roll-off `alpha`.
    rrc_alpha: f64,
    /// RRC span (in symbols, must be a positive even integer).
    rrc_span_symbols: u32,
    /// TX ramp-down duration (ms, Specification §3.E2).
    tx_ramp_ms: f64,

    /// Acquisition FFT size `N_FFT` (Specification §4.B.2).
    nfft_acq: usize,
    /// Acquisition CFO search half-bandwidth `f_search` (Hz) (Specification §4.B.1).
    cfo_search_hz: f64,
    /// Threshold scale for the "coherent preamble + noncoherent pilots" verification statistic.
    gamma_hybrid_mult: f64,
    /// RAKE finger search half window (seconds, Specification §4.B.3 / §4.C.1).
    rake_search_half_s: f64,

    /// Frequency-hopping bandwidth `BW_hop` (Hz).
    ///
    /// The hop range is centered at baseband 0 Hz, spanning `±BW_hop/2`.
    hop_bw_hz: f64,

    /// Jitter/guard interval minimum length `L_jitter,min` in chips.
    jitter_min_chips: usize,
    /// Jitter/guard interval range `L_jitter,span` in chips.
    ///
    /// Total jitter is mapped to `[jitter_min_chips .. jitter_min_chips + jitter_span_chips]`.
    jitter_span_chips: usize,

    /// Guard/noise filler power relative to the data chip power (dB).
    ///
    /// A value of `-3 dB` means the noise has half the power of the data chips.
    gi_noise_db: f64,
    /// Optional soft ramp length applied at the start/end of the guard noise (chips).
    ///
    /// This is a practical implementation knob to avoid hard edges in the GI while still
    /// preserving the "noise fill" property.
    gi_soft_ramp_chips: usize,
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

            hop_bw_hz: 8000.0,
            jitter_min_chips: 40,
            jitter_span_chips: 50,
            gi_noise_db: -3.0,
            gi_soft_ramp_chips: 8,
        }
    }
}

impl Params {
    pub fn fs_hz(&self) -> u32 {
        self.fs_hz
    }

    pub fn rc_chip_sps(&self) -> u32 {
        self.rc_chip_sps
    }

    pub fn osf(&self) -> u32 {
        self.osf
    }

    pub fn sf(&self) -> usize {
        self.sf
    }

    pub fn k_bits_per_sym(&self) -> usize {
        self.k_bits_per_sym
    }

    pub fn mw(&self) -> usize {
        self.mw
    }

    pub fn n_pre(&self) -> usize {
        self.n_pre
    }

    pub fn n_data(&self) -> usize {
        self.n_data
    }

    pub fn n_pilot(&self) -> usize {
        self.n_pilot
    }

    pub fn n_sym(&self) -> usize {
        self.n_sym
    }

    pub fn fec_n(&self) -> usize {
        self.fec_n
    }

    pub fn fec_k(&self) -> usize {
        self.fec_k
    }

    pub fn iv_res_s(&self) -> f64 {
        self.iv_res_s
    }

    pub fn domain_u32(&self) -> u32 {
        self.domain_u32
    }

    pub fn n_tail(&self) -> usize {
        self.n_tail
    }

    pub fn rrc_alpha(&self) -> f64 {
        self.rrc_alpha
    }

    pub fn rrc_span_symbols(&self) -> u32 {
        self.rrc_span_symbols
    }

    pub fn tx_ramp_ms(&self) -> f64 {
        self.tx_ramp_ms
    }

    pub fn nfft_acq(&self) -> usize {
        self.nfft_acq
    }

    pub fn cfo_search_hz(&self) -> f64 {
        self.cfo_search_hz
    }

    pub fn gamma_hybrid_mult(&self) -> f64 {
        self.gamma_hybrid_mult
    }

    pub fn rake_search_half_s(&self) -> f64 {
        self.rake_search_half_s
    }

    /// Hop bandwidth `BW_hop` (Hz).
    pub fn hop_bw_hz(&self) -> f64 {
        self.hop_bw_hz
    }

    /// Jitter/guard minimum length in chips.
    pub fn jitter_min_chips(&self) -> usize {
        self.jitter_min_chips
    }

    /// Jitter/guard span (additional chips beyond min).
    pub fn jitter_span_chips(&self) -> usize {
        self.jitter_span_chips
    }

    /// Jitter/guard maximum length in chips.
    pub fn jitter_max_chips(&self) -> usize {
        self.jitter_min_chips + self.jitter_span_chips
    }

    /// Guard/noise level relative to the data chips (dB).
    pub fn gi_noise_db(&self) -> f64 {
        self.gi_noise_db
    }

    /// Guard/noise power relative to the data chips (linear).
    pub fn gi_noise_power_lin(&self) -> f64 {
        10f64.powf(self.gi_noise_db / 10.0)
    }

    /// Guard/noise stddev relative to unit-power chips.
    pub fn gi_noise_std(&self) -> f64 {
        self.gi_noise_power_lin().max(0.0).sqrt()
    }

    /// Soft ramp length used inside guard intervals (chips).
    pub fn gi_soft_ramp_chips(&self) -> usize {
        self.gi_soft_ramp_chips
    }

    /// Samples per spread symbol: `SF * OSF`.
    pub fn chip_samples(&self) -> usize {
        self.sf * (self.osf as usize)
    }

    /// Upper bound on the total transmitted samples for a frame, assuming max jitter.
    pub fn frame_max_samples_with_jitter(&self) -> usize {
        let per_sym = self.sf + self.jitter_max_chips();
        let chips = self.n_sym * per_sym + self.n_tail * self.sf;
        chips * (self.osf as usize)
    }
}
