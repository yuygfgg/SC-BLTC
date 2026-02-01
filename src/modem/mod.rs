//! The end-to-end SC-BLTC modem (TX + acquisition + demod/decoding).
//!
//! This module is the main integration point for the protocol's DSP pipeline.
//!
//! Public entry points:
//! - TX: [`ScBltcModem::build_frame_samples`]
//! - RX acquisition: [`ScBltcModem::acquire_fft_raw_window`], [`ScBltcModem::acquire_fft_matched_window`]
//! - RX demod/decoding: [`ScBltcModem::demod_decode_raw`], [`ScBltcModem::demod_decode_matched`]
//!
//! The heavy lifting lives in submodules:
//! - `tx`: frame construction + spreading + pulse shaping (Specification §3)
//! - `acq`: blind time/CFO acquisition via FFT (Specification §4.B)
//! - `demod`: RAKE + PLL/DLL tracking + Walsh/FEC decoding (Specification §4.C-§4.D)

use crate::params::Params;
use crate::rrc::{rrc_taps, Fir};
use num_complex::Complex32;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::sync::Arc;

mod acq;
mod demod;
mod tx;
mod util;

/// A fully built transmit frame at complex baseband.
#[derive(Clone, Debug)]
pub struct TxFrame {
    /// `TI_tx = floor(t_tx / IV_res)` used to seed the spreading code (Specification §3.B).
    pub ti_tx: u64,
    /// Complex baseband samples at `F_s`, including optional tail padding and ramp-down.
    pub samples: Vec<Complex32>,
}

/// Result of blind acquisition (time index + offset + CFO + multipath fingers).
#[derive(Clone, Debug)]
pub struct AcqResult {
    /// Estimated transmit `TimeIndex` seed (`TI_tx`).
    pub ti_hat: u64,
    /// Sample offset (>=0) from the start of the acquired IV epoch `ti_hat`.
    pub n0: usize,
    /// Estimated CFO in Hz (coarse+refined).
    pub cfo_hat_hz: f64,
    /// RAKE finger start offsets (>=0) in the same coordinate system as `n0`.
    pub finger_offsets: Vec<usize>,
    /// Verification statistic (larger is better).
    pub p_max: f32,
}

/// Metadata returned by the decoder (header fields + CRC status).
#[derive(Clone, Debug)]
pub struct DecodeMeta {
    /// Whether CRC32C passes after decoding.
    pub crc_ok: bool,
    /// Header.Ver
    pub ver: u8,
    /// Header.Type
    pub typ: u8,
    /// Header.Len
    pub len: u8,
    /// Optional error tag for early exits or malformed inputs.
    pub err: Option<&'static str>,
}

impl DecodeMeta {
    /// Construct an error meta with `crc_ok=false` and no header fields.
    pub fn error(err: &'static str) -> Self {
        Self {
            crc_ok: false,
            ver: 0,
            typ: 0,
            len: 0,
            err: Some(err),
        }
    }
}

/// Modem instance holding parameters, key material, and cached DSP primitives.
pub struct ScBltcModem {
    /// Protocol/PHY parameters.
    pub p: Params,
    /// Shared 256-bit key.
    pub key: [u8; 32],
    /// RRC taps used for TX shaping and RX matched filtering.
    pub rrc: Fir,
    fft_acq: Arc<dyn Fft<f32>>,
    fft_acq_scratch_len: usize,
}

impl ScBltcModem {
    /// Create a modem instance from parameters and a shared key.
    ///
    /// This validates basic parameter consistency and precomputes:
    /// - RRC taps
    /// - FFT plan/scratch size for acquisition
    pub fn new(p: Params, key: [u8; 32]) -> anyhow::Result<Self> {
        if p.fs_hz != p.rc_chip_sps * p.osf {
            anyhow::bail!("Params inconsistent: fs != rc*osf");
        }
        let taps = rrc_taps(p.rrc_alpha, p.osf, p.rrc_span_symbols)?;
        let mut planner = FftPlanner::<f32>::new();
        let fft_acq = planner.plan_fft_forward(p.nfft_acq);
        let fft_acq_scratch_len = fft_acq.get_inplace_scratch_len();
        Ok(Self {
            p,
            key,
            rrc: Fir { taps },
            fft_acq,
            fft_acq_scratch_len,
        })
    }
}

#[cfg(test)]
mod tests;
