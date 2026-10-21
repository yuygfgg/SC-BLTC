//! The end-to-end SC-BLTC modem (TX + acquisition + demod/decoding).
//!
//! This module is the main integration point for the protocol's DSP pipeline.
//!
//! Public entry points:
//! - TX: [`ScBltcModem::build_frame_samples`]
//! - RX acquisition: [`ScBltcModem::acquire_fft_raw_window`]
//! - RX demod/decoding: [`ScBltcModem::demod_decode_raw`], [`ScBltcModem::demod_decode_matched`]
//!
//! The heavy lifting lives in submodules:
//! - `tx`: frame construction + spreading + pulse shaping (Specification §3)
//! - `acq`: blind time/CFO acquisition via FFT (Specification §4.B)
//! - `demod`: RAKE + PLL/DLL tracking + Walsh/FEC decoding (Specification §4.C-§4.D)

use crate::crypto::{gen_code_structured_aes_ctr, JitterSpec};
use crate::params::{Params, MAX_SRO_PPM};
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

/// Result of blind acquisition (time index + arrival sample + CFO + multipath fingers).
#[derive(Clone, Debug)]
pub struct AcqResult {
    /// Estimated transmit `TimeIndex` seed (`TI_tx`).
    pub ti_hat: u64,
    /// Receiver sample index of the selected reference arrival path, relative to the acquisition
    /// window start.
    ///
    /// `ti_hat` selects the spreading-code seed; `arrival_sample` selects the receiver sample
    /// where the reference path arrives.
    pub arrival_sample: usize,
    /// Estimated CFO in Hz (coarse+refined).
    pub cfo_hat_hz: f64,
    /// RAKE finger delays in receiver samples relative to `arrival_sample`.
    ///
    /// The first offset is always zero. Other paths can be earlier or later than the reference.
    pub finger_offsets: Vec<isize>,
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
    /// Verification statistic used for timing refinement fallback (larger is better).
    ///
    /// This is based on known-structure parts of the frame (preamble/pilots).
    pub verify_score: f32,
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
            verify_score: 0.0,
            err: Some(err),
        }
    }
}

/// Modem instance holding parameters, key material, and cached DSP primitives.
pub struct ScBltcModem {
    /// Protocol/PHY parameters.
    pub p: Params,
    /// Shared 256-bit key.
    ///
    /// Kept private to avoid accidental exposure via logs/debug output.
    key: [u8; 32],
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
        if p.fs_hz() != p.rc_chip_sps() * p.osf() {
            anyhow::bail!("Params inconsistent: fs != rc*osf");
        }
        let taps = rrc_taps(p.rrc_alpha(), p.osf(), p.rrc_span_symbols())?;
        let mut planner = FftPlanner::<f32>::new();
        let fft_acq = planner.plan_fft_forward(p.nfft_acq());
        let fft_acq_scratch_len = fft_acq.get_inplace_scratch_len();
        Ok(Self {
            p,
            key,
            rrc: Fir { taps },
            fft_acq,
            fft_acq_scratch_len,
        })
    }

    /// Return the exact sample span of the frame body for a transmit time index.
    ///
    /// The decoder processes the preamble, pilots, and data symbols. The transmit tail is only
    /// filter settling time, so callers that schedule decoding should use this span instead of
    /// the full padded frame length.
    pub fn frame_body_samples_for_ti(&self, ti: u64) -> usize {
        let p = &self.p;
        let g = gen_code_structured_aes_ctr(
            &self.key,
            ti,
            p.n_sym(),
            p.sf(),
            p.domain_u32(),
            p.hop_bw_hz(),
            JitterSpec {
                min_chips: p.jitter_min_chips(),
                span_chips: p.jitter_span_chips(),
            },
        );
        let chips = g
            .j_seq_chips
            .iter()
            .take(p.n_sym())
            .fold(0usize, |total, &j| total.saturating_add(p.sf() + j));
        chips.saturating_mul(p.osf() as usize)
    }

    /// Return the frame-body span with enough room for the maximum positive SRO.
    ///
    /// A positive transmitter sampling-rate offset stretches the received frame. The receiver
    /// must retain the complete stretched body before it starts decoding.
    pub fn frame_body_samples_for_ti_with_max_sro(&self, ti: u64) -> usize {
        let nominal = self.frame_body_samples_for_ti(ti) as f64;
        (nominal * (1.0 + MAX_SRO_PPM * 1e-6)).ceil() as usize
    }
}

#[cfg(test)]
mod tests;
