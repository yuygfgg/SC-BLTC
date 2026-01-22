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

#[derive(Clone, Debug)]
pub struct TxFrame {
    pub ti_tx: u64,
    pub samples: Vec<Complex32>,
}

#[derive(Clone, Debug)]
pub struct AcqResult {
    pub ti_hat: u64,
    /// Sample offset (>=0) from the start of the acquired IV epoch `ti_hat`.
    pub n0: usize,
    pub cfo_hat_hz: f64,
    /// RAKE finger start offsets (>=0) in the same coordinate system as `n0`.
    pub finger_offsets: Vec<usize>,
    pub p_max: f32,
}

#[derive(Clone, Debug)]
pub struct DecodeMeta {
    pub crc_ok: bool,
    pub ver: u8,
    pub typ: u8,
    pub len: u8,
    pub err: Option<&'static str>,
}

impl DecodeMeta {
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

pub struct ScBltcModem {
    pub p: Params,
    pub key: [u8; 32],
    pub rrc: Fir,
    fft_acq: Arc<dyn Fft<f32>>,
    fft_acq_scratch_len: usize,
}

impl ScBltcModem {
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
