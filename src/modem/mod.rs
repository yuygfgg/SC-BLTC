use crate::params::Params;
use crate::rrc::{rrc_taps, Fir};
use num_complex::Complex32;

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
    pub n0: usize,
    pub cfo_coarse_hz: f64,
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
}

impl ScBltcModem {
    pub fn new(p: Params, key: [u8; 32]) -> anyhow::Result<Self> {
        if p.fs_hz != p.rc_chip_sps * p.osf {
            anyhow::bail!("Params inconsistent: fs != rc*osf");
        }
        let taps = rrc_taps(p.rrc_alpha, p.osf, p.rrc_span_symbols)?;
        Ok(Self {
            p,
            key,
            rrc: Fir { taps },
        })
    }
}

#[cfg(test)]
mod tests;
