//! SC-BLTC (Stream Cipher Blind Time Long Code).
//!
//! This crate implements the protocol described in `Specification.pdf`:
//! - TX: frame construction + cryptographic spreading + Walsh modulation
//! - RX: blind time acquisition + tracking demodulation + FEC decoding
//!
//! ```text
//! Tx (Specification §3)
//! payload(bytes)
//!   -> U[256]  = Header(16) || Payload || CRC32C(32) || pad(0)
//!   -> Polar encode (N=512,K=256)
//!   -> interleave (A*j+B mod 512)
//!   -> 64 symbols, each 8 bits -> m[q] in [0,255]
//!   -> [pre0, pre1, (pilot + data*4)*16]  (82 spread symbols total)
//!   -> chips S[ell,j] = C_seq[ell,j] * W_m[j]   (preamble/pilot use W0)
//!   -> RRC pulse shaping + OSF upsampling -> complex baseband samples
//! ```
//!
//! ```text
//! Rx (Specification §4)
//! raw samples
//!   -> blind acquisition (TI search x intra-epoch offset x CFO via FFT)
//!   -> coarse results: (ti_hat, n0, cfo_hat, finger_offsets)
//!   -> matched filter + per-symbol PLL/DLL + RAKE/MRC combining
//!   -> Walsh matched filtering (FHT1024) -> soft LLR
//!   -> deinterleave -> CA-SCL polar decode -> parse Header/Payload/CRC
//! ```

pub mod crypto;
pub mod frame;
pub mod interleaver;
pub mod modem;
pub mod params;
pub mod polar;
pub mod ring;
pub mod rrc;
pub mod sim;
pub mod tracking;
pub mod walsh;
