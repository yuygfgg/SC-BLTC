//! Transmit-side frame construction (Specification §3).
//!
//! This module turns `(payload, ver, typ, t_tx)` into complex baseband samples:
//! 1) build `U` (header + payload + CRC + padding)
//! 2) polar encode + interleave
//! 3) map bits -> 256-ary Walsh symbol indices
//! 4) apply cryptographic chip mask and Walsh spreading to form chips
//! 5) RRC pulse shape + optional tail padding + ramp-down

use super::util::{apply_ramp_down, bits_to_symbols, is_pilot, pulse_shape_chips};
use super::{ScBltcModem, TxFrame};
use crate::crypto::gen_code_aes_ctr;
use crate::frame::build_u_bits;
use crate::interleaver::interleave_frame_bits;
use crate::polar::polar_encode_u256;
use crate::walsh::walsh_sign;
use anyhow::Context;
use num_complex::Complex32;

impl ScBltcModem {
    /// Build a full SC-BLTC frame as complex baseband samples (Specification §3).
    ///
    /// - `payload` is limited to 26 bytes by the fixed `K=256` information bit budget.
    /// - `t_tx` overrides the transmit time (seconds since Unix epoch) and is mainly used by tests.
    pub fn build_frame_samples(
        &self,
        payload: &[u8],
        ver: u8,
        typ: u8,
        t_tx: Option<f64>,
    ) -> anyhow::Result<TxFrame> {
        // Spec §3.A0–§3.E2.
        let p = &self.p;

        let u_bits_vec = build_u_bits(payload, ver, typ)?;
        let mut u_bits = [0u8; 256];
        u_bits.copy_from_slice(&u_bits_vec);
        let b_bits = polar_encode_u256(&u_bits);
        let b_bits = interleave_frame_bits(&b_bits);
        let m = bits_to_symbols(&b_bits, p.k_bits_per_sym());
        assert_eq!(m.len(), p.n_data());

        let t = match t_tx {
            Some(t) => t,
            None => std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .context("system clock is before UNIX_EPOCH")?
                .as_secs_f64(),
        };
        let ti_tx = (t / p.iv_res_s()).floor() as u64;

        let c_seq = gen_code_aes_ctr(&self.key, ti_tx, p.frame_chips(), p.domain_u32());

        let mut s = vec![0i8; p.frame_chips()];
        let mut q = 0usize;
        for ell in 0..p.n_sym() {
            let seg0 = ell * p.sf();
            if ell == 0 {
                // Spec §3.D
                s[seg0..seg0 + p.sf()].copy_from_slice(&c_seq[seg0..seg0 + p.sf()]);
            } else if ell == 1 {
                // Spec §3.D
                for j in 0..p.sf() {
                    s[seg0 + j] = -c_seq[seg0 + j];
                }
            } else if is_pilot(ell) {
                s[seg0..seg0 + p.sf()].copy_from_slice(&c_seq[seg0..seg0 + p.sf()]);
            } else {
                let m_q = m[q] as u16;
                for j in 0..p.sf() {
                    let w = walsh_sign(m_q, j);
                    s[seg0 + j] = w * c_seq[seg0 + j];
                }
                q += 1;
            }
        }
        if q != p.n_data() {
            anyhow::bail!("internal mapping error: data symbol count mismatch");
        }

        if p.n_tail() > 0 {
            s.extend(std::iter::repeat_n(0i8, p.n_tail() * p.sf()));
        }

        let x = pulse_shape_chips(&s, &self.rrc, p.osf() as usize);

        let ramp_n = ((p.tx_ramp_ms() * 1e-3) * (p.fs_hz() as f64)).round() as usize;
        let x = apply_ramp_down(&x, ramp_n);

        Ok(TxFrame { ti_tx, samples: x })
    }

    /// Build the local matched-filtered preamble reference used in acquisition (Specification §4.B.2).
    ///
    /// This returns:
    /// `rrc(rx_shape(rrc(tx_shape(chips))))` cropped to the 2-symbol preamble duration.
    pub fn make_ref_preamble_matched(&self, ti_search: u64) -> Vec<Complex32> {
        // Spec §4.B.2 (local reference).
        let p = &self.p;
        let n_chips = p.n_pre() * p.sf();
        let mut c = gen_code_aes_ctr(&self.key, ti_search, n_chips, p.domain_u32());
        // Spec §4.B.2: two-symbol Barker-2 preamble is [+C_seq, -C_seq] over the first 2*SF chips.
        for c_j in c.iter_mut().take(n_chips).skip(p.sf()) {
            *c_j = -*c_j;
        }
        let x = pulse_shape_chips(&c, &self.rrc, p.osf() as usize);
        let y = self.rrc.filter_same(&x);
        y[..(p.n_pre() * p.chip_samples())].to_vec()
    }

    /// Build the local TX-shaped (but not matched-filtered) preamble reference (Specification §4.B.2).
    pub fn make_ref_preamble_tx_shaped(&self, ti_search: u64) -> Vec<Complex32> {
        let p = &self.p;
        let n_chips = p.n_pre() * p.sf();
        let mut c = gen_code_aes_ctr(&self.key, ti_search, n_chips, p.domain_u32());
        for c_j in c.iter_mut().take(n_chips).skip(p.sf()) {
            *c_j = -*c_j;
        }
        let x = pulse_shape_chips(&c, &self.rrc, p.osf() as usize);
        x[..(p.n_pre() * p.chip_samples())].to_vec()
    }
}
