use super::util::{apply_ramp_down, bits_to_symbols, is_pilot, pulse_shape_chips};
use super::{ScBltcModem, TxFrame};
use crate::crypto::gen_code_aes_ctr;
use crate::frame::build_u_bits;
use crate::ldpc::ldpc_encode_u256;
use crate::walsh::walsh_row;
use num_complex::Complex32;

impl ScBltcModem {
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
        let b_bits = ldpc_encode_u256(&u_bits);
        let m = bits_to_symbols(&b_bits, p.k_bits_per_sym);
        assert_eq!(m.len(), p.n_data);

        let t = t_tx.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        });
        let ti_tx = (t / p.iv_res_s).floor() as u64;

        let c_seq = gen_code_aes_ctr(&self.key, ti_tx, p.frame_chips(), p.domain_u32);

        let mut s = vec![0i8; p.frame_chips()];
        let mut q = 0usize;
        for ell in 0..p.n_sym {
            let seg0 = ell * p.sf;
            if ell == 0 || is_pilot(ell) {
                s[seg0..seg0 + p.sf].copy_from_slice(&c_seq[seg0..seg0 + p.sf]);
            } else {
                let row = walsh_row(m[q] as u16, p.sf);
                for j in 0..p.sf {
                    s[seg0 + j] = row[j] * c_seq[seg0 + j];
                }
                q += 1;
            }
        }
        if q != p.n_data {
            anyhow::bail!("internal mapping error: data symbol count mismatch");
        }

        if p.n_tail > 0 {
            s.extend(std::iter::repeat(0i8).take(p.n_tail * p.sf));
        }

        let x = pulse_shape_chips(&s, &self.rrc, p.osf as usize);

        let ramp_n = ((p.tx_ramp_ms * 1e-3) * (p.fs_hz as f64)).round() as usize;
        let x = apply_ramp_down(&x, ramp_n);

        Ok(TxFrame { ti_tx, samples: x })
    }

    pub fn make_ref_preamble_matched(&self, ti_search: u64) -> Vec<Complex32> {
        // Spec §4.B.2 (local reference).
        let p = &self.p;
        let c = gen_code_aes_ctr(&self.key, ti_search, p.sf, p.domain_u32);
        let x = pulse_shape_chips(&c, &self.rrc, p.osf as usize);
        let y = self.rrc.filter_same(&x);
        y[..p.chip_samples()].to_vec()
    }

    pub fn make_ref_preamble_tx_shaped(&self, ti_search: u64) -> Vec<Complex32> {
        let p = &self.p;
        let c = gen_code_aes_ctr(&self.key, ti_search, p.sf, p.domain_u32);
        let x = pulse_shape_chips(&c, &self.rrc, p.osf as usize);
        x[..p.chip_samples()].to_vec()
    }
}
