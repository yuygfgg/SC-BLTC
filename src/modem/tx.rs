//! Transmit-side frame construction (Specification §3).
//!
//! This module turns `(payload, ver, typ, t_tx)` into complex baseband samples:
//! 1) build `U` (header + payload + CRC + padding)
//! 2) polar encode + interleave
//! 3) map bits -> 256-ary Walsh symbol indices
//! 4) apply cryptographic chip mask and Walsh spreading to form chips
//! 5) insert per-symbol guard/noise
//! 6) RRC pulse shape (before hopping)
//! 7) apply phase-continuous frequency hopping
//! 8) optional tail padding + ramp-down

use super::util::{
    apply_hop_in_place, apply_ramp_down, bits_to_symbols, is_pilot, pulse_shape_real_chips,
    HopParams,
};
use super::{ScBltcModem, TxFrame};
use crate::crypto::{gen_code_structured_aes_ctr, JitterSpec};
use crate::frame::build_u_bits;
use crate::interleaver::interleave_frame_bits;
use crate::polar::polar_encode_u256;
use crate::walsh::walsh_sign;
use anyhow::Context;

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
        // Spec §3.A0–§3.E2 + per-symbol guard/noise + hop mixing.
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

        let gen = gen_code_structured_aes_ctr(
            &self.key,
            ti_tx,
            p.n_sym(),
            p.sf(),
            p.domain_u32(),
            p.hop_bw_hz(),
            JitterSpec {
                min_chips: p.jitter_min_chips(),
                span_chips: p.jitter_span_chips(),
            },
        );

        let c_seq = gen.c_seq;
        let f_seq = gen.f_seq_hz;
        let j_seq = gen.j_seq_chips;

        // Build the chip-rate real baseband, including (data || guard-noise) segments.
        let mut s: Vec<f32> = Vec::new();
        let mut q = 0usize;
        let n_sym = p.n_sym();
        debug_assert_eq!(j_seq.len(), n_sym);
        for (ell, &j_ell) in j_seq.iter().enumerate().take(n_sym) {
            let seg0 = ell * p.sf();
            if ell == 0 {
                // Spec §3.D
                for j in 0..p.sf() {
                    s.push(c_seq[seg0 + j] as f32);
                }
            } else if ell == 1 {
                // Spec §3.D
                for j in 0..p.sf() {
                    s.push(-(c_seq[seg0 + j] as f32));
                }
            } else if is_pilot(ell) {
                for j in 0..p.sf() {
                    s.push(c_seq[seg0 + j] as f32);
                }
            } else {
                let m_q = m[q] as u16;
                for j in 0..p.sf() {
                    let w = walsh_sign(m_q, j);
                    s.push((w as f32) * (c_seq[seg0 + j] as f32));
                }
                q += 1;
            }

            // Spec §3.D1.5: append guard/noise filler of length j_ell chips.
            if j_ell > 0 {
                // Deterministic per-frame/per-symbol noise for reproducible tests.
                // This is not security-critical; it is only used to remove sharp symbol boundaries.
                let mut rng = NoiseRng::new(ti_tx ^ ((ell as u64) << 32) ^ 0xA7F0_31D3_59B2_9C11);
                let noise_std = p.gi_noise_std() as f32;
                let ramp = p.gi_soft_ramp_chips().min(j_ell / 2);
                for k in 0..j_ell {
                    let mut v = rng.next_gauss() * noise_std;
                    if ramp >= 2 {
                        if k < ramp {
                            v *= half_cosine_ramp(k, ramp);
                        } else if k >= j_ell - ramp {
                            v *= half_cosine_ramp(j_ell - 1 - k, ramp);
                        }
                    }
                    s.push(v);
                }
            }
        }
        if q != p.n_data() {
            anyhow::bail!("internal mapping error: data symbol count mismatch");
        }

        // Spec §3.E2: tail padding (zero chips) after the frame.
        if p.n_tail() > 0 {
            s.extend(std::iter::repeat_n(0.0f32, p.n_tail() * p.sf()));
        }

        // Spec: RRC pulse shaping is performed before hopping/mixing.
        let mut x = pulse_shape_real_chips(&s, &self.rrc, p.osf() as usize);

        // Spec §3.D2: apply phase-continuous hopping.
        apply_hop_in_place(
            &mut x,
            HopParams {
                fs_hz: p.fs_hz(),
                sf: p.sf(),
                osf: p.osf() as usize,
                n_tail_syms: p.n_tail(),
            },
            &f_seq,
            &j_seq,
            1.0,
        );

        let ramp_n = ((p.tx_ramp_ms() * 1e-3) * (p.fs_hz() as f64)).round() as usize;
        let x = apply_ramp_down(&x, ramp_n);

        Ok(TxFrame { ti_tx, samples: x })
    }
}

/// Tiny deterministic RNG used for guard/noise filling (not cryptographically secure).
struct NoiseRng {
    st: u64,
    have: bool,
    spare: f32,
}

impl NoiseRng {
    fn new(seed: u64) -> Self {
        Self {
            st: seed ^ 0x9E37_79B9_7F4A_7C15,
            have: false,
            spare: 0.0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.st;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.st = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        ((u as f32) + 1.0) / ((1u32 << 24) as f32 + 2.0)
    }

    fn next_gauss(&mut self) -> f32 {
        if self.have {
            self.have = false;
            return self.spare;
        }
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * th.cos();
        let z1 = r * th.sin();
        self.have = true;
        self.spare = z1;
        z0
    }
}

fn half_cosine_ramp(i: usize, n: usize) -> f32 {
    // i in [0 .. n-1], output in [0 .. 1].
    if n <= 1 {
        return 1.0;
    }
    let t = (i as f32) / ((n - 1) as f32);
    0.5 * (1.0 - (std::f32::consts::PI * t).cos())
}
