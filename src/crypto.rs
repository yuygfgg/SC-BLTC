//! Cryptographic spreading sequence generator (Specification §2).
//!
//! The protocol uses AES-CTR as a CSPRNG. For a shared key `K_sec` and a time counter `TI_tx`, it
//! produces a pseudo-random chip mask `C_seq[j]` where each chip is either `+1` or `-1`.
//!
//! Nonce/counter layout (16 bytes):
//! ```text
//! IV = TI(u64, big-endian) || Domain(u32, big-endian) || BlockCounter(u32, big-endian)
//! ```
//!
//! Keystream-bit to chip mapping:
//! - consume keystream bits MSB -> LSB within each byte
//! - bit=0 -> +1, bit=1 -> -1

use aes::Aes256;
use cipher::{KeyIvInit, StreamCipher};

/// Generator output for frame construction and synchronized control words.
#[derive(Clone, Debug)]
pub struct GenCodeOut {
    /// Chip-mask sequence for de-spreading, concatenated over `N_sym` symbols.
    ///
    /// Length is `N_sym * SF`, each entry is `+1` or `-1`.
    pub c_seq: Vec<i8>,
    /// Per-symbol hop offsets in Hz (`f_ell`), length `N_sym`.
    pub f_seq_hz: Vec<f64>,
    /// Per-symbol guard/jitter length in chips (`j_ell`), length `N_sym`.
    pub j_seq_chips: Vec<usize>,
}

/// Parameters for the structured modem generator that logically belong together.
#[derive(Copy, Clone, Debug)]
pub struct JitterSpec {
    /// Minimum guard/jitter length, in chips.
    pub min_chips: usize,
    /// Additional span beyond `min_chips`, in chips.
    pub span_chips: usize,
}

/// Structured generator used by the modem.
///
/// For each symbol `ell=0..N_sym-1`, consume AES-CTR keystream bytes as:
/// - 128 bytes (1024 bits) -> `SF` chip-mask values `(+1/-1)`
/// - 2 bytes (16 bits)     -> metadata: `V_freq` (high 8) and `V_jitter` (low 8)
///
/// The hop offset mapping is:
/// `f_ell = ((V_freq/256.0) - 0.5) * BW_hop` (Hz)
///
/// The jitter mapping is:
/// `j_ell = jitter_min + floor((V_jitter/256.0) * jitter_span)` (chips)
pub fn gen_code_structured_aes_ctr(
    key: &[u8; 32],
    time_index: u64,
    n_sym: usize,
    sf: usize,
    domain_u32: u32,
    bw_hop_hz: f64,
    jitter: JitterSpec,
) -> GenCodeOut {
    // Spec §2: Nonce = TI(u64 BE) || Domain(u32 BE) || BlockCounter(u32 BE).
    let mut iv = [0u8; 16];
    iv[..8].copy_from_slice(&time_index.to_be_bytes());
    iv[8..12].copy_from_slice(&domain_u32.to_be_bytes());

    type AesCtr = ctr::Ctr128BE<Aes256>;
    let mut cipher = AesCtr::new(key.into(), &iv.into());

    // Per symbol: 1024 bits + 16 bits = 1040 bits = 130 bytes.
    let bytes_per_sym = (sf / 8) + 2;
    debug_assert_eq!(sf, 1024, "this implementation currently assumes SF=1024");
    debug_assert_eq!(
        bytes_per_sym, 130,
        "SF=1024 => 128 bytes + 2 bytes metadata"
    );

    let n_bytes = n_sym.saturating_mul(bytes_per_sym);
    let mut ks = vec![0u8; n_bytes];
    cipher.apply_keystream(&mut ks);

    let mut c_seq: Vec<i8> = Vec::with_capacity(n_sym * sf);
    let mut f_seq_hz: Vec<f64> = Vec::with_capacity(n_sym);
    let mut j_seq_chips: Vec<usize> = Vec::with_capacity(n_sym);

    for ell in 0..n_sym {
        let base = ell * bytes_per_sym;
        let bits = &ks[base..base + (sf / 8)];
        for &b in bits {
            for bit in (0..8).rev() {
                let v = (b >> bit) & 1;
                c_seq.push(if v == 0 { 1 } else { -1 });
            }
        }

        let v_freq = ks[base + (sf / 8)];
        let v_jitter = ks[base + (sf / 8) + 1];

        let f = ((v_freq as f64) / 256.0 - 0.5) * bw_hop_hz;
        f_seq_hz.push(f);

        let mut j = jitter.min_chips + ((v_jitter as usize) * jitter.span_chips) / 256;

        if v_jitter == u8::MAX {
            j = jitter.min_chips + jitter.span_chips;
        }
        j_seq_chips.push(j);
    }

    GenCodeOut {
        c_seq,
        f_seq_hz,
        j_seq_chips,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_code_matches_python_vector() {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let got = gen_code_structured_aes_ctr(
            &key,
            12345,
            1,
            1024,
            0x424C_5443,
            8000.0,
            JitterSpec {
                min_chips: 0,
                span_chips: 0,
            },
        )
        .c_seq;
        let exp: [i8; 64] = [
            -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1,
            1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1,
            1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1,
        ];
        assert_eq!(&got[..64], &exp[..]);
    }

    #[test]
    fn gen_code_structured_lengths_are_consistent() {
        let key = [0u8; 32];
        let out = gen_code_structured_aes_ctr(
            &key,
            1,
            82,
            1024,
            0x424C_5443,
            8000.0,
            JitterSpec {
                min_chips: 40,
                span_chips: 50,
            },
        );
        assert_eq!(out.c_seq.len(), 82 * 1024);
        assert_eq!(out.f_seq_hz.len(), 82);
        assert_eq!(out.j_seq_chips.len(), 82);
        for &j in &out.j_seq_chips {
            assert!((40..=90).contains(&j), "j={j}");
        }
        for &f in &out.f_seq_hz {
            assert!((-4000.0 - 1e-9..=4000.0 + 1e-9).contains(&f), "f={f}");
        }
    }
}
