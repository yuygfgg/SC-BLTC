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

/// Spec §2 (GenCode).
///
/// Generate a chip sequence of length `length` (each chip is `+1` or `-1`).
///
/// - `time_index` is `TI = floor(t / IV_res)` (the spec uses 1ms resolution)
/// - `domain_u32` is domain separation to avoid keystream reuse across applications
pub fn gen_code_aes_ctr(
    key: &[u8; 32],
    time_index: u64,
    length: usize,
    domain_u32: u32,
) -> Vec<i8> {
    // Spec §2: Nonce = TI(u64 BE) || Domain(u32 BE) || BlockCounter(u32 BE).
    let mut iv = [0u8; 16];
    iv[..8].copy_from_slice(&time_index.to_be_bytes());
    iv[8..12].copy_from_slice(&domain_u32.to_be_bytes());

    // Spec §2: matches BlockCounter(u32 BE) increment for our lengths.
    type AesCtr = ctr::Ctr128BE<Aes256>;
    let mut cipher = AesCtr::new(key.into(), &iv.into());

    let n_bytes = length.div_ceil(8);
    let mut ks = vec![0u8; n_bytes];
    cipher.apply_keystream(&mut ks);

    let mut chips = Vec::with_capacity(length);
    for &b in &ks {
        for bit in (0..8).rev() {
            if chips.len() >= length {
                break;
            }
            let v = (b >> bit) & 1;
            chips.push(if v == 0 { 1 } else { -1 });
        }
        if chips.len() >= length {
            break;
        }
    }
    chips
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
        let got = gen_code_aes_ctr(&key, 12345, 64, 0x424C_5443);
        let exp: [i8; 64] = [
            -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1,
            1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1,
            1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1,
        ];
        assert_eq!(&got[..], &exp[..]);
    }
}
