mod ldpc_code {
    include!("ldpc_code.rs");
}

pub const LDPC_N: usize = ldpc_code::N;
pub const LDPC_M_CHECK: usize = ldpc_code::M_CHECK;
pub const LDPC_K_EFF: usize = ldpc_code::K_EFF;
pub const LDPC_PAD_BITS: usize = ldpc_code::PAD_BITS;

fn word_parity(x: u64) -> u8 {
    (x.count_ones() as u8) & 1
}

pub fn ldpc_encode_u256(u_bits: &[u8; 256]) -> [u8; 512] {
    // K_eff=259 (rank-deficient); last 3 message bits are zero.
    debug_assert_eq!(LDPC_K_EFF, 259);
    debug_assert_eq!(LDPC_PAD_BITS, 3);
    debug_assert_eq!(ldpc_code::PARITY_BITS, 253);

    let mut out = [0u8; 512];

    out[..256].copy_from_slice(u_bits);
    out[256] = 0;
    out[257] = 0;
    out[258] = 0;

    let mut msg_words = [0u64; ldpc_code::PARITY_ROW_WORDS];
    for i in 0..LDPC_K_EFF {
        if out[i] & 1 != 0 {
            msg_words[i / 64] |= 1u64 << (i % 64);
        }
    }

    for (i, row) in ldpc_code::PARITY_ROWS.iter().enumerate() {
        let mut p = 0u8;
        for w in 0..ldpc_code::PARITY_ROW_WORDS {
            p ^= word_parity(row[w] & msg_words[w]);
        }
        out[LDPC_K_EFF + i] = p & 1;
    }
    out
}

fn hard_decision(llr: &[f32; 512]) -> [u8; 512] {
    let mut x = [0u8; 512];
    for i in 0..512 {
        x[i] = if llr[i] < 0.0 { 1 } else { 0 };
    }
    x
}

fn syndrome_ok(x: &[u8; 512]) -> bool {
    for c in 0..LDPC_M_CHECK {
        let mut acc = 0u8;
        for i in 0..ldpc_code::CN_DEG {
            let v = ldpc_code::CN_TO_VN[c][i] as usize;
            acc ^= x[v] & 1;
        }
        if acc != 0 {
            return false;
        }
    }
    true
}

/// Spec §3.A0 (LDPC).
pub fn ldpc_decode_to_u256_from_llr(llr_in: &[f64; 512], maxiter: usize) -> [u8; 256] {
    let alpha: f32 = 0.8;
    let msg_clip: f32 = 20.0;

    let mut llr = [0f32; 512];
    for i in 0..512 {
        llr[i] = (llr_in[i] as f32).clamp(-msg_clip, msg_clip);
    }
    llr[256] = msg_clip;
    llr[257] = msg_clip;
    llr[258] = msg_clip;

    let mut v_to_c = [[0f32; ldpc_code::VN_DEG]; 512];
    let mut c_to_v = [[0f32; ldpc_code::CN_DEG]; LDPC_M_CHECK];

    for v in 0..512 {
        for e in 0..ldpc_code::VN_DEG {
            v_to_c[v][e] = llr[v];
        }
    }

    for _it in 0..maxiter {
        for c in 0..LDPC_M_CHECK {
            let mut signs_xor: u8 = 0;
            let mut min1 = f32::INFINITY;
            let mut min2 = f32::INFINITY;
            let mut min1_i = 0usize;

            let mut inc = [0f32; ldpc_code::CN_DEG];
            for i in 0..ldpc_code::CN_DEG {
                let v = ldpc_code::CN_TO_VN[c][i] as usize;
                let slot = ldpc_code::CN_TO_VN_SLOT[c][i] as usize;
                let m = v_to_c[v][slot].clamp(-msg_clip, msg_clip);
                inc[i] = m;
                if m < 0.0 {
                    signs_xor ^= 1;
                }
                let a = m.abs();
                if a < min1 {
                    min2 = min1;
                    min1 = a;
                    min1_i = i;
                } else if a < min2 {
                    min2 = a;
                }
            }

            for i in 0..ldpc_code::CN_DEG {
                let m_i = inc[i];
                let a = if i == min1_i { min2 } else { min1 };
                let sign_excl = signs_xor ^ if m_i < 0.0 { 1 } else { 0 };
                let s = if sign_excl != 0 { -1.0 } else { 1.0 };
                c_to_v[c][i] = (alpha * s * a).clamp(-msg_clip, msg_clip);
            }
        }

        for v in 0..512 {
            let mut sum_all = llr[v];
            for e in 0..ldpc_code::VN_DEG {
                let c = ldpc_code::VN_TO_CN[v][e] as usize;
                let pos = ldpc_code::VN_TO_CN_POS[v][e] as usize;
                sum_all += c_to_v[c][pos];
            }
            for e in 0..ldpc_code::VN_DEG {
                let c = ldpc_code::VN_TO_CN[v][e] as usize;
                let pos = ldpc_code::VN_TO_CN_POS[v][e] as usize;
                v_to_c[v][e] = (sum_all - c_to_v[c][pos]).clamp(-msg_clip, msg_clip);
            }
        }

        let mut post = [0f32; 512];
        for v in 0..512 {
            let mut s = llr[v];
            for e in 0..ldpc_code::VN_DEG {
                let c = ldpc_code::VN_TO_CN[v][e] as usize;
                let pos = ldpc_code::VN_TO_CN_POS[v][e] as usize;
                s += c_to_v[c][pos];
            }
            post[v] = s;
        }
        let x = hard_decision(&post);
        if syndrome_ok(&x) {
            let mut u = [0u8; 256];
            u.copy_from_slice(&x[..256]);
            return u;
        }
    }

    let mut post = [0f32; 512];
    for v in 0..512 {
        let mut s = llr[v];
        for e in 0..ldpc_code::VN_DEG {
            let c = ldpc_code::VN_TO_CN[v][e] as usize;
            let pos = ldpc_code::VN_TO_CN_POS[v][e] as usize;
            s += c_to_v[c][pos];
        }
        post[v] = s;
    }
    let x = hard_decision(&post);
    let mut u = [0u8; 256];
    u.copy_from_slice(&x[..256]);
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame;

    #[test]
    fn adjacency_maps_consistent() {
        for v in 0..512usize {
            for e in 0..ldpc_code::VN_DEG {
                let c = ldpc_code::VN_TO_CN[v][e] as usize;
                let pos = ldpc_code::VN_TO_CN_POS[v][e] as usize;
                assert_eq!(ldpc_code::CN_TO_VN[c][pos] as usize, v);
            }
        }
        for c in 0..LDPC_M_CHECK {
            for i in 0..ldpc_code::CN_DEG {
                let v = ldpc_code::CN_TO_VN[c][i] as usize;
                let slot = ldpc_code::CN_TO_VN_SLOT[c][i] as usize;
                assert_eq!(ldpc_code::VN_TO_CN[v][slot] as usize, c);
            }
        }
    }

    #[test]
    fn encode_systematic_prefix() {
        let mut u = [0u8; 256];
        u[0] = 1;
        u[10] = 1;
        let b = ldpc_encode_u256(&u);
        assert_eq!(&b[..256], &u);
        assert!(b[256..].iter().all(|&x| x <= 1));
    }

    #[test]
    fn encode_then_decode_high_confidence() {
        let mut u = [0u8; 256];
        for i in 0..256 {
            u[i] = ((i * 7 + 3) & 1) as u8;
        }
        let b = ldpc_encode_u256(&u);
        assert!(syndrome_ok(&b));
        let mut llr = [0f64; 512];
        for i in 0..512 {
            llr[i] = if b[i] == 0 { 10.0 } else { -10.0 };
        }
        let u_hat = ldpc_decode_to_u256_from_llr(&llr, 30);
        assert_eq!(u_hat, u);
        let bytes = frame::build_u_bits(b"hi", 1, 1).unwrap();
        assert_eq!(bytes.len(), 256);
    }
}
