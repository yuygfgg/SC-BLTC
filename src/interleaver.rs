//! Full-frame bit interleaver (Specification §3.A0).
//!
//! The polar codeword `B` (512 bits) is permuted with a fixed affine map:
//! ```text
//! b_int[j] = b[(A*j + B) mod 512]
//! ```
//! For `M=512=2^9`, choosing `A` odd guarantees the mapping is bijective.

/// Number of coded bits per frame (`M` in the spec).
pub const FRAME_BITS: usize = 512;

// Spec constants: A and B in pi(j) = (A*j + B) mod M.
const A: usize = 109;
const B: usize = 37;

/// Build the inverse permutation table for `deinterleave_frame_llr`.
const fn inv_map() -> [usize; FRAME_BITS] {
    let mut inv = [0usize; FRAME_BITS];
    let mut j = 0usize;
    while j < FRAME_BITS {
        let i = (A * j + B) & (FRAME_BITS - 1);
        inv[i] = j;
        j += 1;
    }
    inv
}

const INV: [usize; FRAME_BITS] = inv_map();

/// Apply the interleaver permutation to a 512-bit codeword.
pub fn interleave_frame_bits(bits: &[u8; FRAME_BITS]) -> [u8; FRAME_BITS] {
    let mut out = [0u8; FRAME_BITS];
    for (j, out_j) in out.iter_mut().enumerate() {
        let i = (A * j + B) & (FRAME_BITS - 1);
        *out_j = bits[i] & 1;
    }
    out
}

/// Apply the inverse permutation to deinterleave soft LLRs back to the polar decoder order.
pub fn deinterleave_frame_llr(llr_int: &[f64; FRAME_BITS]) -> [f64; FRAME_BITS] {
    let mut out = [0f64; FRAME_BITS];
    for (i, out_i) in out.iter_mut().enumerate() {
        *out_i = llr_int[INV[i]];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaver_is_bijective() {
        let mut bits = [0u8; FRAME_BITS];
        for i in 0..FRAME_BITS {
            bits[i] = (i & 1) as u8;
        }
        let int = interleave_frame_bits(&bits);

        let mut llr_int = [0f64; FRAME_BITS];
        for i in 0..FRAME_BITS {
            llr_int[i] = if int[i] == 0 { 1.0 } else { -1.0 };
        }
        let llr = deinterleave_frame_llr(&llr_int);
        for i in 0..FRAME_BITS {
            let b = if llr[i] < 0.0 { 1 } else { 0 };
            assert_eq!(b, bits[i] as i32);
        }
    }
}
