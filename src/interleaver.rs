pub const FRAME_BITS: usize = 512;

// Full-frame bit interleaver permutation:
//   b_int[j] = b[(A*j + B) mod FRAME_BITS]
// with A odd so the mapping is bijective for FRAME_BITS=2^m.
const A: usize = 109;
const B: usize = 37;

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

pub fn interleave_frame_bits(bits: &[u8; FRAME_BITS]) -> [u8; FRAME_BITS] {
    let mut out = [0u8; FRAME_BITS];
    for j in 0..FRAME_BITS {
        let i = (A * j + B) & (FRAME_BITS - 1);
        out[j] = bits[i] & 1;
    }
    out
}

pub fn deinterleave_frame_llr(llr_int: &[f64; FRAME_BITS]) -> [f64; FRAME_BITS] {
    let mut out = [0f64; FRAME_BITS];
    for i in 0..FRAME_BITS {
        out[i] = llr_int[INV[i]];
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
