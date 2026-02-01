//! Walsh codes and the Fast Hadamard Transform (FHT).
//!
//! SC-BLTC uses rows of the order-`SF` Sylvester-Hadamard matrix as spreading sequences
//! (Specification §1). For a data symbol with Walsh index `m`:
//! ```text
//! W_m[j] = (-1)^(popcount(m & j) mod 2),  j = 0..SF-1
//! ```
//! with `W_0[j] == +1` for all `j`.
//!
//! On the receiver, the Walsh matched filter bank is computed efficiently with a length-1024 FHT
//! (Specification §4.D.2).

use anyhow::bail;
use num_complex::Complex32;

/// Generate the `m`-th Walsh/Hadamard row of length `n` (entries are `+1` or `-1`).
///
/// For orthogonality, `n` must be a power of two; the protocol uses `n=1024`.
pub fn walsh_row(m: u16, n: usize) -> anyhow::Result<Vec<i8>> {
    if n == 0 || !n.is_power_of_two() {
        bail!("n must be a power of two (got n={n})");
    }
    if (m as usize) >= n {
        bail!("m must be in [0,n) (got m={m}, n={n})");
    }
    let mut out = vec![0i8; n];
    for (j, out_j) in out.iter_mut().enumerate() {
        let v = (m as u32) & (j as u32);
        let parity = v.count_ones() & 1;
        *out_j = if parity == 0 { 1 } else { -1 };
    }
    Ok(out)
}

/// In-place FHT for `n=1024` (unnormalized).
///
/// If `x[j]` are chips, then after transform `y[m]` equals `sum_j x[j] * W_m[j]` for the
/// Sylvester-Hadamard matrix ordering used by [`walsh_row`].
pub fn fht1024_in_place(y: &mut [Complex32]) -> anyhow::Result<()> {
    if y.len() != 1024 {
        bail!("fht1024_in_place expects len=1024 (got len={})", y.len());
    }
    let n = 1024;
    let mut h = 1;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in 0..h {
                let a = y[i + j];
                let b = y[i + h + j];
                y[i + j] = a + b;
                y[i + h + j] = a - b;
            }
            i += step;
        }
        h *= 2;
    }
    Ok(())
}

/// Convenience wrapper around [`fht1024_in_place`].
pub fn fht1024(x: &[Complex32]) -> anyhow::Result<Vec<Complex32>> {
    let mut y = x.to_vec();
    fht1024_in_place(&mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walsh_row_basic() {
        let w0 = walsh_row(0, 8).unwrap();
        assert_eq!(w0, vec![1; 8]);
        let w1 = walsh_row(1, 8).unwrap();
        assert_eq!(w1, vec![1, -1, 1, -1, 1, -1, 1, -1]);
    }

    #[test]
    fn fht_matches_direct_for_small_prefix() {
        let mut x = vec![Complex32::new(0.0, 0.0); 1024];
        for i in 0..8 {
            x[i] = Complex32::new((i + 1) as f32, 0.0);
        }
        let y = fht1024(&x).unwrap();
        for m in 0..8u16 {
            let w = walsh_row(m, 8).unwrap();
            let mut s = 0.0f32;
            for j in 0..8 {
                s += (x[j].re) * (w[j] as f32);
            }
            assert!((y[m as usize].re - s).abs() < 1e-4);
        }
    }
}
