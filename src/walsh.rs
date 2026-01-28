use num_complex::Complex32;

/// Spec §1 (Walsh definition).
pub fn walsh_row(m: u16, n: usize) -> Vec<i8> {
    if (m as usize) >= n {
        panic!("m must be in [0,n)");
    }
    let mut out = vec![0i8; n];
    for (j, out_j) in out.iter_mut().enumerate() {
        let v = (m as u32) & (j as u32);
        let parity = v.count_ones() & 1;
        *out_j = if parity == 0 { 1 } else { -1 };
    }
    out
}

/// Spec §4.D.2.
pub fn fht1024_in_place(y: &mut [Complex32]) {
    assert_eq!(y.len(), 1024);
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
}

pub fn fht1024(x: &[Complex32]) -> Vec<Complex32> {
    let mut y = x.to_vec();
    fht1024_in_place(&mut y);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walsh_row_basic() {
        let w0 = walsh_row(0, 8);
        assert_eq!(w0, vec![1; 8]);
        let w1 = walsh_row(1, 8);
        assert_eq!(w1, vec![1, -1, 1, -1, 1, -1, 1, -1]);
    }

    #[test]
    fn fht_matches_direct_for_small_prefix() {
        let mut x = vec![Complex32::new(0.0, 0.0); 1024];
        for i in 0..8 {
            x[i] = Complex32::new((i + 1) as f32, 0.0);
        }
        let y = fht1024(&x);
        for m in 0..8u16 {
            let w = walsh_row(m, 8);
            let mut s = 0.0f32;
            for j in 0..8 {
                s += (x[j].re) * (w[j] as f32);
            }
            assert!((y[m as usize].re - s).abs() < 1e-4);
        }
    }
}
