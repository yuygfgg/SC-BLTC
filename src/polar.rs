use crate::frame;

pub const POLAR_N: usize = 512;
pub const POLAR_K: usize = 256;

const PW_BETA: f64 = 1.189_207_115_002_721; // 2^(1/4)

fn log1pexp(x: f64) -> f64 {
    // Stable log(1 + exp(x)).
    if x.is_finite() {
        if x > 0.0 {
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        }
    } else if x.is_sign_positive() {
        f64::INFINITY
    } else {
        0.0
    }
}

fn pm_update(llr: f64, bit: u8) -> f64 {
    // Negative log-likelihood update in LLR domain
    if (bit & 1) == 0 {
        log1pexp(-llr)
    } else {
        log1pexp(llr)
    }
}

fn f_min_sum(a: f64, b: f64) -> f64 {
    let s = if (a < 0.0) ^ (b < 0.0) { -1.0 } else { 1.0 };
    s * a.abs().min(b.abs())
}

fn g(a: f64, b: f64, u: u8) -> f64 {
    if (u & 1) == 0 {
        b + a
    } else {
        b - a
    }
}

fn polar_pw_order(n: usize) -> Vec<usize> {
    let logn = n.trailing_zeros() as usize;
    assert_eq!(1usize << logn, n, "polar N must be a power of two");

    let mut idx_w: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let mut w = 0.0;
            for j in 0..logn {
                if ((i >> j) & 1) != 0 {
                    w += PW_BETA.powi(j as i32);
                }
            }
            (i, w)
        })
        .collect();

    idx_w.sort_by(|(ia, wa), (ib, wb)| wa.total_cmp(wb).then_with(|| ia.cmp(ib)));
    idx_w.into_iter().map(|(i, _)| i).collect()
}

fn frozen_and_info_positions() -> ([bool; POLAR_N], [usize; POLAR_K]) {
    let order = polar_pw_order(POLAR_N);
    let mut frozen = [true; POLAR_N];
    for &i in &order[POLAR_N - POLAR_K..] {
        frozen[i] = false;
    }

    let mut info: Vec<usize> = frozen
        .iter()
        .enumerate()
        .filter_map(|(i, &is_frozen)| if is_frozen { None } else { Some(i) })
        .collect();
    info.sort_unstable();
    let info: [usize; POLAR_K] = info.try_into().unwrap();

    (frozen, info)
}

fn polar_transform_in_place(x: &mut [u8]) {
    debug_assert_eq!(x.len(), POLAR_N);
    let logn = POLAR_N.trailing_zeros() as usize;
    for stage in 0..logn {
        let step = 1usize << (stage + 1);
        let half = 1usize << stage;
        for i in (0..POLAR_N).step_by(step) {
            for j in 0..half {
                let a = i + j;
                let b = a + half;
                x[a] ^= x[b];
            }
        }
    }
}

/// Spec §3.A0: (N,K)=(512,256)
pub fn polar_encode_u256(u_bits: &[u8; POLAR_K]) -> [u8; POLAR_N] {
    let (_frozen, info_pos) = frozen_and_info_positions();
    let mut u = [0u8; POLAR_N];
    for (j, &pos) in info_pos.iter().enumerate() {
        u[pos] = u_bits[j] & 1;
    }
    polar_transform_in_place(&mut u);
    u
}

#[derive(Clone)]
struct Path {
    metric: f64,
    u: [u8; POLAR_N],
    // Decoder recursion stack: last entry is the LLR vector for the current node.
    llr_stack: Vec<Vec<f64>>,
    // Saved left-child encoded bits (x_left) while decoding the right child.
    left_x_stack: Vec<Vec<u8>>,
    // Encoded bits (x) for the subtree just decoded (returned to the parent).
    x_ret: Vec<u8>,
}

fn decode_node(
    paths: &mut Vec<Path>,
    base: usize,
    size: usize,
    frozen: &[bool; POLAR_N],
    l: usize,
) {
    debug_assert!(size.is_power_of_two());

    if size == 1 {
        let is_frozen = frozen[base];
        let mut next: Vec<Path> = Vec::with_capacity(if is_frozen {
            paths.len()
        } else {
            2 * paths.len()
        });
        for p in paths.drain(..) {
            let llr = p.llr_stack.last().unwrap()[0];
            if is_frozen {
                let mut p0 = p;
                p0.metric += pm_update(llr, 0);
                p0.u[base] = 0;
                p0.x_ret = vec![0];
                next.push(p0);
            } else {
                let mut p0 = p.clone();
                p0.metric += pm_update(llr, 0);
                p0.u[base] = 0;
                p0.x_ret = vec![0];
                next.push(p0);

                let mut p1 = p;
                p1.metric += pm_update(llr, 1);
                p1.u[base] = 1;
                p1.x_ret = vec![1];
                next.push(p1);
            }
        }
        next.sort_by(|a, b| a.metric.total_cmp(&b.metric));
        next.truncate(l.max(1));
        *paths = next;
        return;
    }

    let half = size / 2;

    // Left child: push llr_left for each path.
    for p in paths.iter_mut() {
        let parent = p.llr_stack.last().unwrap();
        debug_assert_eq!(parent.len(), size);
        let mut llr_left = Vec::with_capacity(half);
        for i in 0..half {
            llr_left.push(f_min_sum(parent[i], parent[i + half]));
        }
        p.llr_stack.push(llr_left);
    }
    decode_node(paths, base, half, frozen, l);
    for p in paths.iter_mut() {
        p.llr_stack.pop().unwrap(); // pop llr_left
        p.left_x_stack.push(std::mem::take(&mut p.x_ret));
    }

    // Right child: push llr_right for each path, using the stored x_left.
    for p in paths.iter_mut() {
        let parent = p.llr_stack.last().unwrap();
        debug_assert_eq!(parent.len(), size);
        let x_left = p.left_x_stack.last().unwrap();
        debug_assert_eq!(x_left.len(), half);
        let mut llr_right = Vec::with_capacity(half);
        for i in 0..half {
            llr_right.push(g(parent[i], parent[i + half], x_left[i]));
        }
        p.llr_stack.push(llr_right);
    }
    decode_node(paths, base + half, half, frozen, l);
    for p in paths.iter_mut() {
        p.llr_stack.pop().unwrap(); // pop llr_right
        let x_right = std::mem::take(&mut p.x_ret);
        let x_left = p.left_x_stack.pop().unwrap();
        debug_assert_eq!(x_left.len(), half);
        debug_assert_eq!(x_right.len(), half);

        let mut x = Vec::with_capacity(size);
        for i in 0..half {
            x.push((x_left[i] ^ x_right[i]) & 1);
        }
        x.extend(x_right.into_iter().map(|v| v & 1));
        p.x_ret = x;
    }
}

/// CRC-aided SCL decoder for Spec §4.D: returns the decoded 256-bit `U` payload (header+payload+crc+pad).
pub fn polar_decode_to_u256_from_llr(llr_in: &[f64; POLAR_N], list_size: usize) -> [u8; POLAR_K] {
    let (frozen, info_pos) = frozen_and_info_positions();
    let l = list_size.clamp(1, 64);

    let p0 = Path {
        metric: 0.0,
        u: [0u8; POLAR_N],
        llr_stack: vec![llr_in.to_vec()],
        left_x_stack: Vec::with_capacity(POLAR_N.trailing_zeros() as usize),
        x_ret: Vec::new(),
    };

    // Root decode fills `u` and returns the (reconstructed) codeword in `x_ret`.
    let mut paths = vec![p0];
    decode_node(&mut paths, 0, POLAR_N, &frozen, l);

    // CRC-aided selection on the extracted U-bits.
    let mut best_crc: Option<([u8; POLAR_K], f64)> = None;
    let mut best_any: Option<([u8; POLAR_K], f64)> = None;

    for p in paths {
        let mut u_bits = [0u8; POLAR_K];
        for (j, &pos) in info_pos.iter().enumerate() {
            u_bits[j] = p.u[pos] & 1;
        }

        if best_any
            .as_ref()
            .map(|(_, m)| p.metric < *m)
            .unwrap_or(true)
        {
            best_any = Some((u_bits, p.metric));
        }

        let crc_ok = frame::parse_u_bits(&u_bits)
            .map(|(_, _, ok)| ok)
            .unwrap_or(false);
        if crc_ok
            && best_crc
                .as_ref()
                .map(|(_, m)| p.metric < *m)
                .unwrap_or(true)
        {
            best_crc = Some((u_bits, p.metric));
        }
    }

    best_crc.or(best_any).unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_then_decode_high_confidence() {
        let mut u = [0u8; POLAR_K];
        for i in 0..POLAR_K {
            u[i] = ((i * 7 + 3) & 1) as u8;
        }
        let b = polar_encode_u256(&u);
        let mut llr = [0f64; POLAR_N];
        for i in 0..POLAR_N {
            llr[i] = if b[i] == 0 { 20.0 } else { -20.0 };
        }
        let u_hat = polar_decode_to_u256_from_llr(&llr, 8);
        assert_eq!(u_hat, u);
    }

    #[test]
    fn crc_aided_selection_smoke() -> anyhow::Result<()> {
        let u_vec = crate::frame::build_u_bits(b"hi", 1, 1)?;
        let mut u = [0u8; POLAR_K];
        u.copy_from_slice(&u_vec);

        let b = polar_encode_u256(&u);
        let mut llr = [0f64; POLAR_N];
        for i in 0..POLAR_N {
            llr[i] = if b[i] == 0 { 6.0 } else { -6.0 };
        }
        let u_hat = polar_decode_to_u256_from_llr(&llr, 8);
        let (_hdr, _payload, ok) = crate::frame::parse_u_bits(&u_hat)?;
        assert!(ok);
        Ok(())
    }
}
