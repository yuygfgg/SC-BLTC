use num_complex::Complex32;

#[derive(Debug)]
pub struct RingBuffer {
    buf: Vec<Complex32>,
    cap: usize,
    write_pos: usize,
    total_written: u64,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![Complex32::new(0.0, 0.0); capacity],
            cap: capacity,
            write_pos: 0,
            total_written: 0,
        }
    }

    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn write(&mut self, mut x: &[Complex32]) {
        if x.len() >= self.cap {
            x = &x[x.len() - self.cap..];
        }
        let n = x.len();
        let end = self.write_pos + n;
        if end <= self.cap {
            self.buf[self.write_pos..end].copy_from_slice(x);
        } else {
            let k = self.cap - self.write_pos;
            self.buf[self.write_pos..].copy_from_slice(&x[..k]);
            self.buf[..(end % self.cap)].copy_from_slice(&x[k..]);
        }
        self.write_pos = end % self.cap;
        self.total_written += n as u64;
    }

    /// Returns [abs_start, abs_end) currently kept.
    pub fn available_range(&self) -> (u64, u64) {
        let abs_end = self.total_written();
        let abs_start = abs_end.saturating_sub(self.cap as u64);
        (abs_start, abs_end)
    }

    pub fn read(&self, abs_start: u64, length: usize) -> anyhow::Result<Vec<Complex32>> {
        let abs_end = abs_start + length as u64;
        let (keep_start, keep_end) = self.available_range();
        if abs_start < keep_start || abs_end > keep_end {
            anyhow::bail!("requested range not in buffer");
        }
        let start_idx = (abs_start as usize) % self.cap;
        let end_idx = (abs_end as usize) % self.cap;
        if start_idx < end_idx {
            Ok(self.buf[start_idx..end_idx].to_vec())
        } else {
            let mut out = Vec::with_capacity(length);
            out.extend_from_slice(&self.buf[start_idx..]);
            out.extend_from_slice(&self.buf[..end_idx]);
            debug_assert_eq!(out.len(), length);
            Ok(out)
        }
    }
}
