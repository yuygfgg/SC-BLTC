#[derive(Debug, Clone)]
pub struct Ring<T: Copy> {
    buf: Vec<T>,
    cap: u64,
    abs_base: u64,
    abs_next: u64,
}

impl<T: Copy + Default> Ring<T> {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1) as u64;
        Self {
            buf: vec![T::default(); cap as usize],
            cap,
            abs_base: 0,
            abs_next: 0,
        }
    }

    pub fn len(&self) -> u64 {
        self.abs_next - self.abs_base
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn abs_base(&self) -> u64 {
        self.abs_base
    }

    pub fn abs_next(&self) -> u64 {
        self.abs_next
    }

    pub fn push_slice(&mut self, xs: &[T]) {
        for &x in xs {
            let idx = (self.abs_next % self.cap) as usize;
            self.buf[idx] = x;
            self.abs_next += 1;
            if self.abs_next - self.abs_base > self.cap {
                self.abs_base = self.abs_next - self.cap;
            }
        }
    }

    /// Copy a contiguous window `[start_abs, start_abs + len)` into `out`, reusing its allocation.
    pub fn copy_into(&self, start_abs: u64, len: usize, out: &mut Vec<T>) -> Option<()> {
        let len_u = len as u64;
        if start_abs < self.abs_base {
            return None;
        }
        if start_abs + len_u > self.abs_next {
            return None;
        }

        out.clear();
        if out.capacity() < len {
            out.reserve(len - out.capacity());
        }

        let cap = self.cap as usize;
        let start = (start_abs % self.cap) as usize;
        let first = std::cmp::min(len, cap - start);
        out.extend_from_slice(&self.buf[start..start + first]);
        if first < len {
            out.extend_from_slice(&self.buf[..(len - first)]);
        }
        Some(())
    }

    /// Get up to two slices that represent the requested logical window without copying.
    ///
    /// When the window does not wrap, the second slice is empty.
    pub fn get_slices(&self, start_abs: u64, len: usize) -> Option<(&[T], &[T])> {
        let len_u = len as u64;
        if start_abs < self.abs_base {
            return None;
        }
        if start_abs + len_u > self.abs_next {
            return None;
        }

        let cap = self.cap as usize;
        let start = (start_abs % self.cap) as usize;
        let first = std::cmp::min(len, cap - start);
        let a = &self.buf[start..start + first];
        let b = if first < len {
            &self.buf[..(len - first)]
        } else {
            &self.buf[..0]
        };
        Some((a, b))
    }

    pub fn get_vec(&self, start_abs: u64, len: usize) -> Option<Vec<T>> {
        let mut out = Vec::with_capacity(len);
        self.copy_into(start_abs, len, &mut out)?;
        Some(out)
    }
}
