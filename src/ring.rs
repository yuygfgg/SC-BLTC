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

    pub fn get_vec(&self, start_abs: u64, len: usize) -> Option<Vec<T>> {
        let len_u = len as u64;
        if start_abs < self.abs_base {
            return None;
        }
        if start_abs + len_u > self.abs_next {
            return None;
        }
        let mut out = Vec::with_capacity(len);
        for i in 0..len_u {
            let idx = ((start_abs + i) % self.cap) as usize;
            out.push(self.buf[idx]);
        }
        Some(out)
    }
}
