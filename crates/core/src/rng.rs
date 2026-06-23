//! Small seeded PRNG (xoshiro256**) used across the crate.
//!
//! Extracted from synthetic_data so other modules (e.g. detection) can
//! reuse the same deterministic RNG implementation.

/// Simple seeded PRNG (xoshiro256**)
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Initialise from a single u64 seed using SplitMix64.
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Raw usize output compatible with the previous local PRNG usage.
    /// Mirrors the old `(state >> 33) as usize` behaviour used by
    /// gromov_hyperbolicity's internal next closure.
    pub fn next_raw(&mut self) -> usize {
        (self.next_u64() >> 33) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_raw(), rng2.next_raw());
        }
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = Rng::new(42);
        for _ in 0..10000 {
            let v = rng.uniform();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
