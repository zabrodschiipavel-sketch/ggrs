/// Детерминированный LCG для сидов спидрана (та же формула, что в тестах проекта).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    /// Число в [-0.5, 0.5): ((x>>33) as f32 / 2^31) - 0.5 после шага LCG
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_first_two_values_seed7() {
        let mut rng = Lcg::new(7);

        // Шаг 1: X1
        let x1 = 7u64.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let expected1 = ((x1 >> 33) as f32 / (1u64 << 31) as f32) - 0.5;

        // Шаг 2: X2
        let x2 = x1.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let expected2 = ((x2 >> 33) as f32 / (1u64 << 31) as f32) - 0.5;

        let got1 = rng.next_f32();
        let got2 = rng.next_f32();

        assert_eq!(got1, expected1, "первое значение Lcg(7) не совпало");
        assert_eq!(got2, expected2, "второе значение Lcg(7) не совпало");
    }

    #[test]
    fn lcg_deterministic_100_values() {
        let mut rng_a = Lcg::new(42);
        let mut rng_b = Lcg::new(42);

        for _ in 0..100 {
            let a = rng_a.next_f32();
            let b = rng_b.next_f32();
            assert_eq!(a, b, "детерминизм LCG нарушен");
        }
    }
}
