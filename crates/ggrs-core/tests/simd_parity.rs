use ggrs_core::simd;

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    // детерминированный LCG, без зависимостей
    let mut s = seed as u64;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }).collect()
}

#[test]
fn scalar_vs_avx2_parity() {
    if !simd::have_avx2() {
        eprintln!("AVX2 недоступен — тест пропущен");
        return;
    }
    for &n in &[1usize, 7, 8, 15, 16, 33, 1024, 1000] {
        let a = pseudo(n, 1);
        let b = pseudo(n, 2);

        let ds = {
            let mut d = vec![0.0; n];
            simd::scalar::vec_add(&a, &b, &mut d);
            d
        };
        let dv = {
            let mut d = vec![0.0; n];
            unsafe { simd::avx2::vec_add(&a, &b, &mut d) };
            d
        };
        assert_eq!(ds, dv, "vec_add n={n}"); // сложение поэлементное — бит-в-бит

        let dot_s = simd::scalar::vec_dot(&a, &b);
        let dot_v = unsafe { simd::avx2::vec_dot(&a, &b) };
        let tol = 1e-5 * (n as f32).sqrt().max(1.0);
        assert!((dot_s - dot_v).abs() <= tol * dot_s.abs().max(1.0), "vec_dot n={n}: {dot_s} vs {dot_v}");
    }
}
