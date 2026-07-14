use ggrs_core::*;
use ggrs_core::util::Lcg;

/// Два одинаковых сида дают одинаковые последовательности next_normal.
#[test]
fn normal_deterministic() {
    let mut a = Lcg::new(42);
    let mut b = Lcg::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_normal(0.0, 1.0), b.next_normal(0.0, 1.0));
    }
}

/// Статистика на 10 000 сэмплов N(0, 0.02): среднее и σ в допусках.
#[test]
fn normal_statistics() {
    let mut rng = Lcg::new(7);
    let n = 10_000usize;
    let vals: Vec<f32> = (0..n).map(|_| rng.next_normal(0.0, 0.02)).collect();
    let mean = vals.iter().sum::<f32>() / n as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    assert!(mean.abs() < 0.002, "mean = {mean}");
    assert!((std - 0.02).abs() < 0.001, "std = {std}");
    // все значения конечны
    assert!(vals.iter().all(|v| v.is_finite()));
}

/// fill_normal заливает весь тензор, детерминированно по сиду.
#[test]
fn fill_normal_tensor() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 8, 4);
    let mut rng = Lcg::new(3);
    ctx.fill_normal(a, 0.0, 0.02, &mut rng);
    let d = ctx.data_f32(a);
    assert_eq!(d.len(), 32);
    // не все нули и всё конечно
    assert!(d.iter().any(|&v| v != 0.0));
    assert!(d.iter().all(|v| v.is_finite()));
    // повторная заливка с тем же сидом — бит-в-бит
    let mut ctx2 = Context::new(1 << 20);
    let b = ctx2.new_tensor_2d(DType::F32, 8, 4);
    let mut rng2 = Lcg::new(3);
    ctx2.fill_normal(b, 0.0, 0.02, &mut rng2);
    assert_eq!(ctx.data_f32(a), ctx2.data_f32(b));
}

/// fill_uniform: значения в [lo, hi).
#[test]
fn fill_uniform_bounds() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_1d(DType::F32, 1000);
    let mut rng = Lcg::new(11);
    ctx.fill_uniform(a, -0.5, 0.5, &mut rng);
    for &v in ctx.data_f32(a) {
        assert!((-0.5..0.5).contains(&v), "v = {v}");
    }
}

/// mem_used растёт с аллокациями и учитывает 32-байтное выравнивание.
#[test]
fn mem_used_grows_aligned() {
    let mut ctx = Context::new(1 << 20);
    assert_eq!(ctx.mem_used(), 0);
    let _a = ctx.new_tensor_1d(DType::F32, 1); // 4 байта
    let used1 = ctx.mem_used();
    assert_eq!(used1, 4);
    let _b = ctx.new_tensor_1d(DType::F32, 1); // оффсет выровнен к 32
    let used2 = ctx.mem_used();
    assert_eq!(used2, 36, "вторая аллокация должна начаться с оффсета 32");
}
