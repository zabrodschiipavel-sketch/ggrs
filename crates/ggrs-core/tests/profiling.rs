use ggrs_core::*;
use ggrs_core::util::Lcg;

#[test]
fn profile_returns_timings() {
    let mut ctx = Context::new(1 << 24);

    // Маленький граф: mul_mat 32x32 -> soft_max
    // a: [k=32, m=32], b: [k=32, n=32] -> mul_mat даёт [m=32, n=32] -> soft_max
    let k = 32usize;
    let m = 32;
    let n = 32;

    let a = ctx.new_tensor_2d(DType::F32, k, m);
    let b = ctx.new_tensor_2d(DType::F32, k, n);

    // Заполняем случайными числами через Lcg
    let mut rng = Lcg::new(7);
    let av: Vec<f32> = (0..k * m).map(|_| rng.next_f32()).collect();
    let bv: Vec<f32> = (0..k * n).map(|_| rng.next_f32()).collect();
    ctx.set_f32(a, &av);
    ctx.set_f32(b, &bv);

    let d = ctx.mul_mat(a, b);
    let s = ctx.soft_max(d);
    let graph = build_forward(&ctx, s);

    let timings = compute_profiled(&mut ctx, &graph, 1);

    // В графе: a (None, view), b (None, view), d (MulMat, compute), s (SoftMax, compute)
    // + a, b могут быть Param/None... проверим только не-view узлы.
    // Всего узлов в графе: a, b, mul_mat, soft_max = 4 узла.
    // Из них compute: MulMat и SoftMax -> 2.
    let total_calls: u32 = timings.iter().map(|(_, c, _)| c).sum();
    assert_eq!(total_calls, 2, "должно быть 2 compute-узла: MulMat и SoftMax");

    for (_, _, ms) in &timings {
        assert!(*ms >= 0.0, "время не может быть отрицательным");
    }

    let has_mulmat = timings.iter().any(|(op, _, _)| *op == Op::MulMat);
    assert!(has_mulmat, "должен быть MulMat в результатах");
}

#[test]
fn profiled_equals_plain() {
    // Строим граф, запускаем compute (plain) и compute_profiled на разных контекстах,
    // проверяем побитное равенство data_f32 результата.

    let mut ctx_plain = Context::new(1 << 24);
    let mut ctx_prof = Context::new(1 << 24);

    let k = 32usize;
    let m = 16;
    let n = 24;

    // Одинаковые данные
    let mut rng_a = Lcg::new(123);
    let mut rng_b = Lcg::new(456);

    let av: Vec<f32> = (0..k * m).map(|_| rng_a.next_f32()).collect();
    let bv: Vec<f32> = (0..k * n).map(|_| rng_b.next_f32()).collect();

    // Контекст для plain
    let a1 = ctx_plain.new_tensor_2d(DType::F32, k, m);
    let b1 = ctx_plain.new_tensor_2d(DType::F32, k, n);
    ctx_plain.set_f32(a1, &av);
    ctx_plain.set_f32(b1, &bv);
    let d1 = ctx_plain.mul_mat(a1, b1);
    let s1 = ctx_plain.soft_max(d1);
    let g1 = build_forward(&ctx_plain, s1);

    // Контекст для profiled
    let a2 = ctx_prof.new_tensor_2d(DType::F32, k, m);
    let b2 = ctx_prof.new_tensor_2d(DType::F32, k, n);
    ctx_prof.set_f32(a2, &av);
    ctx_prof.set_f32(b2, &bv);
    let d2 = ctx_prof.mul_mat(a2, b2);
    let s2 = ctx_prof.soft_max(d2);
    let g2 = build_forward(&ctx_prof, s2);

    // Plain compute
    compute(&mut ctx_plain, &g1, 1);

    // Profiled compute (через compute_profiled напрямую)
    let _timings = compute_profiled(&mut ctx_prof, &g2, 1);

    // Сравнение
    assert_eq!(ctx_plain.data_f32(s1), ctx_prof.data_f32(s2),
        "profiled и plain дают разные результаты");
}
