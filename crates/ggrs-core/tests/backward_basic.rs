use ggrs_core::{
    build_backward, build_forward, compute,
    Context, DType, Graph, TensorId,
};

/// Сравнение аналитического градиента с конечными разностями.
/// Для каждого элемента param: x±eps → два compute полного forward-графа → FD;
/// сравнить с data_f32(grad)[i] после compute(groot-граф,1) на восстановленных данных.
#[allow(clippy::too_many_arguments)]
fn gradcheck(
    ctx: &mut Context,
    gf: &Graph,
    loss: TensorId,
    grad: TensorId,
    groot: &Graph,
    param: TensorId,
    eps: f32,
    tol: f32,
) {
    let data = ctx.data_f32(param).to_vec();

    for (i, &x) in data.iter().enumerate() {

        // x + eps
        ctx.data_f32_mut(param)[i] = x + eps;
        compute(ctx, gf, 1);
        let l1 = ctx.data_f32(loss)[0];

        // x - eps
        ctx.data_f32_mut(param)[i] = x - eps;
        compute(ctx, gf, 1);
        let l2 = ctx.data_f32(loss)[0];

        // восстановить
        ctx.data_f32_mut(param)[i] = x;

        // конечная разность
        let fd = (l1 - l2) / (2.0 * eps);

        // вычислить градиент на невозмущённых данных
        compute(ctx, gf, 1);
        compute(ctx, groot, 1);
        let analytic = ctx.data_f32(grad)[i];

        let denom = fd.abs() + 1e-4;
        let rel_err = (analytic - fd).abs() / denom;
        assert!(
            rel_err < tol,
            "gradcheck: param[{}] analytic={:.6} fd={:.6} rel_err={:.6} > tol={}",
            i, analytic, fd, rel_err, tol,
        );
    }
}

/// Создать LCG-заполненный F32-тензор формы [rows, cols].
fn fill_lcg(ctx: &mut Context, shape: [usize; 4], seed: u64) -> TensorId {
    let t = ctx.new_tensor(DType::F32, shape);
    let mut rng = ggrs_core::util::Lcg::new(seed);
    let data = ctx.data_f32_mut(t);
    for v in data.iter_mut() {
        *v = rng.next_f32();
    }
    t
}

// ----------------------------------------------------------------
// Тест 1: базовый backward для Add/Mul/Scale/SumAll
// ----------------------------------------------------------------
#[test]
fn test_backward_basic() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [6, 3, 1, 1], 1);
    let b = fill_lcg(&mut ctx, [6, 3, 1, 1], 2);
    let c = fill_lcg(&mut ctx, [6, 3, 1, 1], 3);

    // z = scale(mul(add(a,b), c), 0.5)
    let t = ctx.add(a, b);
    let t = ctx.mul(t, c);
    let z = ctx.scale(t, 0.5);

    // loss = sum_all(z)
    let loss = ctx.sum_all(z);

    // помечаем параметры
    ctx.set_param(a);
    ctx.set_param(c);

    // строим forward-граф
    let gf = build_forward(&ctx, loss);

    // строим backward
    let bw = build_backward(&mut ctx, &gf, loss);

    // граф для backward (collect всех param-градиентов)
    let groot = build_forward(&ctx, bw.root);

    // градиенты параметров
    let grad_a = bw.grads[&a];
    let grad_c = bw.grads[&c];

    // gradcheck для a
    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
    // gradcheck для c
    gradcheck(&mut ctx, &gf, loss, grad_c, &groot, c, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 2: аккумуляция градиентов (a дважды используется в mul)
// ----------------------------------------------------------------
#[test]
fn test_backward_accumulation() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [6, 3, 1, 1], 5);

    // y = mul(a, a)
    let y = ctx.mul(a, a);
    let loss = ctx.sum_all(y);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    // аналитический ∂a = 2*a (g=1)
    // gradcheck
    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 3: цепочка collect при >4 src
// ----------------------------------------------------------------
#[test]
fn test_collect_chain() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [2, 1, 1, 1], 10);
    let b = fill_lcg(&mut ctx, [2, 1, 1, 1], 11);
    let c = fill_lcg(&mut ctx, [2, 1, 1, 1], 12);
    let d = fill_lcg(&mut ctx, [2, 1, 1, 1], 13);
    let e = fill_lcg(&mut ctx, [2, 1, 1, 1], 14);

    // 5 src — должна создать цепочку collect'ов
    let collected = ctx.collect(&[a, b, c, d, e]);
    let gf = build_forward(&ctx, collected);
    // compute должен пройти без ошибок (no-op ядра)
    compute(&ctx, &gf, 1);
}

// ----------------------------------------------------------------
// Аудит P1: broadcast-backward для Mul запрещён (молча неверный градиент хуже паники)
// ----------------------------------------------------------------
#[test]
#[should_panic(expected = "Mul backward: broadcast не поддержан")]
fn mul_backward_rejects_broadcast() {
    let mut ctx = Context::new(1 << 20);
    let a = fill_lcg(&mut ctx, [4, 3, 1, 1], 90);
    let b = fill_lcg(&mut ctx, [4, 1, 1, 1], 91); // broadcast по строкам — forward валиден
    let prod = ctx.mul(a, b);
    let loss = ctx.sum_all(prod);
    ctx.set_param(a);
    let gf = build_forward(&ctx, loss);
    let _ = build_backward(&mut ctx, &gf, loss);
}
