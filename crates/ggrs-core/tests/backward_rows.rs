use ggrs_core::{
    build_backward, build_forward, compute,
    Context, DType, Graph, TensorId,
};

// ----------------------------------------------------------------
// gradcheck-хелпер (скопирован из backward_basic.rs)
// ----------------------------------------------------------------
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
// Тест: gradcheck_softmax
// ----------------------------------------------------------------
#[test]
fn gradcheck_softmax() {
    let mut ctx = Context::new(1 << 24);

    // a = [5,3] Lcg(71)
    let a = fill_lcg(&mut ctx, [5, 3, 1, 1], 71);
    // b = [5,3] Lcg(72) — фиксированный НЕ-параметр
    let b = fill_lcg(&mut ctx, [5, 3, 1, 1], 72);

    // loss = sum_all(mul(soft_max(a), b))
    let sm = ctx.soft_max(a);
    let prod = ctx.mul(sm, b);
    let loss = ctx.sum_all(prod);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    // tol увеличен из-за численной чувствительности softmax
    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 5e-2);
}

// ----------------------------------------------------------------
// Тест: gradcheck_rmsnorm
// ----------------------------------------------------------------
#[test]
fn gradcheck_rmsnorm() {
    let mut ctx = Context::new(1 << 24);

    // a = [8,2] Lcg(73)
    let a = fill_lcg(&mut ctx, [8, 2, 1, 1], 73);
    // b = [8,2] Lcg(74) — фиксированный НЕ-параметр
    let b = fill_lcg(&mut ctx, [8, 2, 1, 1], 74);
    let eps = 1e-5;

    // loss = sum_all(mul(rms_norm(a, eps), b))
    let rn = ctx.rms_norm(a, eps);
    let prod = ctx.mul(rn, b);
    let loss = ctx.sum_all(prod);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест: gradcheck_rope
// ----------------------------------------------------------------
#[test]
fn gradcheck_rope() {
    let mut ctx = Context::new(1 << 24);

    // a = [4,2,3] Lcg(75) — 4 головы, 2 строки, 3 позиции
    let a = fill_lcg(&mut ctx, [4, 2, 3, 1], 75);
    // b = [4,2,3] Lcg(76) — фиксированный НЕ-параметр
    let b = fill_lcg(&mut ctx, [4, 2, 3, 1], 76);
    // pos = [0, 1, 2]
    let pos = ctx.new_tensor_1d(DType::I32, 3);
    ctx.set_i32(pos, &[0, 1, 2]);

    // loss = sum_all(mul(rope(a, pos, 4, 10000.0), b))
    let r = ctx.rope(a, pos, 4, 10000.0);
    let prod = ctx.mul(r, b);
    let loss = ctx.sum_all(prod);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 5e-2);
}

// ----------------------------------------------------------------
// Тест: gradcheck_permute_chain (permute + cont backward)
// ----------------------------------------------------------------
#[test]
fn gradcheck_permute_chain() {
    let mut ctx = Context::new(1 << 24);

    // a = [3,4] Lcg(77)
    let a = fill_lcg(&mut ctx, [3, 4, 1, 1], 77);
    // b формы [4,3] Lcg(78)
    let b = fill_lcg(&mut ctx, [4, 3, 1, 1], 78);

    // p = permute(a, [1,0,2,3]) → форма [4,3]
    let p = ctx.permute(a, [1, 0, 2, 3]);
    // c = cont(p) → форма [4,3], contiguous
    let c = ctx.cont(p);
    // loss = sum_all(mul(c, b))
    let prod = ctx.mul(c, b);
    let loss = ctx.sum_all(prod);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}
