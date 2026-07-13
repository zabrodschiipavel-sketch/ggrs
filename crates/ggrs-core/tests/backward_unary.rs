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
// Тест 1: gradcheck silu через matmul
// loss = sum_all(silu(mul_mat(a,b))) с a=[6,3], b=[6,2]
// ----------------------------------------------------------------
#[test]
fn gradcheck_silu() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [6, 3, 1, 1], 41);
    let b = fill_lcg(&mut ctx, [6, 2, 1, 1], 42);

    // mul_mat(a[6,3], b[6,2]) -> [3,2]
    let mm = ctx.mul_mat(a, b);
    let s = ctx.silu(mm);
    let loss = ctx.sum_all(s);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 2: gradcheck gelu через matmul
// loss = sum_all(gelu(mul_mat(a,b))) с a=[6,3], b=[6,2]
// ----------------------------------------------------------------
#[test]
fn gradcheck_gelu() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [6, 3, 1, 1], 43);
    let b = fill_lcg(&mut ctx, [6, 2, 1, 1], 44);

    // mul_mat(a[6,3], b[6,2]) -> [3,2]
    let mm = ctx.mul_mat(a, b);
    let g = ctx.gelu(mm);
    let loss = ctx.sum_all(g);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 3: gradcheck silu_direct — прямой градиент без matmul
// loss = sum_all(silu(a)), a=[5,4]
// ----------------------------------------------------------------
#[test]
fn gradcheck_silu_direct() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [5, 4, 1, 1], 45);

    let s = ctx.silu(a);
    let loss = ctx.sum_all(s);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}
