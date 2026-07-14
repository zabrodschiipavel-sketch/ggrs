use ggrs_core::{
    build_backward, build_forward, compute,
    Context, DType, Graph, TensorId,
};

// ----------------------------------------------------------------
// gradcheck-хелпер (скопирован из backward_rows.rs)
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
// Тест 1: diag_mask_forward_2d
// ----------------------------------------------------------------
#[test]
fn diag_mask_forward_2d() {
    let mut ctx = Context::new(1 << 20);

    // a [4,4,1,1] со значениями 1..=16
    let a = ctx.new_tensor_2d(DType::F32, 4, 4);
    let vals: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    ctx.set_f32(a, &vals);

    let mask = ctx.diag_mask_inf(a);
    let gf = build_forward(&ctx, mask);
    compute(&ctx, &gf, 1);

    // Проверка: i0 > i1 → NEG_INFINITY, иначе исходное значение
    for i1 in 0..4 {
        for i0 in 0..4 {
            let val = ctx.get_f32(mask, [i0, i1, 0, 0]);
            let expected = if i0 > i1 {
                f32::NEG_INFINITY
            } else {
                (i1 * 4 + i0 + 1) as f32
            };
            assert_eq!(val, expected,
                "diag_mask_forward_2d: [i0={}, i1={}] ожидалось {:?}, получено {:?}",
                i0, i1, expected, val);
        }
    }
}

// ----------------------------------------------------------------
// Тест 2: diag_mask_forward_3d
// ----------------------------------------------------------------
#[test]
fn diag_mask_forward_3d() {
    let mut ctx = Context::new(1 << 20);

    // a [3,3,2,1]
    let a = ctx.new_tensor_3d(DType::F32, 3, 3, 2);
    let vals: Vec<f32> = (1..=18).map(|v| v as f32).collect();
    ctx.set_f32(a, &vals);

    let mask = ctx.diag_mask_inf(a);
    let gf = build_forward(&ctx, mask);
    compute(&ctx, &gf, 1);

    // Проверка обеих голов
    for i2 in 0..2 {
        for i1 in 0..3 {
            for i0 in 0..3 {
                let val = ctx.get_f32(mask, [i0, i1, i2, 0]);
                let idx_1d = i2 * 9 + i1 * 3 + i0;
                let expected = if i0 > i1 {
                    f32::NEG_INFINITY
                } else {
                    (idx_1d + 1) as f32
                };
                assert_eq!(val, expected,
                    "diag_mask_forward_3d: [i0={}, i1={}, i2={}] ожидалось {:?}, получено {:?}",
                    i0, i1, i2, expected, val);
            }
        }
    }
}

// ----------------------------------------------------------------
// Тест 3: diag_mask_softmax_row0
// ----------------------------------------------------------------
#[test]
fn diag_mask_softmax_row0() {
    let mut ctx = Context::new(1 << 20);

    // a [4,4,1,1] — все единицы
    let a = ctx.new_tensor_2d(DType::F32, 4, 4);
    ctx.set_f32(a, &[1.0; 16]);

    // цепочка: diag_mask_inf → soft_max
    let mask = ctx.diag_mask_inf(a);
    let sm = ctx.soft_max(mask);
    let gf = build_forward(&ctx, sm);
    compute(&ctx, &gf, 1);

    // Строка запроса i1=0: только i0=0 валидна → softmax = [1, 0, 0, 0]
    for i0 in 0..4 {
        let val = ctx.get_f32(sm, [i0, 0, 0, 0]);
        let expected = if i0 == 0 { 1.0 } else { 0.0 };
        let diff = (val - expected).abs();
        assert!(diff < 1e-6,
            "diag_mask_softmax_row0: [i0={}] ожидалось {:.6}, получено {:.6}",
            i0, expected, val);
    }
}

// ----------------------------------------------------------------
// Тест 4: gradcheck_diag_mask_chain
// ----------------------------------------------------------------
#[test]
fn gradcheck_diag_mask_chain() {
    let mut ctx = Context::new(1 << 24);

    // x [4,4] Lcg(31) — параметр
    let x = fill_lcg(&mut ctx, [4, 4, 1, 1], 31);
    // b [4,4] Lcg(32) — фиксированный
    let b = fill_lcg(&mut ctx, [4, 4, 1, 1], 32);

    // loss = sum_all(mul(soft_max(diag_mask_inf(x)), b))
    let mask = ctx.diag_mask_inf(x);
    let sm = ctx.soft_max(mask);
    let prod = ctx.mul(sm, b);
    let loss = ctx.sum_all(prod);

    ctx.set_param(x);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_x = bw.grads[&x];

    gradcheck(&mut ctx, &gf, loss, grad_x, &groot, x, 1e-3, 5e-2);
}

// ----------------------------------------------------------------
// Тест 5: diag_mask_threads_parity
// ----------------------------------------------------------------
#[test]
fn diag_mask_threads_parity() {
    let mut ctx1 = Context::new(1 << 20);
    let mut ctx2 = Context::new(1 << 20);

    // a [8,8,4,1] Lcg(41)
    let a1 = fill_lcg(&mut ctx1, [8, 8, 4, 1], 41);
    let a2 = fill_lcg(&mut ctx2, [8, 8, 4, 1], 41);

    let mask1 = ctx1.diag_mask_inf(a1);
    let mask2 = ctx2.diag_mask_inf(a2);

    let gf1 = build_forward(&ctx1, mask1);
    let gf2 = build_forward(&ctx2, mask2);

    compute(&ctx1, &gf1, 1);
    compute(&ctx2, &gf2, 4);

    // Сравнение поэлементно через get_f32
    for i3 in 0..1 {
        for i2 in 0..4 {
            for i1 in 0..8 {
                for i0 in 0..8 {
                    let v1 = ctx1.get_f32(mask1, [i0, i1, i2, i3]);
                    let v2 = ctx2.get_f32(mask2, [i0, i1, i2, i3]);
                    assert_eq!(v1, v2,
                        "diag_mask_threads_parity: [i0={}, i1={}, i2={}] 1-thread={:?} 4-threads={:?}",
                        i0, i1, i2, v1, v2);
                }
            }
        }
    }
}
