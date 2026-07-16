use ggrs_core::{
    build_backward, build_forward, compute,
    Context, DType, Graph, TensorId,
};

/// Сравнение аналитического градиента с конечными разностями.
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
// Тест 1: gradcheck градиента logits для CrossEntropyLoss
// ----------------------------------------------------------------
#[test]
fn gradcheck_ce_logits() {
    let mut ctx = Context::new(1 << 24);

    // logits = [5, 3] (5 классов, 3 строки)
    let logits = fill_lcg(&mut ctx, [5, 3, 1, 1], 51);

    // one-hot targets: строка 0 → класс 2, строка 1 → класс 0, строка 2 → класс 4
    let targets = ctx.new_tensor_2d(DType::F32, 5, 3);
    {
        let data = ctx.data_f32_mut(targets);
        data.fill(0.0);
        data[2] = 1.0;   // строка 0, класс 2
        data[5] = 1.0;   // строка 1, класс 0
        data[14] = 1.0;  // строка 2, класс 4
    }

    let loss = ctx.cross_entropy_loss(logits, targets);
    ctx.set_param(logits);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_logits = bw.grads[&logits];

    gradcheck(&mut ctx, &gf, loss, grad_logits, &groot, logits, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 2: аналитическая проверка нормировки при нулевых logits
// ----------------------------------------------------------------
#[test]
fn ce_back_rows_scaling() {
    let mut ctx = Context::new(1 << 24);

    // logits = [5, 3] — все нули (softmax равномерный 1/5)
    let logits = ctx.new_tensor_2d(DType::F32, 5, 3);
    {
        let data = ctx.data_f32_mut(logits);
        data.fill(0.0);
    }

    // one-hot targets
    let targets = ctx.new_tensor_2d(DType::F32, 5, 3);
    {
        let data = ctx.data_f32_mut(targets);
        data.fill(0.0);
        data[2] = 1.0;  // строка 0 → класс 2 (idx = 0*5 + 2)
        data[5] = 1.0;  // строка 1 → класс 0 (idx = 1*5 + 0)
        data[14] = 1.0; // строка 2 → класс 4 (idx = 2*5 + 4)
    }

    let loss = ctx.cross_entropy_loss(logits, targets);
    ctx.set_param(logits);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    // Вычислить forward и backward
    compute(&mut ctx, &gf, 1);
    compute(&mut ctx, &groot, 1);

    let grad = bw.grads[&logits];
    let grad_data = ctx.data_f32(grad);

    // Аналитические ожидания: softmax = 1/5 = 0.2 для каждого элемента.
    // g0 = 1.0, nrows = 3, inv_nrows = 1/3.
    // Для целевого класса (t=1): (0.2 - 1.0) * 1.0 / 3 = -0.8/3 ≈ -0.2666667
    // Для остальных (t=0): (0.2 - 0.0) * 1.0 / 3 = 0.2/3 ≈ 0.0666667
    let expected_target = (0.2f32 - 1.0) / 3.0; // ≈ -0.2666667
    let expected_other = (0.2f32 - 0.0) / 3.0; // ≈ 0.0666667

    let tol = 1e-6;

    // Строка 0 (idx 0..4): класс 2 → idx=2
    assert!((grad_data[2] - expected_target).abs() < tol,
        "строка 0, класс 2: {} vs {}", grad_data[2], expected_target);
    for &c in &[0, 1, 3, 4] {
        assert!((grad_data[c] - expected_other).abs() < tol,
            "строка 0, класс {}: {} vs {}", c, grad_data[c], expected_other);
    }

    // Строка 1 (idx 5..9): класс 0 → idx=5
    assert!((grad_data[5] - expected_target).abs() < tol,
        "строка 1, класс 0: {} vs {}", grad_data[5], expected_target);
    for &c in &[6, 7, 8, 9] {
        assert!((grad_data[c] - expected_other).abs() < tol,
            "строка 1, класс {}: {} vs {}", c - 5, grad_data[c], expected_other);
    }

    // Строка 2 (idx 10..14): класс 4 → idx=14
    assert!((grad_data[14] - expected_target).abs() < tol,
        "строка 2, класс 4: {} vs {}", grad_data[14], expected_target);
    for &c in &[10, 11, 12, 13] {
        assert!((grad_data[c] - expected_other).abs() < tol,
            "строка 2, класс {}: {} vs {}", c - 10, grad_data[c], expected_other);
    }
}
