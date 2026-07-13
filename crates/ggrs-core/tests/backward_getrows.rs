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
// Тест 1: gradcheck для get_rows с повторяющимися индексами
// ----------------------------------------------------------------
#[test]
fn gradcheck_getrows_with_repeats() {
    let mut ctx = Context::new(1 << 24);

    // table: [4, 6] — 6 эмбеддингов длины 4
    let table = fill_lcg(&mut ctx, [4, 6, 1, 1], 61);

    // ids: [2, 0, 2] — повтор строки 2 дважды, строка 0 один раз
    let ids = ctx.new_tensor_1d(DType::I32, 3);
    ctx.set_i32(ids, &[2, 0, 2]);

    // y = get_rows(table, ids) — shape [4, 3]
    let y = ctx.get_rows(table, ids);

    // loss = sum_all(y) — скаляр
    let loss = ctx.sum_all(y);

    // помечаем table как параметр
    ctx.set_param(table);

    // строим forward-граф
    let gf = build_forward(&ctx, loss);

    // строим backward
    let bw = build_backward(&mut ctx, &gf, loss);

    // граф для backward (collect всех param-градиентов)
    let groot = build_forward(&ctx, bw.root);

    // градиент таблицы
    let grad_table = bw.grads[&table];

    // gradcheck
    gradcheck(&mut ctx, &gf, loss, grad_table, &groot, table, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 2: аналитическая проверка: строки, не входящие в ids, имеют нулевой градиент
// ----------------------------------------------------------------
#[test]
#[allow(clippy::needless_range_loop)]
fn getrows_back_zero_untouched() {
    let mut ctx = Context::new(1 << 24);

    // table: [3, 5] — 5 эмбеддингов длины 3
    let table = fill_lcg(&mut ctx, [3, 5, 1, 1], 99);

    // ids: [1, 3] — строки 1 и 3, строки 0,2,4 не задействованы
    let ids = ctx.new_tensor_1d(DType::I32, 2);
    ctx.set_i32(ids, &[1, 3]);

    let y = ctx.get_rows(table, ids);
    let loss = ctx.sum_all(y);

    ctx.set_param(table);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_table = bw.grads[&table];

    // Вычисляем backward
    compute(&ctx, &gf, 1);
    compute(&ctx, &groot, 1);

    let grads = ctx.data_f32(grad_table);

    // table: [3, 5]; градиент — 3×5 = 15 элементов
    // Строка 0: элементы 0..2  (не участвует → 0)
    // Строка 1: элементы 3..5  (участвует → ненулевой, равен 1)
    // Строка 2: элементы 6..8  (не участвует → 0)
    // Строка 3: элементы 9..11 (участвует → ненулевой, равен 1)
    // Строка 4: элементы 12..14 (не участвует → 0)

    // Строки 0, 2, 4 — строго 0
    for i in 0..3 {
        assert_eq!(grads[i], 0.0, "строка 0, элемент {i}: должен быть 0");
    }
    for i in 6..9 {
        assert_eq!(grads[i], 0.0, "строка 2, элемент {}: должен быть 0", i - 6);
    }
    for i in 12..15 {
        assert_eq!(grads[i], 0.0, "строка 4, элемент {}: должен быть 0", i - 12);
    }

    // Строки 1 и 3 — ненулевые (равны 1, так как sum_all даёт g=1)
    // ids = [1, 3], строка 1 один раз, строка 3 один раз → каждый элемент = 1
    for i in 3..6 {
        assert_eq!(grads[i], 1.0, "строка 1, элемент {}: должен быть 1", i - 3);
    }
    for i in 9..12 {
        assert_eq!(grads[i], 1.0, "строка 3, элемент {}: должен быть 1", i - 9);
    }
}
