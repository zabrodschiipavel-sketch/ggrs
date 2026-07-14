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

/// Как gradcheck, но проходит, если ЛИБО абсолютная, ЛИБО относительная ошибка мала.
/// Нужен для нелинейных цепочек (softmax): на элементах с истинно околонулевым
/// градиентом относительная ошибка бессмысленна (знаменатель ~0), тогда как
/// абсолютная ошибка остаётся на уровне floor'а конечных разностей f32 (~5e-5).
/// На градиентах O(1) работает rel_tol и ловит реальные ошибки формул.
#[allow(clippy::too_many_arguments)]
fn gradcheck_absrel(
    ctx: &mut Context,
    gf: &Graph,
    loss: TensorId,
    grad: TensorId,
    groot: &Graph,
    param: TensorId,
    eps: f32,
    abs_tol: f32,
    rel_tol: f32,
) {
    let data = ctx.data_f32(param).to_vec();
    for (i, &x) in data.iter().enumerate() {
        ctx.data_f32_mut(param)[i] = x + eps;
        compute(ctx, gf, 1);
        let l1 = ctx.data_f32(loss)[0];
        ctx.data_f32_mut(param)[i] = x - eps;
        compute(ctx, gf, 1);
        let l2 = ctx.data_f32(loss)[0];
        ctx.data_f32_mut(param)[i] = x;
        let fd = (l1 - l2) / (2.0 * eps);
        compute(ctx, gf, 1);
        compute(ctx, groot, 1);
        let analytic = ctx.data_f32(grad)[i];
        let abs_err = (analytic - fd).abs();
        let rel_err = abs_err / (fd.abs() + 1e-4);
        assert!(
            abs_err < abs_tol || rel_err < rel_tol,
            "gradcheck_absrel: param[{}] analytic={:.6} fd={:.6} abs_err={:.6} rel_err={:.6} (abs_tol={} rel_tol={})",
            i, analytic, fd, abs_err, rel_err, abs_tol, rel_tol,
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
// Тест 1: gradcheck для ∂a и ∂b mul_mat (2D)
// ----------------------------------------------------------------
#[test]
fn gradcheck_mulmat_a_and_b() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [4, 3, 1, 1], 31);
    let b = fill_lcg(&mut ctx, [4, 2, 1, 1], 32);

    // loss = sum_all(mul_mat(a, b))
    let mm = ctx.mul_mat(a, b);
    let loss = ctx.sum_all(mm);

    // помечаем оба параметрами
    ctx.set_param(a);
    ctx.set_param(b);

    // строим forward-граф
    let gf = build_forward(&ctx, loss);

    // строим backward
    let bw = build_backward(&mut ctx, &gf, loss);

    // граф для backward (collect всех param-градиентов)
    let groot = build_forward(&ctx, bw.root);

    // градиенты параметров
    let grad_a = bw.grads[&a];
    let grad_b = bw.grads[&b];

    // gradcheck для a и b
    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
    gradcheck(&mut ctx, &gf, loss, grad_b, &groot, b, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 2: gradcheck для mul_mat в цепочке с scale (2D)
// ----------------------------------------------------------------
#[test]
fn gradcheck_mulmat_chain() {
    let mut ctx = Context::new(1 << 24);

    let a = fill_lcg(&mut ctx, [5, 3, 1, 1], 41);
    let b = fill_lcg(&mut ctx, [5, 4, 1, 1], 42);

    // loss = sum_all(scale(mul_mat(a, b), 0.3))
    let mm = ctx.mul_mat(a, b);
    let scaled = ctx.scale(mm, 0.3);
    let loss = ctx.sum_all(scaled);

    ctx.set_param(a);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 3: gradcheck 3D mul_mat для ∂a и ∂b
// ----------------------------------------------------------------
#[test]
fn gradcheck_mulmat_3d_a_and_b() {
    let mut ctx = Context::new(1 << 24);

    // a: [k=4, m=3, b=2], b: [k=4, n=5, b=2] → dst: [3, 5, 2]
    let a = fill_lcg(&mut ctx, [4, 3, 2, 1], 61);
    let b = fill_lcg(&mut ctx, [4, 5, 2, 1], 62);

    // loss = sum_all(mul_mat(a, b))
    let mm = ctx.mul_mat(a, b);
    assert_eq!(ctx.t(mm).ne, [3, 5, 2, 1]);
    let loss = ctx.sum_all(mm);

    ctx.set_param(a);
    ctx.set_param(b);

    let gf = build_forward(&ctx, loss);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];
    let grad_b = bw.grads[&b];

    // Проверяем формы градиентов
    assert_eq!(ctx.t(grad_a).ne, [4, 3, 2, 1], "grad_a форма");
    assert_eq!(ctx.t(grad_b).ne, [4, 5, 2, 1], "grad_b форма");

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 2e-2);
    gradcheck(&mut ctx, &gf, loss, grad_b, &groot, b, 1e-3, 2e-2);
}

// ----------------------------------------------------------------
// Тест 4: 3D mul_mat → softmax → поэлементное mul → sum_all
// (не-uniform градиент, чтобы softmax backward давал ненулевые значения)
// ----------------------------------------------------------------
#[test]
fn gradcheck_mulmat_softmax_3d() {
    let mut ctx = Context::new(1 << 24);

    // a: [4, 3, 2], b: [4, 3, 2] → mm: [3, 3, 2] → softmax → mul(w) → sum_all
    let a = fill_lcg(&mut ctx, [4, 3, 2, 1], 81);
    let b = fill_lcg(&mut ctx, [4, 3, 2, 1], 82);
    // w: фиксированный не-uniform вес той же формы, что и softmax-выход
    let w = fill_lcg(&mut ctx, [3, 3, 2, 1], 83);

    let mm = ctx.mul_mat(a, b);
    let sm = ctx.soft_max(mm);
    // поэлементное умножение на w даёт не-uniform градиент для softmax
    let weighted = ctx.mul(sm, w);
    let loss = ctx.sum_all(weighted);

    ctx.set_param(a);
    ctx.set_param(b);

    let gf = build_forward(&ctx, loss);
    compute(&ctx, &gf, 1);
    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_a = bw.grads[&a];
    let grad_b = bw.grads[&b];

    gradcheck(&mut ctx, &gf, loss, grad_a, &groot, a, 1e-3, 5e-2);
    gradcheck(&mut ctx, &gf, loss, grad_b, &groot, b, 1e-3, 5e-2);
}

// ----------------------------------------------------------------
// Тест 5: мини-attention 3D — gradcheck по q, k, v
// ----------------------------------------------------------------
#[test]
fn gradcheck_attention_mini_3d() {
    let mut ctx = Context::new(1 << 26);

    let hd = 4usize; // head_dim
    let t = 3usize;  // seq_len
    let h = 2usize;  // n_heads

    // q, k, v: [hd, t, h] — все параметры
    let q = fill_lcg(&mut ctx, [hd, t, h, 1], 71);
    let k = fill_lcg(&mut ctx, [hd, t, h, 1], 72);
    let v = fill_lcg(&mut ctx, [hd, t, h, 1], 73);

    // attention:
    // att0 = mul_mat(k, q)            — k: [hd, t, h], q: [hd, t, h] → [t, t, h]
    // att1 = scale(att0, 1/sqrt(hd))
    // att2 = diag_mask_inf(att1)      — каузальная
    // att  = soft_max(att2)
    // vt   = cont(transpose(v))       — v: [hd, t, h] → transpose → [t, hd, h] → cont
    // out  = mul_mat(vt, att)         — vt: [t, hd, h], att: [t, t, h] → [hd, t, h]
    // loss = sum_all(out)

    let att0 = ctx.mul_mat(k, q);
    assert_eq!(ctx.t(att0).ne, [t, t, h, 1]);

    let scale_factor = 1.0 / (hd as f32).sqrt();
    let att1 = ctx.scale(att0, scale_factor);

    let att2 = ctx.diag_mask_inf(att1);

    let att = ctx.soft_max(att2);

    // v: [hd, t, h] → transpose → [t, hd, h] (не-contiguous) → cont
    let vt = ctx.transpose(v);
    assert!(!ctx.t(vt).is_contiguous(), "transpose(v) должен быть не-contiguous");
    let vtc = ctx.cont(vt);

    let out = ctx.mul_mat(vtc, att);
    assert_eq!(ctx.t(out).ne, [hd, t, h, 1]);

    let loss = ctx.sum_all(out);

    ctx.set_param(q);
    ctx.set_param(k);
    ctx.set_param(v);

    let gf = build_forward(&ctx, loss);

    // Вычисляем forward один раз, чтобы soft_max и т.п. имели актуальные значения
    compute(&ctx, &gf, 1);

    let bw = build_backward(&mut ctx, &gf, loss);
    let groot = build_forward(&ctx, bw.root);

    let grad_q = bw.grads[&q];
    let grad_k = bw.grads[&k];
    let grad_v = bw.grads[&v];

    // Смешанный допуск: на околонулевых градиентах (softmax насыщается на
    // границе каузальной маски) относительная ошибка бессмысленна — там
    // проверяем абсолютную (floor FD f32 ~5e-5, берём запас 1e-3); на
    // градиентах O(1) действует rel_tol=5e-2 и ловит ошибки формул.
    gradcheck_absrel(&mut ctx, &gf, loss, grad_q, &groot, q, 1e-3, 1e-3, 5e-2);
    gradcheck_absrel(&mut ctx, &gf, loss, grad_k, &groot, k, 1e-3, 1e-3, 5e-2);
    gradcheck_absrel(&mut ctx, &gf, loss, grad_v, &groot, v, 1e-3, 1e-3, 5e-2);
}
