use ggrs_core::{
    build_backward, build_forward, compute,
    util::Lcg,
    Context, DType, Graph, TensorId,
};

/// Построить граф, задействующий все backward-ядра проекта:
/// GetRowsBack, OutProd (MulMat-back), SiluBack, GeluBack, RmsNormBack,
/// SoftMaxBack, RopeBack, ReshapeBack, CrossEntropyLossBack, SumAllBack,
/// AddBack, MulBack.
///
/// Возвращает (forward-граф, backward-граф, TensorId градиента emb, TensorId градиента w).
fn build_case(ctx: &mut Context) -> (Graph, Graph, TensorId, TensorId) {
    // --- эмбеддинги ---
    let emb = ctx.new_tensor_2d(DType::F32, 8, 10);
    ctx.set_param(emb);
    {
        let mut rng = Lcg::new(81);
        let data = ctx.data_f32_mut(emb);
        for v in data.iter_mut() {
            *v = rng.next_f32();
        }
    }

    // --- ids с повтором ---
    let ids = ctx.new_tensor_1d(DType::I32, 4);
    ctx.set_i32(ids, &[0, 3, 7, 3]);

    // --- x = get_rows(emb, ids) → [8, 4] ---
    let x = ctx.get_rows(emb, ids);

    // --- весовая матрица w [8, 8] ---
    let w = ctx.new_tensor_2d(DType::F32, 8, 8);
    ctx.set_param(w);
    {
        let mut rng = Lcg::new(82);
        let data = ctx.data_f32_mut(w);
        for v in data.iter_mut() {
            *v = rng.next_f32();
        }
    }

    // --- h = mul_mat(w, x) → [8, 4] ---
    let h = ctx.mul_mat(w, x);

    // --- hs = silu(h) → [8, 4] ---
    let hs = ctx.silu(h);

    // --- hg = gelu(hs) → [8, 4] ---
    let hg = ctx.gelu(hs);

    // --- n = rms_norm(hg, 1e-5) → [8, 4] ---
    let n = ctx.rms_norm(hg, 1e-5);

    // --- targets one-hot [8, 4], классы 1, 2, 3, 4 ---
    let targets = ctx.new_tensor_2d(DType::F32, 8, 4);
    {
        let data = ctx.data_f32_mut(targets);
        data.fill(0.0);
        for t in 0..4 {
            let cls = (t + 1) % 8;
            data[cls + t * 8] = 1.0;
        }
    }

    // --- loss1 = cross_entropy_loss(n, targets) → [1] ---
    let loss1 = ctx.cross_entropy_loss(n, targets);

    // --- rope-ветка: rope(reshape_3d(h, 8, 1, 4), pos, 4, 10000.0) ---
    let h3d = ctx.reshape_3d(h, 8, 1, 4); // [8, 1, 4]
    let pos = ctx.new_tensor_1d(DType::I32, 4);
    ctx.set_i32(pos, &[0, 1, 2, 3]);
    let q = ctx.rope(h3d, pos, 4, 10000.0); // [8, 1, 4], contiguous
    let q2d = ctx.reshape_2d(q, 8, 4); // [8, 4]

    // --- sm = soft_max(q2d) → [8, 4] ---
    let sm = ctx.soft_max(q2d);

    // --- k = [8, 4], Lcg(83), НЕ параметр ---
    let k = ctx.new_tensor_2d(DType::F32, 8, 4);
    {
        let mut rng = Lcg::new(83);
        let data = ctx.data_f32_mut(k);
        for v in data.iter_mut() {
            *v = rng.next_f32();
        }
    }

    // --- mul(sm, k) → [8, 4] ---
    let mk = ctx.mul(sm, k);

    // --- loss2 = sum_all(mk) → [1] ---
    let loss2 = ctx.sum_all(mk);

    // --- total = add(loss1, loss2) → [1] ---
    let loss = ctx.add(loss1, loss2);

    // Строим forward-граф
    let gf = build_forward(ctx, loss);

    // Строим backward
    let bw = build_backward(ctx, &gf, loss);

    // Строим groot (граф градиентов параметров)
    let groot = build_forward(ctx, bw.root);

    // Градиенты emb и w
    let g_emb = bw.grads[&emb];
    let g_w = bw.grads[&w];

    (gf, groot, g_emb, g_w)
}

/// Многопоточная эквивалентность: бит-в-бит совпадение градиентов
/// при 1 потоке и при 4 потоках для ВСЕХ backward-ядер.
#[test]
fn threading_backward_equivalence() {
    // --- 1 поток ---
    let mut c1 = Context::new(1 << 24);
    let (gf1, groot1, g_emb1, g_w1) = build_case(&mut c1);

    compute(&mut c1, &gf1, 1);
    compute(&mut c1, &groot1, 1);

    // --- 4 потока ---
    let mut c4 = Context::new(1 << 24);
    let (gf4, groot4, g_emb4, g_w4) = build_case(&mut c4);

    compute(&mut c4, &gf4, 4);
    compute(&mut c4, &groot4, 4);

    // --- бит-в-бит сравнение градиентов ---
    assert_eq!(
        c1.data_f32(g_emb1),
        c4.data_f32(g_emb4),
        "threading_backward_equivalence: градиент emb не совпадает бит-в-бит"
    );
    assert_eq!(
        c1.data_f32(g_w1),
        c4.data_f32(g_w4),
        "threading_backward_equivalence: градиент w не совпадает бит-в-бит"
    );
}
