use ggrs_core::*;

/// Один трансформер-блок (1 голова, без масок упрощений не делаем — маска causal через add)
/// на псевдослучайных весах: проверяем конечность значений и эквивалентность 1 vs 8 потоков.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.2
}

fn fill(ctx: &mut Context, id: TensorId, seed: &mut u64) {
    let n = ctx.t(id).nelements();
    let v: Vec<f32> = (0..n).map(|_| lcg(seed)).collect();
    ctx.set_f32(id, &v);
}

fn build(ctx: &mut Context) -> TensorId {
    let (d, t, vocab) = (16usize, 6usize, 32usize);
    let mut seed = 7u64;

    let emb = ctx.new_tensor_2d(DType::F32, d, vocab);
    fill(ctx, emb, &mut seed);
    let wq = ctx.new_tensor_2d(DType::F32, d, d);
    let wk = ctx.new_tensor_2d(DType::F32, d, d);
    let wv = ctx.new_tensor_2d(DType::F32, d, d);
    let wo = ctx.new_tensor_2d(DType::F32, d, d);
    for w in [wq, wk, wv, wo] {
        fill(ctx, w, &mut seed);
    }
    let w_up = ctx.new_tensor_2d(DType::F32, d, 4 * d);
    let w_gate = ctx.new_tensor_2d(DType::F32, d, 4 * d);
    let w_down = ctx.new_tensor_2d(DType::F32, 4 * d, d);
    for w in [w_up, w_gate, w_down] {
        fill(ctx, w, &mut seed);
    }

    let ids = ctx.new_tensor_1d(DType::I32, t);
    ctx.set_i32(ids, &[3, 1, 4, 1, 5, 9]);
    let pos = ctx.new_tensor_1d(DType::I32, t);
    ctx.set_i32(pos, &[0, 1, 2, 3, 4, 5]);

    // causal маска [t, t]: 0 на/ниже диагонали, -1e9 выше
    let mask = ctx.new_tensor_2d(DType::F32, t, t);
    let mv: Vec<f32> = (0..t * t)
        .map(|i| if i % t > i / t { -1e9 } else { 0.0 })
        .collect();
    ctx.set_f32(mask, &mv);

    let x = ctx.get_rows(emb, ids); // [d, t]
    let xn = ctx.rms_norm(x, 1e-5);

    // 1 голова: q,k,v = [d, t]
    let q0 = ctx.mul_mat(wq, xn);
    let k0 = ctx.mul_mat(wk, xn);
    let v0 = ctx.mul_mat(wv, xn);
    let q = ctx.reshape_3d(q0, d, 1, t);
    let k = ctx.reshape_3d(k0, d, 1, t);
    let qr0 = ctx.rope(q, pos, d, 10000.0);
    let kr0 = ctx.rope(k, pos, d, 10000.0);
    let qr = ctx.reshape_2d(qr0, d, t);
    let kr = ctx.reshape_2d(kr0, d, t);

    let att0 = ctx.mul_mat(kr, qr); // [t(k-строки), t(q-строки)]
    let att1 = ctx.scale(att0, 1.0 / (d as f32).sqrt());
    let att2 = ctx.add(att1, mask);
    let att = ctx.soft_max(att2);

    // out[d, t]: для каждого q-токена — взвешенная сумма v; v^T [t, d] строками
    let vt0 = ctx.transpose(v0); // [t, d] view
    let vt = ctx.cont(vt0);
    let out0 = ctx.mul_mat(vt, att); // [d, t]
    let att_out = ctx.mul_mat(wo, out0);

    let h = ctx.add(x, att_out);
    let hn = ctx.rms_norm(h, 1e-5);
    let up = ctx.mul_mat(w_up, hn);
    let gate = ctx.mul_mat(w_gate, hn);
    let gate_s = ctx.silu(gate);
    let ff0 = ctx.mul(gate_s, up);
    let ff = ctx.mul_mat(w_down, ff0);
    let h2 = ctx.add(h, ff);

    let logits = ctx.mul_mat(emb, h2); // tied embeddings: [vocab, t]

    // one-hot цели: следующий токен
    let targets = ctx.new_tensor_2d(DType::F32, vocab, t);
    let mut tv = vec![0.0f32; vocab * t];
    for (r, &c) in [1i32, 4, 1, 5, 9, 2].iter().enumerate() {
        tv[r * vocab + c as usize] = 1.0;
    }
    ctx.set_f32(targets, &tv);
    ctx.cross_entropy_loss(logits, targets)
}

#[test]
fn transformer_block_forward() {
    let mut c1 = Context::new(1 << 24);
    let l1 = build(&mut c1);
    let g1 = build_forward(&c1, l1);
    compute(&mut c1, &g1, 1);
    let loss1 = c1.data_f32(l1)[0];
    assert!(loss1.is_finite(), "loss = {loss1}");
    // при случайных весах и vocab=32 loss ~ ln(32) ± немного
    assert!(loss1 > 1.0 && loss1 < 8.0, "loss = {loss1}");

    let mut c8 = Context::new(1 << 24);
    let l8 = build(&mut c8);
    let g8 = build_forward(&c8, l8);
    compute(&mut c8, &g8, 8);
    assert_eq!(loss1, c8.data_f32(l8)[0], "1 поток vs 8 потоков");
}
