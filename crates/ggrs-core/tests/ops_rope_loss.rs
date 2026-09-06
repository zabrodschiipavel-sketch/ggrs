use ggrs_core::*;

#[test]
fn rope_rotates_pairs() {
    let mut ctx = Context::new(1 << 20);
    // head_dim=4, 1 голова, 2 позиции
    let a = ctx.new_tensor_3d(DType::F32, 4, 1, 2);
    ctx.set_f32(a, &[1., 0., 1., 0., 1., 0., 1., 0.]);
    let pos = ctx.new_tensor_1d(DType::I32, 2);
    ctx.set_i32(pos, &[0, 1]);
    let r = ctx.rope(a, pos, 4, 10000.0);
    let g = build_forward(&ctx, r);
    compute(&mut ctx, &g, 1);
    let v = ctx.data_f32(r);
    // pos=0: без изменений
    assert!((v[0] - 1.0).abs() < 1e-6 && v[1].abs() < 1e-6);
    // pos=1, пара 0: theta = 1 * 10000^(0) = 1.0 → (cos1, sin1)
    assert!((v[4] - 1f32.cos()).abs() < 1e-6);
    assert!((v[5] - 1f32.sin()).abs() < 1e-6);
    // pos=1, пара 1: theta = 10000^(-2/4) = 0.01 → (cos0.01, sin0.01)
    assert!((v[6] - 0.01f32.cos()).abs() < 1e-6);
    assert!((v[7] - 0.01f32.sin()).abs() < 1e-6);
    // норма пары сохраняется
    assert!((v[4] * v[4] + v[5] * v[5] - 1.0).abs() < 1e-5);
}

#[test]
fn cross_entropy_uniform() {
    let mut ctx = Context::new(1 << 20);
    let logits = ctx.new_tensor_2d(DType::F32, 4, 2);
    ctx.set_f32(logits, &[0.; 8]); // равномерные логиты
    let targets = ctx.new_tensor_2d(DType::F32, 4, 2);
    ctx.set_f32(targets, &[1., 0., 0., 0., 0., 1., 0., 0.]); // one-hot
    let l = ctx.cross_entropy_loss(logits, targets);
    let g = build_forward(&ctx, l);
    compute(&mut ctx, &g, 1);
    // loss = -log(1/4) = ln4
    assert!((ctx.data_f32(l)[0] - 4.0f32.ln()).abs() < 1e-5);
}

#[test]
fn cross_entropy_large_uniform_logits_preserve_loss() {
    let mut ctx = Context::new(1024);
    let logits = ctx.new_tensor_2d(DType::F32, 4, 2);
    let targets = ctx.new_tensor_2d(DType::F32, 4, 2);
    ctx.set_f32(targets, &[1., 0., 0., 0., 0., 1., 0., 0.]);
    let loss = ctx.cross_entropy_loss(logits, targets);
    let graph = build_forward(&ctx, loss);
    for offset in [0.0, 1e20, -1e20] {
        ctx.set_f32(logits, &[offset; 8]);
        compute(&mut ctx, &graph, 1);
        assert!((ctx.data_f32(loss)[0] - 4.0f32.ln()).abs() < 1e-6,
            "incorrect loss with logit offset {offset}");
    }
}
