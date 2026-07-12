use ggrs_core::*;

fn ctx1m() -> Context {
    Context::new(1 << 20)
}

#[test]
fn add_mul_scale() {
    let mut ctx = ctx1m();
    let a = ctx.new_tensor_1d(DType::F32, 5);
    let b = ctx.new_tensor_1d(DType::F32, 5);
    ctx.set_f32(a, &[1., 2., 3., 4., 5.]);
    ctx.set_f32(b, &[10., 20., 30., 40., 50.]);
    let s = ctx.add(a, b);
    let m = ctx.mul(s, a);
    let r = ctx.scale(m, 0.5);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    assert_eq!(ctx.data_f32(r), &[5.5, 22.0, 49.5, 88.0, 137.5]);
}

#[test]
fn add_broadcast_rows() {
    // маска [4,1] прибавляется к [4,3] построчно
    let mut ctx = ctx1m();
    let a = ctx.new_tensor_2d(DType::F32, 4, 3);
    let m = ctx.new_tensor_2d(DType::F32, 4, 1);
    ctx.set_f32(a, &[0.; 12]);
    ctx.set_f32(m, &[0., -1., -2., -3.]);
    let r = ctx.add(a, m);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    assert_eq!(ctx.data_f32(r), &[0., -1., -2., -3., 0., -1., -2., -3., 0., -1., -2., -3.]);
}

#[test]
fn silu_gelu_values() {
    let mut ctx = ctx1m();
    let a = ctx.new_tensor_1d(DType::F32, 3);
    ctx.set_f32(a, &[-1.0, 0.0, 2.0]);
    let s = ctx.silu(a);
    let g_ = ctx.gelu(a);
    let graph = build_forward(&ctx, s);
    compute(&ctx, &graph, 1);
    let graph2 = build_forward(&ctx, g_);
    compute(&ctx, &graph2, 1);
    let sv = ctx.data_f32(s);
    // silu(x) = x*sigmoid(x): silu(-1) ≈ -0.26894, silu(0)=0, silu(2) ≈ 1.76159
    assert!((sv[0] + 0.26894143).abs() < 1e-5);
    assert!(sv[1].abs() < 1e-9);
    assert!((sv[2] - 1.7615942).abs() < 1e-5);
    let gv = ctx.data_f32(g_);
    // gelu tanh-аппрокс: gelu(-1) ≈ -0.15881, gelu(0)=0, gelu(2) ≈ 1.95460
    assert!((gv[0] + 0.15880796).abs() < 1e-4);
    assert!((gv[2] - 1.9545977).abs() < 1e-4);
}
