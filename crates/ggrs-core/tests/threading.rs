use ggrs_core::*;

fn build_case(ctx: &mut Context) -> TensorId {
    let k = 96usize;
    let m = 33;
    let n = 47;
    let a = ctx.new_tensor_2d(DType::F32, k, m);
    let b = ctx.new_tensor_2d(DType::F32, k, n);
    let av: Vec<f32> = (0..k * m).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| (i % 23) as f32 * 0.07 - 0.7).collect();
    ctx.set_f32(a, &av);
    ctx.set_f32(b, &bv);
    let d = ctx.mul_mat(a, b);
    let s = ctx.soft_max(d);
    ctx.rms_norm(s, 1e-5)
}

#[test]
fn threads_produce_identical_results() {
    let mut c1 = Context::new(1 << 24);
    let r1 = build_case(&mut c1);
    let g1 = build_forward(&c1, r1);
    compute(&c1, &g1, 1);

    let mut c4 = Context::new(1 << 24);
    let r4 = build_case(&mut c4);
    let g4 = build_forward(&c4, r4);
    compute(&c4, &g4, 4);

    // деление по строкам не меняет порядок редукций → бит-в-бит
    assert_eq!(c1.data_f32(r1), c4.data_f32(r4));
}
