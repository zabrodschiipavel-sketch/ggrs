use ggrs_core::*;

#[test]
fn softmax_rows() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2);
    ctx.set_f32(a, &[1., 2., 3., 1000., 1000., 1000.]);
    let s = ctx.soft_max(a);
    let g = build_forward(&ctx, s);
    compute(&mut ctx, &g, 1);
    let v = ctx.data_f32(s);
    let e = |x: f32| x.exp();
    let z = e(1.) + e(2.) + e(3.);
    assert!((v[0] - e(1.) / z).abs() < 1e-6);
    assert!((v[1] - e(2.) / z).abs() < 1e-6);
    assert!((v[2] - e(3.) / z).abs() < 1e-6);
    // большие значения не дают NaN (стабильность через вычитание max)
    for &x in &v[3..6] {
        assert!((x - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn softmax_fully_masked_row_is_zero_not_nan() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2);
    ctx.set_f32(a, &[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1., 2., 3.]);
    let s = ctx.soft_max(a);
    let g = build_forward(&ctx, s);
    compute(&mut ctx, &g, 1);
    let v = ctx.data_f32(s);
    assert_eq!(&v[0..3], &[0.0, 0.0, 0.0], "замаскированная строка должна дать нули");
    assert!(v[3..6].iter().all(|x| x.is_finite()));
}

#[test]
fn rms_norm_row() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 4, 1);
    ctx.set_f32(a, &[1., 2., 3., 4.]);
    let r = ctx.rms_norm(a, 1e-5);
    let g = build_forward(&ctx, r);
    compute(&mut ctx, &g, 1);
    let ms = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let v = ctx.data_f32(r);
    for (i, &x) in [1.0f32, 2., 3., 4.].iter().enumerate() {
        assert!((v[i] - x * inv).abs() < 1e-6);
    }
}
