use ggrs_core::*;

#[test]
fn mulmat_2x3_times_4x3() {
    // a: k=3, m=2 строки; b: k=3, n=4 строки; dst: [m=2, n=4]
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2);
    let b = ctx.new_tensor_2d(DType::F32, 3, 4);
    ctx.set_f32(a, &[1., 2., 3., 4., 5., 6.]);
    ctx.set_f32(b, &[1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.]);
    let d = ctx.mul_mat(a, b);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);
    // dst[i0=строка a, i1=строка b]: dot(a_i0, b_i1)
    assert_eq!(ctx.data_f32(d), &[1., 4., 2., 5., 3., 6., 6., 15.]);
    assert_eq!(ctx.t(d).ne, [2, 4, 1, 1]);
}

#[test]
fn mulmat_matches_naive_random() {
    let mut ctx = Context::new(1 << 24);
    let (k, m, n) = (64usize, 17, 23);
    let a = ctx.new_tensor_2d(DType::F32, k, m);
    let b = ctx.new_tensor_2d(DType::F32, k, n);
    let av: Vec<f32> = (0..k * m).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| ((i * 40503usize + 7) % 1000) as f32 / 500.0 - 1.0).collect();
    ctx.set_f32(a, &av);
    ctx.set_f32(b, &bv);
    let d = ctx.mul_mat(a, b);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);
    let out = ctx.data_f32(d);
    for i1 in 0..n {
        for i0 in 0..m {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += av[i0 * k + kk] as f64 * bv[i1 * k + kk] as f64;
            }
            let got = out[i1 * m + i0];
            assert!((got as f64 - acc).abs() < 1e-3, "({i0},{i1}): {got} vs {acc}");
        }
    }
}
