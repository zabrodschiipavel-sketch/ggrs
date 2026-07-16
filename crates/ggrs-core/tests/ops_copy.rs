use ggrs_core::*;

#[test]
fn get_rows_basic() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 2, 3); // 3 строки по 2
    ctx.set_f32(a, &[0., 1., 10., 11., 20., 21.]);
    let ids = ctx.new_tensor_1d(DType::I32, 2);
    ctx.set_i32(ids, &[2, 0]);
    let r = ctx.get_rows(a, ids);
    let g = build_forward(&ctx, r);
    compute(&mut ctx, &g, 1);
    assert_eq!(ctx.data_f32(r), &[20., 21., 0., 1.]);
}

#[test]
fn cont_materializes_transpose() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2); // [[0,1,2],[3,4,5]]
    ctx.set_f32(a, &[0., 1., 2., 3., 4., 5.]);
    let t = ctx.transpose(a); // логически [[0,3],[1,4],[2,5]]
    let c = ctx.cont(t);
    let g = build_forward(&ctx, c);
    compute(&mut ctx, &g, 1);
    assert!(ctx.t(c).is_contiguous());
    assert_eq!(ctx.data_f32(c), &[0., 3., 1., 4., 2., 5.]);
}
