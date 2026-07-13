use crate::dtype::DType;

pub fn rope(ctx: &crate::context::Context, dst_id: crate::tensor::TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let pos = ctx.data_i32(dst.src[1].unwrap());
    let n_dims = dst.op_params[0] as usize;
    let base = f32::from_bits(dst.op_params[1]);
    // op_params[2]: 0 = forward, 1 = backward (транспонированное вращение)
    let backward = dst.op_params[2] != 0;
    assert_eq!(dst.dtype, DType::F32, "rope: только F32");
    assert_eq!(a.nb[0], a.dtype.type_size(), "rope: src0 строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = super::split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = super::unravel_row(dst, ir);
        let p = pos[i2] as f32;
        unsafe {
            let pa = super::row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pd = super::row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            for i in 0..n_dims / 2 {
                let theta = p * base.powf(-2.0 * i as f32 / n_dims as f32);
                let (sin_t, cos_t) = theta.sin_cos();
                // backward: используем -sin_t (транспонированное вращение)
                let sin_t = if backward { -sin_t } else { sin_t };
                let x0 = *pa.add(2 * i);
                let x1 = *pa.add(2 * i + 1);
                *pd.add(2 * i) = x0 * cos_t - x1 * sin_t;
                *pd.add(2 * i + 1) = x0 * sin_t + x1 * cos_t;
            }
            for i in n_dims..ne0 {
                *pd.add(i) = *pa.add(i);
            }
        }
    }
}
