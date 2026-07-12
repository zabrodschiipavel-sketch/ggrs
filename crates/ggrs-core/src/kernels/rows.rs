use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn soft_max(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let mut max = f32::NEG_INFINITY;
            for i in 0..ne0 {
                max = max.max(*pa.add(i));
            }
            let mut sum = 0.0f32;
            for i in 0..ne0 {
                let e = (*pa.add(i) - max).exp();
                *pd.add(i) = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for i in 0..ne0 {
                *pd.add(i) *= inv;
            }
        }
    }
}

pub fn rms_norm(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let eps = f32::from_bits(dst.op_params[0]);
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let mut ss = 0.0f32;
            for i in 0..ne0 {
                let x = *pa.add(i);
                ss += x * x;
            }
            let inv = 1.0 / (ss / ne0 as f32 + eps).sqrt();
            for i in 0..ne0 {
                *pd.add(i) = *pa.add(i) * inv;
            }
        }
    }
}
