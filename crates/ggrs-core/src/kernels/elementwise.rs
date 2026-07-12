use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

/// Общий цикл бинарной операции с broadcast src1 (ne==ne или 1 по каждому измерению).
fn binary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, f: impl Fn(f32, f32) -> f32) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let b = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.nb[0], 4, "binary: dst строки должны быть плотными");
    assert_eq!(a.nb[0], 4, "binary: src0 строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pb = row_ptr(ctx, b, i1 % b.ne[1], i2 % b.ne[2], i3 % b.ne[3]) as *const f32;
            if b.ne[0] == ne0 {
                assert_eq!(b.nb[0], 4);
                for i in 0..ne0 {
                    *pd.add(i) = f(*pa.add(i), *pb.add(i));
                }
            } else {
                assert_eq!(b.ne[0], 1, "binary: broadcast только ne0==1");
                let s = *pb;
                for i in 0..ne0 {
                    *pd.add(i) = f(*pa.add(i), s);
                }
            }
        }
    }
}

fn unary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, f: impl Fn(f32) -> f32) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(dst.nb[0], 4);
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            for i in 0..ne0 {
                *pd.add(i) = f(*pa.add(i));
            }
        }
    }
}

pub fn add(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, |x, y| x + y);
}
pub fn mul(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, |x, y| x * y);
}
pub fn scale(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    let s = f32::from_bits(ctx.t(dst).op_params[0]);
    unary(ctx, dst, ith, nth, |x| x * s);
}
/// silu(x) = x·σ(x) = x/(1+e^(−x))
pub fn silu(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    unary(ctx, dst, ith, nth, |x| x / (1.0 + (-x).exp()));
}
pub fn gelu(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    unary(ctx, dst, ith, nth, |x| {
        0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
    });
}
