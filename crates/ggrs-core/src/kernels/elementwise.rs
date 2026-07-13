use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

#[derive(Copy, Clone)]
enum BinKind { Add, Mul }

fn binary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, kind: BinKind) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let b = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.dtype, crate::dtype::DType::F32, "binary: только F32");
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
                let sa = std::slice::from_raw_parts(pa, ne0);
                let sb = std::slice::from_raw_parts(pb, ne0);
                let sd = std::slice::from_raw_parts_mut(pd, ne0);
                match kind {
                    BinKind::Add => crate::simd::vec_add(sa, sb, sd),
                    BinKind::Mul => crate::simd::vec_mul(sa, sb, sd),
                }
            } else {
                assert_eq!(b.ne[0], 1, "binary: broadcast только ne0==1");
                let s = *pb;
                match kind {
                    BinKind::Add => for i in 0..ne0 { *pd.add(i) = *pa.add(i) + s; },
                    BinKind::Mul => for i in 0..ne0 { *pd.add(i) = *pa.add(i) * s; },
                }
            }
        }
    }
}

fn unary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, f: impl Fn(f32) -> f32) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(dst.dtype, crate::dtype::DType::F32, "unary: только F32");
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

/// Ядро обратного распространения Silu: dst = g * silu'(x).
/// g и x — плотные F32-строки одинаковой формы (БЕЗ broadcast).
pub fn silu_back(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    let x = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.dtype, crate::dtype::DType::F32, "silu_back: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "silu_back: dst строки должны быть плотными");
    assert_eq!(g.nb[0], g.dtype.type_size(), "silu_back: g строки должны быть плотными");
    assert_eq!(x.nb[0], x.dtype.type_size(), "silu_back: x строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let pg = row_ptr(ctx, g, i1, i2, i3) as *const f32;
            let px = row_ptr(ctx, x, i1, i2, i3) as *const f32;
            for i in 0..ne0 {
                let xv = *px.add(i);
                // σ(x) = 1/(1+exp(-x))
                let sig = 1.0 / (1.0 + (-xv).exp());
                // silu'(x) = σ(x) * (1 + x * (1 - σ(x)))
                let deriv = sig * (1.0 + xv * (1.0 - sig));
                *pd.add(i) = *pg.add(i) * deriv;
            }
        }
    }
}

/// Ядро обратного распространения Gelu (tanh-аппроксимация): dst = g * gelu'(x).
/// g и x — плотные F32-строки одинаковой формы (БЕЗ broadcast).
pub fn gelu_back(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    const C: f32 = 0.797_884_6; // √(2/π)
    const B: f32 = 0.044715;
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    let x = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.dtype, crate::dtype::DType::F32, "gelu_back: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "gelu_back: dst строки должны быть плотными");
    assert_eq!(g.nb[0], g.dtype.type_size(), "gelu_back: g строки должны быть плотными");
    assert_eq!(x.nb[0], x.dtype.type_size(), "gelu_back: x строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let pg = row_ptr(ctx, g, i1, i2, i3) as *const f32;
            let px = row_ptr(ctx, x, i1, i2, i3) as *const f32;
            for i in 0..ne0 {
                let xv = *px.add(i);
                // u = C*(x + B*x³)
                let u = C * (xv + B * xv * xv * xv);
                let t = u.tanh();
                // gelu'(x) = 0.5*(1+t) + 0.5*x*(1-t*t)*C*(1+3*B*x*x)
                let deriv = 0.5 * (1.0 + t)
                    + 0.5 * xv * (1.0 - t * t) * C * (1.0 + 3.0 * B * xv * xv);
                *pd.add(i) = *pg.add(i) * deriv;
            }
        }
    }
}

pub fn add(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, BinKind::Add);
}
pub fn mul(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, BinKind::Mul);
}
pub fn scale(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    let s = f32::from_bits(ctx.t(dst).op_params[0]);
    let dst = ctx.t(dst);
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
            std::ptr::copy_nonoverlapping(pa, pd, ne0);
            let sd = std::slice::from_raw_parts_mut(pd, ne0);
            crate::simd::vec_scale(sd, s);
        }
    }
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
