use crate::context::Context;
use crate::dtype::DType;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn soft_max(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(dst.dtype, DType::F32, "soft_max: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "soft_max: dst строки должны быть плотными");
    assert_eq!(a.nb[0], a.dtype.type_size(), "soft_max: src0 строки должны быть плотными");
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
            // F3: замаскированная строка — все -inf, даёт NaN
            if max == f32::NEG_INFINITY {
                for i in 0..ne0 {
                    *pd.add(i) = 0.0;
                }
                continue;
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
    assert_eq!(dst.dtype, DType::F32, "rms_norm: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "rms_norm: dst строки должны быть плотными");
    assert_eq!(a.nb[0], a.dtype.type_size(), "rms_norm: src0 строки должны быть плотными");
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

/// Сумма всех элементов src в dst[0] (скаляр). Только ith==0, F32 contiguous.
pub fn sum_all(ctx: &Context, dst_id: TensorId, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(dst.dtype, DType::F32, "sum_all: dst только F32");
    assert_eq!(a.dtype, DType::F32, "sum_all: src только F32");
    assert!(a.is_contiguous(), "sum_all: src обязан быть contiguous");
    let data = unsafe {
        std::slice::from_raw_parts(
            ctx.base().add(a.offset) as *const f32,
            a.nelements(),
        )
    };
    let sum: f32 = data.iter().sum();
    unsafe {
        let pd = ctx.base().add(dst.offset) as *mut f32;
        *pd = sum;
    }
}

/// Заполнить dst формы like значением g[0] (обратное распространение SumAll).
/// g — тензор [1] с градиентом. Каждый поток заполняет свои строки.
pub fn sum_all_back(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    assert_eq!(dst.dtype, DType::F32, "sum_all_back: dst только F32");
    assert_eq!(g.dtype, DType::F32, "sum_all_back: g только F32");
    assert!(dst.is_contiguous(), "sum_all_back: dst обязан быть contiguous");
    let g_val = unsafe {
        *(ctx.base().add(g.offset) as *const f32)
    };
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            for i in 0..ne0 {
                *pd.add(i) = g_val;
            }
        }
    }
}
