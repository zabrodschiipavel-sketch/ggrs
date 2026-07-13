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

/// Обратное распространение SoftMax.
/// На строку: s = Σ_i g[i]*y[i]; dst[i] = y[i] * (g[i] − s).
/// src[0]=g (градиент выхода), src[1]=y (выход softmax).
pub fn soft_max_back(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    let y = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.dtype, DType::F32, "soft_max_back: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "soft_max_back: dst строки должны быть плотными");
    assert_eq!(g.nb[0], g.dtype.type_size(), "soft_max_back: g строки должны быть плотными");
    assert_eq!(y.nb[0], y.dtype.type_size(), "soft_max_back: y строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pg = row_ptr(ctx, g, i1, i2, i3) as *const f32;
            let py = row_ptr(ctx, y, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            // s = Σ_i g[i] * y[i]
            let mut s = 0.0f32;
            for i in 0..ne0 {
                s += *pg.add(i) * *py.add(i);
            }
            // dst[i] = y[i] * (g[i] - s)
            for i in 0..ne0 {
                *pd.add(i) = *py.add(i) * (*pg.add(i) - s);
            }
        }
    }
}

/// Обратное распространение RmsNorm.
/// На строку: ss = Σ x²; r = 1/sqrt(ss/ne0 + eps);
/// dot = Σ_i g[i]*x[i]; dst[j] = r*g[j] − x[j] * r³ * dot / (ne0 as f32).
/// src[0]=g, src[1]=x; op_params[0]=eps.to_bits().
pub fn rms_norm_back(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    let x = ctx.t(dst.src[1].unwrap());
    let eps = f32::from_bits(dst.op_params[0]);
    assert_eq!(dst.dtype, DType::F32, "rms_norm_back: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "rms_norm_back: dst строки должны быть плотными");
    assert_eq!(g.nb[0], g.dtype.type_size(), "rms_norm_back: g строки должны быть плотными");
    assert_eq!(x.nb[0], x.dtype.type_size(), "rms_norm_back: x строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pg = row_ptr(ctx, g, i1, i2, i3) as *const f32;
            let px = row_ptr(ctx, x, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            // ss = Σ x²
            let mut ss = 0.0f32;
            for i in 0..ne0 {
                let xv = *px.add(i);
                ss += xv * xv;
            }
            let r = 1.0 / (ss / ne0 as f32 + eps).sqrt();
            // dot = Σ_i g[i] * x[i]
            let mut dot = 0.0f32;
            for i in 0..ne0 {
                dot += *pg.add(i) * *px.add(i);
            }
            let r3 = r * r * r;
            let factor = r3 * dot / ne0 as f32;
            for j in 0..ne0 {
                let xv = *px.add(j);
                *pd.add(j) = r * *pg.add(j) - xv * factor;
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
