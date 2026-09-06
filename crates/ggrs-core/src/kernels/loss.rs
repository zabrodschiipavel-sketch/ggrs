use crate::context::Context;
use crate::dtype::DType;
use crate::tensor::TensorId;
use super::{row_ptr, split};

/// Одним потоком (ith==0), Фаза 1. Численно стабильный log_softmax.
pub fn cross_entropy_loss(ctx: &Context, dst_id: TensorId, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let dst = ctx.t(dst_id);
    let logits = ctx.t(dst.src[0].unwrap());
    let targets = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.dtype, DType::F32, "cross_entropy_loss: dst только F32");
    assert_eq!(logits.dtype, DType::F32, "cross_entropy_loss: logits только F32");
    assert_eq!(targets.dtype, DType::F32, "cross_entropy_loss: targets только F32");
    assert_eq!(logits.nb[0], logits.dtype.type_size(), "cross_entropy_loss: logits строки должны быть плотными");
    assert_eq!(targets.nb[0], targets.dtype.type_size(), "cross_entropy_loss: targets строки должны быть плотными");
    let ne0 = logits.ne[0];
    let nrows = logits.nrows();
    let mut total = 0.0f64;
    for ir in 0..nrows {
        let i3 = ir / (logits.ne[1] * logits.ne[2]);
        let i2 = (ir / logits.ne[1]) % logits.ne[2];
        let i1 = ir % logits.ne[1];
        unsafe {
            let pl = row_ptr(ctx, logits, i1, i2, i3) as *const f32;
            let pt = row_ptr(ctx, targets, i1, i2, i3) as *const f32;
            let mut max = f32::NEG_INFINITY;
            for i in 0..ne0 {
                max = max.max(*pl.add(i));
            }
            let mut sum = 0.0f32;
            for i in 0..ne0 {
                sum += (*pl.add(i) - max).exp();
            }
            // Keep the subtraction in shifted coordinates: adding max first
            // loses ln(sum) entirely for large logits, producing a zero loss.
            let log_sum = sum.ln();
            for i in 0..ne0 {
                let t = *pt.add(i);
                if t != 0.0 {
                    total += (t * ((*pl.add(i) - max) - log_sum)) as f64;
                }
            }
        }
    }
    unsafe {
        let pd = (ctx.base().add(dst.offset)) as *mut f32;
        *pd = -(total / nrows as f64) as f32;
    }
}

/// Обратное распространение CrossEntropyLoss.
/// Параллелизм по строкам.
/// Для каждой строки: max → sum exp → softmax → dst = (softmax − t) * g0 / nrows.
pub fn cross_entropy_loss_back(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    let logits = ctx.t(dst.src[1].unwrap());
    let targets = ctx.t(dst.src[2].unwrap());

    assert_eq!(dst.dtype, DType::F32, "cross_entropy_loss_back: dst только F32");
    assert_eq!(g.dtype, DType::F32, "cross_entropy_loss_back: g только F32");
    assert_eq!(logits.dtype, DType::F32, "cross_entropy_loss_back: logits только F32");
    assert_eq!(targets.dtype, DType::F32, "cross_entropy_loss_back: targets только F32");
    assert_eq!(logits.nb[0], logits.dtype.type_size(), "cross_entropy_loss_back: logits строки должны быть плотными");
    assert_eq!(targets.nb[0], targets.dtype.type_size(), "cross_entropy_loss_back: targets строки должны быть плотными");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "cross_entropy_loss_back: dst строки должны быть плотными");

    let ne0 = logits.ne[0];
    let nrows = logits.nrows();
    let (row_start, row_end) = split(nrows, ith, nth);

    // Читаем градиент лосса g0
    let g0 = unsafe { *(ctx.base().add(g.offset) as *const f32) };
    let inv_nrows = 1.0 / nrows as f32;

    for ir in row_start..row_end {
        let i3 = ir / (logits.ne[1] * logits.ne[2]);
        let i2 = (ir / logits.ne[1]) % logits.ne[2];
        let i1 = ir % logits.ne[1];
        unsafe {
            let pl = row_ptr(ctx, logits, i1, i2, i3) as *const f32;
            let pt = row_ptr(ctx, targets, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;

            // max по строке для численной стабильности
            let mut max = f32::NEG_INFINITY;
            for i in 0..ne0 {
                max = max.max(*pl.add(i));
            }

            // sum exp(logits - max)
            let mut sum = 0.0f32;
            for i in 0..ne0 {
                sum += (*pl.add(i) - max).exp();
            }
            let inv_sum = 1.0 / sum;

            // dst = (softmax − t) * g0 / nrows
            for i in 0..ne0 {
                let softmax_i = ((*pl.add(i) - max).exp()) * inv_sum;
                let t_i = *pt.add(i);
                *pd.add(i) = (softmax_i - t_i) * g0 * inv_nrows;
            }
        }
    }
}
