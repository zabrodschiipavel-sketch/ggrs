use crate::context::Context;
use crate::tensor::TensorId;
use super::row_ptr;

/// Одним потоком (ith==0), Фаза 1. Численно стабильный log_softmax.
pub fn cross_entropy_loss(ctx: &Context, dst_id: TensorId, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let dst = ctx.t(dst_id);
    let logits = ctx.t(dst.src[0].unwrap());
    let targets = ctx.t(dst.src[1].unwrap());
    assert_eq!(logits.nb[0], 4);
    assert_eq!(targets.nb[0], 4);
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
            let log_z = sum.ln() + max;
            for i in 0..ne0 {
                let t = *pt.add(i);
                if t != 0.0 {
                    total += (t * (*pl.add(i) - log_z)) as f64;
                }
            }
        }
    }
    unsafe {
        let pd = (ctx.base().add(dst.offset)) as *mut f32;
        *pd = -(total / nrows as f64) as f32;
    }
}
