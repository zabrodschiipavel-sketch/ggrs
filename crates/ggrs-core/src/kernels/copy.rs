use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn get_rows(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let ids = ctx.data_i32(dst.src[1].unwrap());
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.ne[1], ith, nth);
    for (ir, &id) in ids.iter().enumerate().take(ir1).skip(ir0) {
        let row = id as usize;
        assert!(row < a.ne[1], "get_rows: индекс {row} вне диапазона");
        unsafe {
            let ps = row_ptr(ctx, a, row, 0, 0) as *const f32;
            let pd = row_ptr(ctx, dst, ir, 0, 0) as *mut f32;
            std::ptr::copy_nonoverlapping(ps, pd, ne0);
        }
    }
}

pub fn cont(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let base = ctx.base().add(a.offset + i1 * a.nb[1] + i2 * a.nb[2] + i3 * a.nb[3]);
            for i0 in 0..ne0 {
                *pd.add(i0) = *(base.add(i0 * a.nb[0]) as *const f32);
            }
        }
    }
}
