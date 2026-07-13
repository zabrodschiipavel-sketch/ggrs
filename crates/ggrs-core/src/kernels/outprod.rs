use crate::context::Context;
use crate::dtype::DType;
use crate::tensor::TensorId;
use super::{split};

/// OutProd: dst[ix, iy] = Σ_{r=0..R} x[ix, r] * y[iy, r].
/// Параллелизм по строкам dst (iy).
pub fn out_prod(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let x = ctx.t(dst.src[0].unwrap());
    let y = ctx.t(dst.src[1].unwrap());

    assert_eq!(x.dtype, DType::F32, "out_prod: x только F32");
    assert_eq!(y.dtype, DType::F32, "out_prod: y только F32");
    assert_eq!(dst.dtype, DType::F32, "out_prod: dst только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "out_prod: dst строки должны быть плотными");
    assert_eq!(x.nb[0], x.dtype.type_size(), "out_prod: строки x должны быть плотными (сделай cont)");
    assert_eq!(y.nb[0], y.dtype.type_size(), "out_prod: строки y должны быть плотными (сделай cont)");

    let dx = x.ne[0];
    let dy = y.ne[0];
    let r = x.ne[1]; // R = ne[1] обоих, проверено в билдере

    // параллелизм по строкам dst = dy (все строки dst — это ne[1])
    let (iy0, iy1) = split(dy, ith, nth);

    unsafe {
        for iy in iy0..iy1 {
            let pdst = (ctx.base().add(dst.offset + iy * dst.nb[1])) as *mut f32;

            for ix in 0..dx {
                let mut acc = 0.0f32;
                for r0 in 0..r {
                    // элемент x[ix, r0] = *(x_base + r0*x.nb[1] + ix*x.nb[0])
                    let px = ctx.base().add(x.offset + r0 * x.nb[1] + ix * x.nb[0]) as *const f32;
                    // элемент y[iy, r0] = *(y_base + r0*y.nb[1] + iy*y.nb[0])
                    let py = ctx.base().add(y.offset + r0 * y.nb[1] + iy * y.nb[0]) as *const f32;
                    acc += *px * *py;
                }
                *pdst.add(ix) = acc;
            }
        }
    }
}
