use crate::context::Context;
use crate::dtype::DType;
use crate::tensor::TensorId;
use super::split;

/// OutProd с батчем: dst[ix, iy, b] = Σ_{r} x[ix, r, b] · y[iy, r, b].
/// Параллелизм по строкам dst = (iy, ib).
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
    let batch = dst.ne[2]; // ne[3]==1 гарантировано билдером

    // строки dst = dy * batch; строка = (iy, ib)
    let nrows = dy * batch;
    let (ir0, ir1) = split(nrows, ith, nth);

    unsafe {
        for ir in ir0..ir1 {
            let ib = ir / dy;
            let iy = ir % dy;
            let pdst = ctx.base().add(dst.offset + iy * dst.nb[1] + ib * dst.nb[2]) as *mut f32;

            for ix in 0..dx {
                let mut acc = 0.0f32;
                for r0 in 0..r {
                    let px = ctx.base().add(x.offset + ix * x.nb[0] + r0 * x.nb[1] + ib * x.nb[2]) as *const f32;
                    let py = ctx.base().add(y.offset + iy * y.nb[0] + r0 * y.nb[1] + ib * y.nb[2]) as *const f32;
                    acc += *px * *py;
                }
                *pdst.add(ix) = acc;
            }
        }
    }
}
