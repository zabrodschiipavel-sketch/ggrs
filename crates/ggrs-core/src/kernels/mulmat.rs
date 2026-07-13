use crate::context::Context;
use crate::dtype::DType;
use crate::simd;
use crate::tensor::TensorId;
use super::{row_ptr, split};

pub fn mul_mat(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let b = ctx.t(dst.src[1].unwrap());

    assert_eq!(a.dtype, DType::F32, "mul_mat: a только F32");
    assert_eq!(b.dtype, DType::F32, "mul_mat: b только F32");
    assert_eq!(dst.dtype, DType::F32, "mul_mat: только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "mul_mat: dst строки должны быть плотными");
    assert_eq!(a.nb[0], a.dtype.type_size(), "mul_mat: строки a должны быть плотными (сделай cont)");
    assert_eq!(b.nb[0], b.dtype.type_size(), "mul_mat: строки b должны быть плотными (сделай cont)");

    let k = a.ne[0];
    let m = a.ne[1];

    // работа: все строки dst = b.ne1 * b.ne2 * b.ne3; делим их между потоками
    let nr = dst.ne[1] * dst.ne[2] * dst.ne[3];
    let (ir0, ir1) = split(nr, ith, nth);

    for ir in ir0..ir1 {
        let i3 = ir / (dst.ne[1] * dst.ne[2]);
        let i2 = (ir / dst.ne[1]) % dst.ne[2];
        let i1 = ir % dst.ne[1];

        let a2 = i2 % a.ne[2];
        let a3 = i3 % a.ne[3];

        unsafe {
            let pb = row_ptr(ctx, b, i1, i2, i3) as *const f32;
            let brow = std::slice::from_raw_parts(pb, k);
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;

            for i0 in 0..m {
                let pa = (ctx.base().add(a.offset + i0 * a.nb[1] + a2 * a.nb[2] + a3 * a.nb[3])) as *const f32;
                let arow = std::slice::from_raw_parts(pa, k);
                *pd.add(i0) = simd::vec_dot(arow, brow);
            }
        }
    }
}
