pub mod elementwise;
pub mod mulmat;
pub mod rows;
pub mod copy;
pub mod rope;
pub mod loss;
pub mod outprod;

use crate::context::Context;
use crate::op::Op;
use crate::tensor::{Tensor, TensorId};

pub(crate) fn dispatch(ctx: &Context, id: TensorId, ith: usize, nth: usize) {
    match ctx.t(id).op {
        Op::None | Op::Reshape | Op::Permute | Op::Collect => {}
        Op::Add => elementwise::add(ctx, id, ith, nth),
        Op::Mul => elementwise::mul(ctx, id, ith, nth),
        Op::Scale => elementwise::scale(ctx, id, ith, nth),
        Op::Silu => elementwise::silu(ctx, id, ith, nth),
        Op::Gelu => elementwise::gelu(ctx, id, ith, nth),
        Op::MulMat => mulmat::mul_mat(ctx, id, ith, nth),
        Op::SoftMax => rows::soft_max(ctx, id, ith, nth),
        Op::RmsNorm => rows::rms_norm(ctx, id, ith, nth),
        Op::GetRows => copy::get_rows(ctx, id, ith, nth),
        Op::Cont => copy::cont(ctx, id, ith, nth),
        Op::Rope => rope::rope(ctx, id, ith, nth),
        Op::CrossEntropyLoss => loss::cross_entropy_loss(ctx, id, ith, nth),
        Op::SumAll => rows::sum_all(ctx, id, ith, nth),
        Op::SumAllBack => rows::sum_all_back(ctx, id, ith, nth),
        Op::OutProd => outprod::out_prod(ctx, id, ith, nth),
    }
}

/// Диапазон строк потока ith из nth.
pub(crate) fn split(n: usize, ith: usize, nth: usize) -> (usize, usize) {
    let per = n.div_ceil(nth);
    ((ith * per).min(n), ((ith + 1) * per).min(n))
}

/// Указатель на строку (i1,i2,i3) тензора.
pub(crate) unsafe fn row_ptr(ctx: &Context, t: &Tensor, i1: usize, i2: usize, i3: usize) -> *mut u8 {
    ctx.base().add(t.offset + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3])
}

/// Разложить плоский индекс строки в (i1,i2,i3) по ne тензора.
pub(crate) fn unravel_row(t: &Tensor, ir: usize) -> (usize, usize, usize) {
    let i3 = ir / (t.ne[1] * t.ne[2]);
    let i2 = (ir / t.ne[1]) % t.ne[2];
    let i1 = ir % t.ne[1];
    (i1, i2, i3)
}
