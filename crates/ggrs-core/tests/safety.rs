use ggrs_core::{Context, DType};

#[test]
#[should_panic(expected = "tensor element count overflow")]
fn reshape_rejects_wrapped_element_count() {
    let mut ctx = Context::new(64);
    let a = ctx.new_tensor_1d(DType::F32, 2);
    // In release the unchecked product wraps to 2, matching the source.
    ctx.reshape_3d(a, 1, usize::MAX / 2 + 2, 2);
}

#[test]
#[should_panic(expected = "tensor dimensions must be nonzero")]
fn empty_tensor_is_rejected_before_building_strided_views() {
    let mut ctx = Context::new(64);
    ctx.new_tensor_2d(DType::F32, 2, 0);
}

#[test]
fn binary_ops_reject_non_f32_broadcast_operands() {
    for dtype in [DType::F16, DType::I32] {
        for multiply in [false, true] {
            let mut ctx = Context::new(128);
            let a = ctx.new_tensor_1d(DType::F32, 2);
            let b = ctx.new_tensor_1d(dtype, 1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if multiply {
                    ctx.mul(a, b)
                } else {
                    ctx.add(a, b)
                }
            }));
            assert!(result.is_err(), "accepted {dtype:?} as a float operand");
        }
    }
}

#[test]
#[should_panic(expected = "unary_op: only F32")]
fn scale_rejects_integer_storage() {
    let mut ctx = Context::new(64);
    let a = ctx.new_tensor_1d(DType::I32, 1);
    ctx.scale(a, 2.0);
}

#[test]
#[should_panic(expected = "sum_all_back: gradient must be scalar")]
fn sum_all_back_rejects_non_scalar_gradient() {
    let mut ctx = Context::new(128);
    let g = ctx.new_tensor_1d(DType::F32, 2);
    let a = ctx.new_tensor_1d(DType::F32, 3);
    ctx.sum_all_back(g, a);
}
