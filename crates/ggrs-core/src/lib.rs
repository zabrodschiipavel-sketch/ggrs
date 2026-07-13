//! ggrs-core: порт ядра ggml на Rust — тензоры, граф, CPU-ядра.

pub mod context;
pub mod dtype;
pub mod op;
pub mod tensor;
pub mod graph;
pub mod compute;
pub mod simd;
pub mod util;
pub mod backward;
pub mod optim;
mod kernels;

pub use context::Context;
pub use dtype::DType;
pub use op::Op;
pub use tensor::{Tensor, TensorId, MAX_DIMS, MAX_SRC};
pub use graph::{Graph, build_forward};
pub use compute::compute;
pub use compute::compute_profiled;
pub use backward::{Backward, build_backward};
pub use optim::{AdamW, LrSchedule};
