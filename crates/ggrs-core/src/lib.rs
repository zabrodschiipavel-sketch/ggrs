//! ggrs-core: порт ядра ggml на Rust — тензоры, граф, CPU-ядра.

pub mod context;
pub mod dtype;
pub mod op;
pub mod tensor;

pub use context::Context;
pub use dtype::DType;
pub use op::Op;
pub use tensor::{Tensor, TensorId, MAX_DIMS, MAX_SRC};
