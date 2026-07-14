//! ggrs-model: BPE-токенизатор, GPT, тренер и спидран поверх ggrs-core.

pub mod bpe;
pub mod checkpoint;

pub use bpe::Bpe;
pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointExtra};
