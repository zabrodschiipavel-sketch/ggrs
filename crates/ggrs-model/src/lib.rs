//! ggrs-model: BPE-токенизатор, GPT, тренер и спидран поверх ggrs-core.

pub mod checkpoint;

pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointExtra};
