//! ggrs-model: BPE-токенизатор, GPT, тренер и спидран поверх ggrs-core.

pub mod bpe;
pub mod checkpoint;
pub mod dataset;
pub mod gpt;
pub mod trainer;

pub use bpe::Bpe;
pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointExtra};
pub use dataset::{sample_corpus, val_windows, write_token_bin, TokenBin, WindowSampler};
pub use gpt::{build_gpt, Gpt, GptConfig};
pub use trainer::{train, TrainConfig, TrainReport};
