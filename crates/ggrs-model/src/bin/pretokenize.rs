//! pretokenize — пре-токенизация корпуса в бинарный поток токенов (GGTK).
//!
//! Аргументы: <corpus.txt> <vocab.bpe> <out_train.bin> <out_val.bin> [val_frac=0.01]
//!
//! Загружает BPE, токенизирует весь корпус, разделяет train/val,
//! записывает два GGTK-файла.

use ggrs_model::{write_token_bin, Bpe};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <corpus.txt> <vocab.bpe> <out_train.bin> <out_val.bin> [val_frac=0.01]",
            args.first().map(|s| s.as_str()).unwrap_or("pretokenize")
        );
        std::process::exit(2);
    }

    let corpus_path = &args[1];
    let vocab_path = args[2].as_str();
    let out_train = args[3].as_str();
    let out_val = args[4].as_str();
    let val_frac: f64 = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);

    if !(0.0..=1.0).contains(&val_frac) {
        eprintln!("val_frac должен быть в [0, 1], получено {}", val_frac);
        std::process::exit(1);
    }

    let bpe = Bpe::load(Path::new(vocab_path)).unwrap_or_else(|e| {
        eprintln!("Ошибка загрузки BPE '{}': {}", vocab_path, e);
        std::process::exit(1);
    });

    let data = std::fs::read(corpus_path).unwrap_or_else(|e| {
        eprintln!("Ошибка чтения корпуса '{}': {}", corpus_path, e);
        std::process::exit(1);
    });

    let tokens = bpe.encode(&data);
    let n_val = ((tokens.len() as f64) * val_frac) as usize;
    let split = tokens.len() - n_val;

    write_token_bin(Path::new(out_train), bpe.vocab_size(), &tokens[..split])
        .unwrap_or_else(|e| {
            eprintln!("Ошибка записи train '{}': {}", out_train, e);
            std::process::exit(1);
        });

    write_token_bin(Path::new(out_val), bpe.vocab_size(), &tokens[split..])
        .unwrap_or_else(|e| {
            eprintln!("Ошибка записи val '{}': {}", out_val, e);
            std::process::exit(1);
        });

    println!(
        "train: {} токенов, val: {} токенов, vocab_size: {}",
        split,
        n_val,
        bpe.vocab_size()
    );
}
