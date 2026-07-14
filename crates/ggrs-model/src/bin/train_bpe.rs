//! train_bpe — обучение BPE на сэмпле корпуса.
//!
//! Аргументы: <corpus.txt> <vocab.bpe> [vocab_size=4096] [sample_mib=16]
//!
//! Читает corpus.txt, сэмплирует sample_corpus с sample_mib, обучает BPE
//! vocab_size слияний, сохраняет в vocab.bpe. Печатает итоговый vocab_size
//! и первые 20 слияний (декодированных в lossy utf8).

use ggrs_model::{sample_corpus, Bpe};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <corpus.txt> <vocab.bpe> [vocab_size=4096] [sample_mib=16]",
            args.first().map(|s| s.as_str()).unwrap_or("train_bpe")
        );
        std::process::exit(2);
    }

    let corpus_path = &args[1];
    let vocab_path = args[2].as_str();
    let vocab_size: u32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let sample_mib: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let data = std::fs::read(corpus_path).unwrap_or_else(|e| {
        eprintln!("Ошибка чтения корпуса '{}': {}", corpus_path, e);
        std::process::exit(1);
    });

    let sample = sample_corpus(&data, sample_mib);
    let bpe = Bpe::train(&sample, vocab_size);
    let final_vocab = bpe.vocab_size();

    bpe.save(Path::new(vocab_path)).unwrap_or_else(|e| {
        eprintln!("Ошибка сохранения BPE '{}': {}", vocab_path, e);
        std::process::exit(1);
    });

    println!("vocab_size = {} (запрошено {})", final_vocab, vocab_size);

    // Печать первых 20 слияний: декодируем каждый новый токен (256+i) в lossy utf8.
    let n_display = final_vocab.saturating_sub(256).min(20) as usize;
    println!("Первые {} слияний:", n_display);
    for i in 0..n_display {
        let token_id = 256u16 + i as u16;
        let bytes = bpe.decode(&[token_id]);
        let s = String::from_utf8_lossy(&bytes);
        println!("  токен {} (слияние {}): {:?}", token_id, i, s);
    }

    if final_vocab < 256 + n_display as u32 {
        println!("(слияний меньше 20 — корпус мал или достигнуто насыщение)");
    }
}
