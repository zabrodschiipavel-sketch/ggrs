//! Бинарник тренировки GPT: загрузка GGTK-датасетов, построение модели, тренировка.
//!
//! Аргументы:
//!   --data-train <path>  --data-val <path>  --out <dir>
//!   [--resume <ckpt>]  [--steps N=1000]  [--threads N=4]
//!   [--config d10m|tiny=d10m]  [--lr X=3e-4]  [--grad-accum G=4]  [--seed S=1]

use std::path::PathBuf;

use ggrs_model::{
    build_gpt, train, GptConfig, TokenBin, TrainConfig,
};

fn usage() -> ! {
    eprintln!(
        "Usage: train --data-train <path> --data-val <path> --out <dir> \
         [--resume <ckpt>] [--steps N] [--threads N] \
         [--config d10m|tiny] [--lr X] [--grad-accum G] [--seed S]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut iter = args.iter().skip(1).peekable();

    let mut data_train: Option<PathBuf> = None;
    let mut data_val: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut resume: Option<PathBuf> = None;
    let mut steps: u64 = 1000;
    let mut threads: usize = 4;
    let mut config_name: String = "d10m".to_string();
    let mut lr: f32 = 3e-4;
    let mut grad_accum: u32 = 4;
    let mut seed: u64 = 1;

    while let Some(key) = iter.next() {
        match key.as_str() {
            "--data-train" => {
                data_train = Some(PathBuf::from(
                    iter.next().unwrap_or_else(|| usage()),
                ));
            }
            "--data-val" => {
                data_val = Some(PathBuf::from(
                    iter.next().unwrap_or_else(|| usage()),
                ));
            }
            "--out" => {
                out_dir = Some(PathBuf::from(
                    iter.next().unwrap_or_else(|| usage()),
                ));
            }
            "--resume" => {
                resume = Some(PathBuf::from(
                    iter.next().unwrap_or_else(|| usage()),
                ));
            }
            "--steps" => {
                let s = iter.next().unwrap_or_else(|| usage());
                steps = s.parse().expect("--steps: неверное число");
            }
            "--threads" => {
                let s = iter.next().unwrap_or_else(|| usage());
                threads = s.parse().expect("--threads: неверное число");
            }
            "--config" => {
                config_name = iter.next().unwrap_or_else(|| usage()).clone();
            }
            "--lr" => {
                let s = iter.next().unwrap_or_else(|| usage());
                lr = s.parse().expect("--lr: неверное число");
            }
            "--grad-accum" => {
                let s = iter.next().unwrap_or_else(|| usage());
                grad_accum = s.parse().expect("--grad-accum: неверное число");
            }
            "--seed" => {
                let s = iter.next().unwrap_or_else(|| usage());
                seed = s.parse().expect("--seed: неверное число");
            }
            _ => {
                eprintln!("Неизвестный аргумент: {}", key);
                usage();
            }
        }
    }

    let data_train = data_train.unwrap_or_else(|| usage());
    let data_val = data_val.unwrap_or_else(|| usage());
    let out_dir = out_dir.unwrap_or_else(|| usage());

    // Загрузка датасетов
    let bin_train = TokenBin::load(&data_train).expect("Не удалось загрузить train-датасет");
    let bin_val = TokenBin::load(&data_val).expect("Не удалось загрузить val-датасет");

    // Конфиг модели
    let cfg = match config_name.as_str() {
        "d10m" => GptConfig::d10m(),
        "tiny" => GptConfig::tiny(),
        _ => {
            eprintln!("Неизвестный конфиг: '{}'. Допустимые: d10m, tiny", config_name);
            std::process::exit(1);
        }
    };

    // Проверка vocab
    let vocab_size_bins = bin_train.vocab_size.max(bin_val.vocab_size);
    if (cfg.vocab as u32) < vocab_size_bins {
        eprintln!(
            "Ошибка: vocab модели ({}) меньше vocab_size датасетов (train={}, val={})",
            cfg.vocab, bin_train.vocab_size, bin_val.vocab_size
        );
        std::process::exit(1);
    }

    // Контекст: для d10m — 1<<30, для tiny — 1<<26
    let arena_size = match config_name.as_str() {
        "d10m" => 1 << 30,
        "tiny" => 1 << 26,
        _ => 1 << 26,
    };
    let mut ctx = ggrs_core::Context::new(arena_size);

    // Построение модели
    let gpt = build_gpt(&mut ctx, &cfg);

    // TrainConfig: total_steps = steps (обычный запуск — одна цель)
    let train_cfg = TrainConfig {
        steps,
        total_steps: steps,
        grad_accum,
        lr,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 50,
        eval_windows: 8,
        ckpt_every: 200,
        threads,
        out_dir,
        seed,
    };

    // Тренировка
    let report = train(
        &mut ctx,
        &gpt,
        &bin_train,
        &bin_val,
        &train_cfg,
        resume.as_deref(),
    )
    .expect("train: ошибка");

    println!(
        "Train report: final_train_loss={:.6}, final_val_loss={:.6}, \
         tokens_seen={}, skipped_steps={}",
        report.final_train_loss,
        report.final_val_loss,
        report.tokens_seen,
        report.skipped_steps,
    );
}
