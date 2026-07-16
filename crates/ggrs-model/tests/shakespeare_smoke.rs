//! Смоук-тест (Фаза 3, T13): byte-level (BPE vocab=256) обучение GPT на фрагменте
//! Шекспира — сквозная проверка пайплайна BPE -> GPT -> тренер на реальном тексте.
//!
//! Фикстура: `tests/fixtures/shakespeare_64k.txt` — 64 КиБ (обрезано по границе строки)
//! public-domain текста из tinyshakespeare (github.com/karpathy/char-rnn,
//! data/tinyshakespeare/input.txt — компиляция пьес Шекспира, public domain).

use ggrs_core::Context;
use ggrs_model::{build_gpt, train, Bpe, GptConfig, TokenBin, TrainConfig};

/// Клэмп числа потоков доступным параллелизмом — на CI-раннерах (2-4 vCPU)
/// `cap` потоков переподписывают барьерную модель компута и раздувают время смоука.
fn ci_thread_budget(cap: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(cap)
}

fn load_fixture() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/shakespeare_64k.txt"
    ))
    .expect("фикстура tests/fixtures/shakespeare_64k.txt отсутствует")
}

/// Байт-уровневая GPT-конфигурация для этого смоука.
///
/// `GptConfig::tiny()` (из T10) имеет vocab=65 — недостаточно для 256 байтовых
/// токенов, поэтому здесь отдельный маленький конфиг с vocab=256; остальные числа —
/// из исходного плана Фазы 3 для byte-level tiny-смоука.
fn byte_tiny_config() -> GptConfig {
    GptConfig {
        vocab: 256,
        d: 64,
        h: 2,
        layers: 2,
        t: 64,
        d_ff: 128,
        rope_base: 1e4,
        seed: 1,
    }
}

/// Токенизировать фикстуру byte-level BPE (vocab_size=256 -> 0 merges -> 1 токен/байт)
/// и разбить на train/val (последние ~10% токенов - в val).
fn prepare_bins(t: usize) -> (TokenBin, TokenBin) {
    let text = load_fixture();
    let bpe = Bpe::train(&text, 256);
    assert_eq!(
        bpe.vocab_size(),
        256,
        "byte-level: vocab_size должен остаться 256 (0 merges)"
    );
    let tokens: Vec<u16> = bpe.encode(&text);
    assert_eq!(
        tokens.len(),
        text.len(),
        "byte-level: 1 токен должен соответствовать 1 байту"
    );

    let val_len = (tokens.len() / 10).max(t + 8);
    let split = tokens.len() - val_len;
    let bin_train = TokenBin {
        tokens: tokens[..split].to_vec(),
        vocab_size: 256,
    };
    let bin_val = TokenBin {
        tokens: tokens[split..].to_vec(),
        vocab_size: 256,
    };
    (bin_train, bin_val)
}

/// Смоук в CI-бюджете (<5 мин): 200 шагов, grad_accum 4, ожидаем ощутимое падение loss.
#[test]
fn shakespeare_char_smoke() {
    let cfg = byte_tiny_config();
    let (bin_train, bin_val) = prepare_bins(cfg.t);

    let mut ctx = Context::new(1 << 27); // 128 МиБ с запасом
    let gpt = build_gpt(&mut ctx, &cfg);

    let train_cfg = TrainConfig {
        steps: 300,
        total_steps: 300,
        grad_accum: 4,
        lr: 2e-3,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 50,
        eval_windows: 8,
        ckpt_every: 0,
        threads: ci_thread_budget(8),
        out_dir: std::env::temp_dir().join("ggrs_shakespeare_smoke"),
        seed: 1,
    };

    let report = train(&mut ctx, &gpt, &bin_train, &bin_val, &train_cfg, None)
        .expect("train не должен паниковать");

    // Стартовый loss ~ ln(256) ≈ 5.545 (равномерное распределение по 256 байтам).
    let ln256 = 256.0f32.ln();
    assert!(
        report.final_train_loss < 0.55 * ln256,
        "final_train_loss={:.4}, ожидалось < {:.4} (0.55*ln(256))",
        report.final_train_loss,
        0.55 * ln256
    );

    eprintln!(
        "shakespeare_char_smoke: final_train_loss={:.4}, final_val_loss={:.4}, tokens_seen={}, skipped={}",
        report.final_train_loss, report.final_val_loss, report.tokens_seen, report.skipped_steps
    );
}

/// Прогон подлиннее (2000 шагов) — запускается руками (`cargo test -- --ignored`),
/// не в CI. Отчёт с образцами генерации — задача архитектора (T14), здесь не пишем.
#[test]
#[ignore]
fn shakespeare_char_long_run() {
    let cfg = byte_tiny_config();
    let (bin_train, bin_val) = prepare_bins(cfg.t);

    let mut ctx = Context::new(1 << 27);
    let gpt = build_gpt(&mut ctx, &cfg);

    let train_cfg = TrainConfig {
        steps: 2000,
        total_steps: 2000,
        grad_accum: 4,
        lr: 1e-3,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 200,
        eval_windows: 8,
        ckpt_every: 500,
        threads: ci_thread_budget(4),
        out_dir: std::env::temp_dir().join("ggrs_shakespeare_long"),
        seed: 1,
    };

    let report = train(&mut ctx, &gpt, &bin_train, &bin_val, &train_cfg, None)
        .expect("train не должен паниковать");

    eprintln!(
        "shakespeare_char_long_run: final_train_loss={:.4}, final_val_loss={:.4}, tokens_seen={}, skipped={}",
        report.final_train_loss, report.final_val_loss, report.tokens_seen, report.skipped_steps
    );
}
