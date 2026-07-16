//! Бинарник генерации текста из чекпоинта: загрузка BPE + модели, итеративный forward.
//!
//! Аргументы:
//!   --ckpt <path>  --vocab <path.bpe>  --prompt <строка>
//!   [--n N=200]  [--temp T=0.8]  [--top-k K=40]
//!   [--config d10m|tiny=d10m]  [--seed S=42]  [--threads N=4]

use std::path::PathBuf;

use ggrs_core::{compute, util::Lcg};
use ggrs_model::{
    build_gpt, load_checkpoint, sample_next, Bpe, GptConfig,
};

fn usage() -> ! {
    eprintln!(
        "Usage: generate --ckpt <path> --vocab <path.bpe> --prompt <строка> \
         [--n N] [--temp T] [--top-k K] [--config d10m|tiny] [--seed S] [--threads N]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut iter = args.iter().skip(1).peekable();

    let mut ckpt: Option<PathBuf> = None;
    let mut vocab: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut n_tokens: usize = 200;
    let mut temperature: f32 = 0.8;
    let mut top_k: usize = 40;
    let mut config_name: String = "d10m".to_string();
    let mut seed: u64 = 42;
    let mut threads: usize = 4;

    while let Some(key) = iter.next() {
        match key.as_str() {
            "--ckpt" => {
                ckpt = Some(PathBuf::from(iter.next().unwrap_or_else(|| usage())));
            }
            "--vocab" => {
                vocab = Some(PathBuf::from(iter.next().unwrap_or_else(|| usage())));
            }
            "--prompt" => {
                prompt = Some(iter.next().unwrap_or_else(|| usage()).clone());
            }
            "--n" => {
                let s = iter.next().unwrap_or_else(|| usage());
                n_tokens = s.parse().expect("--n: неверное число");
            }
            "--temp" => {
                let s = iter.next().unwrap_or_else(|| usage());
                temperature = s.parse().expect("--temp: неверное число");
            }
            "--top-k" => {
                let s = iter.next().unwrap_or_else(|| usage());
                top_k = s.parse().expect("--top-k: неверное число");
            }
            "--config" => {
                config_name = iter.next().unwrap_or_else(|| usage()).clone();
            }
            "--seed" => {
                let s = iter.next().unwrap_or_else(|| usage());
                seed = s.parse().expect("--seed: неверное число");
            }
            "--threads" => {
                let s = iter.next().unwrap_or_else(|| usage());
                threads = s.parse().expect("--threads: неверное число");
            }
            _ => {
                eprintln!("Неизвестный аргумент: {}", key);
                usage();
            }
        }
    }

    let ckpt_path = ckpt.unwrap_or_else(|| usage());
    let vocab_path = vocab.unwrap_or_else(|| usage());
    let prompt_str = prompt.unwrap_or_else(|| usage());

    // Загрузка BPE
    let bpe = Bpe::load(&vocab_path).expect("Не удалось загрузить BPE-словарь");

    // Конфиг модели
    let cfg = match config_name.as_str() {
        "d10m" => GptConfig::d10m(),
        "tiny" => GptConfig::tiny(),
        _ => {
            eprintln!(
                "Неизвестный конфиг: '{}'. Допустимые: d10m, tiny",
                config_name
            );
            std::process::exit(1);
        }
    };

    // Размер арены: для d10m — 1<<30, для tiny — 1<<26
    let arena_size: usize = match config_name.as_str() {
        "d10m" => 1 << 30,
        "tiny" => 1 << 26,
        _ => 1 << 26,
    };
    let mut ctx = ggrs_core::Context::new(arena_size);

    // Построение модели
    let gpt = build_gpt(&mut ctx, &cfg);
    let named: Vec<(&str, _)> = gpt
        .params
        .iter()
        .map(|(name, id)| (name.as_str(), *id))
        .collect();

    // Загрузка чекпоинта
    let result = load_checkpoint(&ckpt_path, &mut ctx, &named);
    match result {
        Ok(_extra) => {}
        Err(e) => {
            // Популярная ошибка: несовпадение форм → не тот конфиг
            let msg = format!("{}", e);
            if msg.contains("форма") || msg.contains("размер") || msg.contains("не совпадает") {
                eprintln!(
                    "Ошибка: чекпоинт не от этого конфига, укажи --config. \
                     Подробнее: {}",
                    e
                );
            } else {
                eprintln!("Ошибка загрузки чекпоинта: {}", e);
            }
            std::process::exit(1);
        }
    }

    // ГРАФ ТОЛЬКО ДО ЛОГИТОВ (loss/targets не нужны)
    let gf = ggrs_core::build_forward(&ctx, gpt.logits);

    // Кодируем промпт
    let mut tokens: Vec<u16> = bpe.encode(prompt_str.as_bytes());
    assert!(!tokens.is_empty(), "prompt не должен быть пустым");

    let vocab = cfg.vocab;
    let t_ctx = cfg.t;
    let mut rng = Lcg::new(seed);

    // Цикл генерации
    for _ in 0..n_tokens {
        let window_len = tokens.len().min(t_ctx);
        let window = &tokens[tokens.len() - window_len..];

        // ids: [окно, добивка нулями до t_ctx]
        let mut buf: Vec<i32> = Vec::with_capacity(t_ctx);
        buf.extend(window.iter().map(|&t| t as i32));
        buf.resize(t_ctx, 0i32);
        ctx.set_i32(gpt.ids, &buf);

        // Forward
        compute(&mut ctx, &gf, threads);

        // Последняя реальная позиция p = window_len - 1
        let p = window_len - 1;
        // Логиты: ne = [vocab, t] → элемент (v, pos) лежит по индексу pos*vocab + v
        let logits_data = ctx.data_f32(gpt.logits);
        let slice = &logits_data[p * vocab..(p + 1) * vocab];

        let next = sample_next(slice, temperature, top_k, &mut rng);
        tokens.push(next as u16);
    }

    // Декодируем и выводим
    let output = bpe.decode(&tokens);
    // Печатаем как UTF-8 (lossy для некорректных байтов)
    print!("{}", String::from_utf8_lossy(&output));
}
