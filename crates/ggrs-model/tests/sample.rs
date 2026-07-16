//! Тесты сэмплера: argmax, top-k, распределение, e2e smoke с tiny-моделью.

use ggrs_core::{compute, util::Lcg};
use ggrs_model::{build_gpt, sample_next, GptConfig};

// ═══════════════════════════════════════════════════════════════════════════
// Тест 1: argmax детерминирован
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn argmax_deterministic() {
    let logits = [0.1f32, 3.0, -1.0, 2.9];
    let mut rng = Lcg::new(7);
    for _ in 0..100 {
        let token = sample_next(&logits, 0.0, 0, &mut rng);
        // argmax: индекс 1 (значение 3.0)
        assert_eq!(token, 1, "argmax вернул {} вместо 1", token);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 2: top_k=1 эквивалентен argmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn top_k_1_equals_argmax() {
    let logits = [0.1f32, 3.0, -1.0, 2.9];
    let mut rng = Lcg::new(7);
    for _ in 0..100 {
        let token = sample_next(&logits, 1.0, 1, &mut rng);
        // top-1: всегда самый большой — индекс 1
        assert_eq!(token, 1, "top_k=1 вернул {} вместо 1", token);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 3: распределение двух токенов
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn distribution_two_tokens() {
    // Логиты [0.0, ln(2.0)] ≈ [0.0, 0.693147]
    // softmax при t=1.0: p0 = e^0/(e^0+e^ln2) = 1/(1+2) = 1/3, p1 = 2/3
    let logits = [0.0f32, std::f32::consts::LN_2];
    let mut rng = Lcg::new(7);
    let n = 10_000;
    let mut count0 = 0usize;
    let mut count1 = 0usize;
    for _ in 0..n {
        let token = sample_next(&logits, 1.0, 0, &mut rng);
        match token {
            0 => count0 += 1,
            1 => count1 += 1,
            _ => panic!("неожиданный токен {}", token),
        }
    }
    let frac0 = count0 as f64 / n as f64;
    let frac1 = count1 as f64 / n as f64;
    // Истинные: 1/3 и 2/3. Допуск ±7% (700 из 10000) → ~ [0.27, 0.40] и [0.60, 0.73]
    assert!(
        (0.27..=0.40).contains(&frac0),
        "frac0={} вне [0.27, 0.40]",
        frac0
    );
    assert!(
        (0.60..=0.73).contains(&frac1),
        "frac1={} вне [0.60, 0.73]",
        frac1
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 4: e2e smoke с tiny-моделью (без обучения)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_generate_smoke() {
    let cfg = GptConfig::tiny();
    let mut ctx = ggrs_core::Context::new(1 << 26); // 64 МиБ
    let gpt = build_gpt(&mut ctx, &cfg);

    // Граф только до логитов (loss/targets не нужны)
    let gf = ggrs_core::build_forward(&ctx, gpt.logits);

    // Промпт-токены [1, 2, 3]
    let mut tokens: Vec<u16> = vec![1, 2, 3];
    let vocab = cfg.vocab;
    let t_ctx = cfg.t;
    let mut rng = Lcg::new(42);

    // 3 итерации генерации
    for _ in 0..3 {
        let window_len = tokens.len().min(t_ctx);
        let window = &tokens[tokens.len() - window_len..];

        let mut buf: Vec<i32> = Vec::with_capacity(t_ctx);
        buf.extend(window.iter().map(|&t| t as i32));
        buf.resize(t_ctx, 0i32);
        ctx.set_i32(gpt.ids, &buf);

        compute(&mut ctx, &gf, 1);

        let p = window_len - 1;
        let logits_data = ctx.data_f32(gpt.logits);
        let slice = &logits_data[p * vocab..(p + 1) * vocab];

        let next = sample_next(slice, 0.8, 40, &mut rng);
        assert!(
            next < vocab,
            "сгенерированный токен {} >= vocab {}",
            next,
            vocab
        );
        tokens.push(next as u16);
    }

    // Проверка: токенов изначально 3 + 3 новых = 6
    assert_eq!(
        tokens.len(),
        6,
        "ожидалось 6 токенов, получено {}",
        tokens.len()
    );
}
