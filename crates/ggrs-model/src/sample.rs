//! Сэмплер для генерации текста: top-k, температура, argmax.

use ggrs_core::util::Lcg;

/// Выбрать следующий токен из логитов одной позиции.
/// temperature == 0.0 → argmax (детерминированный). Иначе: top_k > 0 обрезает до k
/// максимальных, softmax по f64 с температурой, сэмпл через rng.
pub fn sample_next(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Lcg) -> usize {
    assert!(!logits.is_empty());

    // argmax-режим
    if temperature <= 0.0 {
        let mut best = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return best;
    }

    // кандидаты: топ-k по логиту (top_k == 0 или >= len → все)
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = if top_k == 0 {
        idx.len()
    } else {
        top_k.min(idx.len())
    };
    let cand = &idx[..k];

    // softmax по f64 с температурой (стабилизация максимумом)
    let mx = logits[cand[0]] as f64;
    let mut probs: Vec<f64> = cand
        .iter()
        .map(|&i| ((logits[i] as f64 - mx) / temperature as f64).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // сэмпл: u ∈ [0,1) из Lcg
    let u = (rng.next_f32() + 0.5).clamp(0.0, 0.999_999) as f64;
    let mut acc = 0.0f64;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return cand[j];
        }
    }
    cand[k - 1]
}
