use ggrs_model::{val_windows, write_token_bin, TokenBin, WindowSampler};

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

// ── Тест 1: roundtrip записи и загрузки TokenBin ─────────────────────────────

#[test]
fn tokenbin_roundtrip() {
    let tokens: Vec<u16> = (0..300).collect();
    let vocab_size = 4096u32;
    let path = tmp_path("ggrs_test_tokenbin_roundtrip.bin");

    let _ = std::fs::remove_file(&path);

    write_token_bin(&path, vocab_size, &tokens).unwrap();
    let loaded = TokenBin::load(&path).unwrap();

    assert_eq!(loaded.tokens, tokens, "токены бит-в-бит не совпали");
    assert_eq!(loaded.vocab_size, vocab_size, "vocab_size не совпал");

    let _ = std::fs::remove_file(&path);
}

// ── Тест 2: детерминизм WindowSampler ────────────────────────────────────────

#[test]
fn sampler_deterministic() {
    let tokens: Vec<u16> = (0..300).collect();
    let bin = TokenBin {
        tokens,
        vocab_size: 4096,
    };
    let t = 64;

    let mut s1 = WindowSampler::new(7, t);
    let mut s2 = WindowSampler::new(7, t);

    for _ in 0..10 {
        let (ids1, targets1) = s1.next_window(&bin);
        let (ids2, targets2) = s2.next_window(&bin);
        assert_eq!(ids1, ids2, "ids не совпали для двух sampler'ов");
        assert_eq!(
            targets1, targets2,
            "targets не совпали для двух sampler'ов"
        );
    }
}

// ── Тест 3: инварианты окон (длины и сдвиг targets = ids сдвиг на 1) ────────

#[test]
fn sampler_bounds_and_targets() {
    let tokens: Vec<u16> = (0..300).collect();
    let bin = TokenBin {
        tokens,
        vocab_size: 4096,
    };
    let t = 64;

    let mut sampler = WindowSampler::new(42, t);

    for _ in 0..1000 {
        let (ids, targets) = sampler.next_window(&bin);
        assert_eq!(ids.len(), t, "длина ids != t");
        assert_eq!(targets.len(), t, "длина targets != t");
        // Все значения ids — валидные u16 (неотрицательные i32, < 65536)
        for &id in &ids {
            assert!((0..65536).contains(&id), "ids содержит невалидный токен {}", id);
        }
        for &tg in &targets {
            assert!((0..65536).contains(&tg), "targets содержит невалидный токен {}", tg);
        }
        // Инвариант: targets[j] == ids[j+1] для j=0..t-2
        for j in 0..t - 1 {
            assert_eq!(
                targets[j], ids[j + 1],
                "targets[{}] != ids[{}] ({} != {})",
                j,
                j + 1,
                targets[j],
                ids[j + 1]
            );
        }
    }
}

// Вспомогательная: найти начало окна ids в tokens (токены уникальны 0..299)
fn find_start(tokens: &[u16], ids: &[i32]) -> usize {
    let first = ids[0] as u16;
    tokens.iter().position(|&x| x == first).unwrap()
}

// ── Тест 4: val_windows стабильна и старты неубывающие ───────────────────────

#[test]
fn val_windows_stable_and_disjoint_starts() {
    let tokens: Vec<u16> = (0..300).collect();
    let bin = TokenBin {
        tokens: tokens.clone(),
        vocab_size: 4096,
    };
    let t = 32;
    let n = 5;

    let w1 = val_windows(&bin, t, n);
    let w2 = val_windows(&bin, t, n);

    // Стабильность
    assert_eq!(w1, w2, "val_windows нестабильна между вызовами");

    // Стартовые позиции неубывающие
    let starts: Vec<usize> = w1
        .iter()
        .map(|(ids, _)| find_start(&bin.tokens, ids))
        .collect();

    for pair in starts.windows(2) {
        assert!(
            pair[0] <= pair[1],
            "стартовые позиции не монотонны: {} > {}",
            pair[0],
            pair[1]
        );
    }

    // Всего n окон
    assert_eq!(w1.len(), n, "число окон != n");
}
