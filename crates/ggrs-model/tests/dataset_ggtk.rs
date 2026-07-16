use ggrs_model::{sample_corpus, write_token_bin};

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

// ── Тест 1: заголовок и payload GGTK ─────────────────────────────────────────

#[test]
fn write_token_bin_header_and_payload() {
    let tokens: Vec<u16> = vec![0, 255, 256, 4095, 42];
    let vocab_size = 4096u32;
    let path = tmp_path("ggrs_test_ggtk_header.bin");

    // Удаляем если остался от предыдущего запуска
    let _ = std::fs::remove_file(&path);

    write_token_bin(&path, vocab_size, &tokens).unwrap();

    let raw = std::fs::read(&path).unwrap();

    // Проверяем размер: 4 (magic) + 4 (vocab_size) + 8 (n_tokens) + 5*2 (payload)
    assert_eq!(raw.len(), 4 + 4 + 8 + 5 * 2);

    // magic
    assert_eq!(&raw[0..4], b"GGTK");

    // vocab_size (u32 LE)
    let got_vocab =
        u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    assert_eq!(got_vocab, 4096);

    // n_tokens (u64 LE)
    let got_n =
        u64::from_le_bytes([raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]]);
    assert_eq!(got_n, 5);

    // payload: 5 × u16 LE
    let payload = &raw[16..];
    let mut got_tokens = Vec::with_capacity(5);
    for chunk in payload.chunks_exact(2) {
        let t = u16::from_le_bytes([chunk[0], chunk[1]]);
        got_tokens.push(t);
    }
    assert_eq!(got_tokens, tokens);

    let _ = std::fs::remove_file(&path);
}

// ── Тест 2: маленькие данные возвращаются целиком ───────────────────────────

#[test]
fn sample_corpus_small_returns_all() {
    let data: Vec<u8> = (0..1000usize).map(|i| (i % 256) as u8).collect();
    let result = sample_corpus(&data, 16);
    assert_eq!(result.len(), 1000);
    assert_eq!(result, data);
}

// ── Тест 3: большие данные сэмплируются равномерно ──────────────────────────

#[test]
fn sample_corpus_large_subsamples() {
    // 40 МиБ данных (одинаковые байты — неважно, нас интересует размер)
    let data = vec![b'x'; 40 << 20]; // 40 MiB
    let result = sample_corpus(&data, 8);

    // Результат должен быть в диапазоне [~7 MiB, ~9 MiB]
    let min_expected = 7 * 1024 * 1024;
    let max_expected = 9 * 1024 * 1024;
    assert!(
        result.len() >= min_expected,
        "результат {} байт, ожидалось >= {}",
        result.len(),
        min_expected
    );
    assert!(
        result.len() <= max_expected,
        "результат {} байт, ожидалось <= {}",
        result.len(),
        max_expected
    );

    // Не пустой
    assert!(!result.is_empty());

    // Все байты — 'x' (семплирование не должно менять содержимое)
    assert!(result.iter().all(|&b| b == b'x'));
}
