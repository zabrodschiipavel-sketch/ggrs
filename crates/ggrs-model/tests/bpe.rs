use ggrs_model::Bpe;

/// Подготовка сэмпла со смесью: английский, русский, emoji, бинарные байты.
fn mixed_sample() -> Vec<u8> {
    let mut s = Vec::new();
    // Английский текст с повторами (чтобы были частые пары)
    for _ in 0..20 {
        s.extend_from_slice(b"hello world! the quick brown fox jumps over the lazy dog.\n");
    }
    // Русский текст
    for _ in 0..15 {
        s.extend_from_slice("привет мир! как дела?\n".as_bytes());
    }
    // Emoji (многобайтовые UTF-8)
    for _ in 0..10 {
        s.extend_from_slice("😀🎉".as_bytes()); // 8 байт
    }
    // Бинарные байты: 0x00, 0xFF, и соседство
    for _ in 0..5 {
        s.extend_from_slice(&[0x00, 0xFF, 0x00, 0x01, 0x02]);
    }
    s
}

// ── Тест 1: roundtrip (encode → decode) на не-ASCII данных ──────────────────

#[test]
fn roundtrip_utf8() {
    let sample = mixed_sample();
    let bpe = Bpe::train(&sample, 512);
    // vocab должен быть ≤ 512; если сэмпл мал, может быть меньше.
    assert!(bpe.vocab_size() <= 512);
    assert!(bpe.vocab_size() >= 256);

    let test_strings: &[&[u8]] = &[
        b"hello world!",
        "привет мир!".as_bytes(),
        "😀🎉".as_bytes(),
        &[0x00, 0xFF, 0x01],
        b"", // пустая строка
        // Смесь английского, русского, emoji — через String→bytes
        "mixed : английский 😀 text".as_bytes(),
        b"the quick brown fox", // часть обучающего корпуса
    ];

    for ts in test_strings {
        let encoded = bpe.encode(ts);
        let decoded = bpe.decode(&encoded);
        assert_eq!(
            decoded, *ts,
            "roundtrip failed for: {:?}",
            std::str::from_utf8(ts).unwrap_or("<binary>")
        );
    }
}

// ── Тест 2: детерминизм обучения ────────────────────────────────────────────

#[test]
fn train_deterministic() {
    let sample = mixed_sample();
    let bpe1 = Bpe::train(&sample, 400);
    let bpe2 = Bpe::train(&sample, 400);

    assert_eq!(bpe1.vocab_size(), bpe2.vocab_size());

    let test_inputs: &[&[u8]] = &[
        b"hello",
        "привет".as_bytes(),
        "😀".as_bytes(),
        &[0x00, 0xFF],
        b"the quick brown fox",
    ];

    for inp in test_inputs {
        let e1 = bpe1.encode(inp);
        let e2 = bpe2.encode(inp);
        assert_eq!(e1, e2, "encode diverged for same sample+vocab");
    }
}

// ── Тест 3: ручной случай "aaa" с vocab_size=257 ───────────────────────────

#[test]
fn train_hand_case_aaa() {
    let bpe = Bpe::train(b"aaa", 257);
    // Ровно 1 слияние: пара (97,97) — 'a','a'.
    assert_eq!(bpe.vocab_size(), 257);

    let encoded = bpe.encode(b"aaa");
    // Левая пара сливается: [256, 97]
    assert_eq!(encoded, vec![256u16, 97u16]);

    // decode roundtrip
    assert_eq!(bpe.decode(&encoded), b"aaa");

    // decode одного токена 256 → "aa"
    assert_eq!(bpe.decode(&[256]), b"aa");
}

// ── Тест 4: vocab_size=256 — без слияний, чисто byte-level ─────────────────

#[test]
fn vocab_256_is_identity() {
    let sample = mixed_sample();
    let bpe = Bpe::train(&sample, 256);
    assert_eq!(bpe.vocab_size(), 256);

    let input = b"Hi!";
    let encoded = bpe.encode(input);
    // Каждый байт как u16
    assert_eq!(encoded, vec![72u16, 105u16, 33u16]);
    assert_eq!(bpe.decode(&encoded), input);

    // Проверим и бинарные байты
    let bin = &[0x00u8, 0xFFu8, 0x7Fu8, 0x80u8];
    let enc_bin = bpe.encode(bin);
    assert_eq!(enc_bin, vec![0u16, 255u16, 127u16, 128u16]);
    assert_eq!(bpe.decode(&enc_bin), bin);
}

// ── Тест 5: save/load roundtrip ─────────────────────────────────────────────

#[test]
fn save_load_roundtrip() {
    let sample = mixed_sample();
    let orig = Bpe::train(&sample, 350);
    assert!(orig.vocab_size() >= 256);

    let path = std::env::temp_dir().join("ggrs_bpe_test_save_load.txt");
    orig.save(&path).unwrap();

    let loaded = Bpe::load(&path).unwrap();
    assert_eq!(loaded.vocab_size(), orig.vocab_size());

    let test_inputs: &[&[u8]] = &[
        b"hello world!",
        "привет мир!".as_bytes(),
        "😀🎉".as_bytes(),
        &[0x00, 0xFF, 0x01],
        b"",
        b"the quick brown fox",
    ];

    for inp in test_inputs {
        let e_orig = orig.encode(inp);
        let e_loaded = loaded.encode(inp);
        assert_eq!(e_loaded, e_orig, "save/load encode mismatch");
        // И roundtrip через loaded
        assert_eq!(loaded.decode(&e_loaded), *inp);
    }

    // Удалим временный файл
    let _ = std::fs::remove_file(&path);
}

// ── Тест 6: load повреждённого файла → Err ──────────────────────────────────

#[test]
fn load_corrupt_errors() {
    let path = std::env::temp_dir().join("ggrs_bpe_corrupt.txt");

    // Неверный vocab_size (не число)
    std::fs::write(&path, b"abc\n").unwrap();
    assert!(Bpe::load(&path).is_err());

    // vocab_size вне диапазона
    std::fs::write(&path, b"100000\n").unwrap();
    assert!(Bpe::load(&path).is_err());

    // vocab_size 255 (меньше 256)
    std::fs::write(&path, b"255\n").unwrap();
    assert!(Bpe::load(&path).is_err());

    // Не хватает слияний
    std::fs::write(&path, b"258\n10 20\n").unwrap();
    assert!(Bpe::load(&path).is_err());

    // Неверный формат строки слияния (одно число)
    std::fs::write(&path, b"257\n10\n").unwrap();
    assert!(Bpe::load(&path).is_err());

    // Неверное число в слиянии
    std::fs::write(&path, b"257\nabc 20\n").unwrap();
    assert!(Bpe::load(&path).is_err());

    let _ = std::fs::remove_file(&path);
}

// ── Тест 7: досрочная остановка при нехватке пар ────────────────────────────

#[test]
fn early_stop_on_no_pairs() {
    // В "ab" всего одна пара (97,98) — можно сделать максимум 1 слияние.
    let bpe = Bpe::train(b"ab", 1000);
    // vocab_size будет 257 (256 + 1 слияние), а не 1000.
    assert_eq!(bpe.vocab_size(), 257);
    let encoded = bpe.encode(b"ab");
    // Пара (97,98) слита → один токен 256.
    assert_eq!(encoded, vec![256u16]);
    assert_eq!(bpe.decode(&encoded), b"ab");
}

// ── Тест 8: несколько одинаковых пар с разными рангами ──────────────────────

#[test]
fn repeated_pair_different_rank() {
    // "aaaa" — три пары (a,a). Первое слияние схлопнет две из них.
    // vocab=258 → два слияния.
    let bpe = Bpe::train(b"aaaa", 258);
    assert_eq!(bpe.vocab_size(), 258);

    // Первое слияние: (97,97) → 256, "aaaa" → [256, 256]
    // Второе слияние: (256,256) → 257
    let encoded = bpe.encode(b"aaaa");
    assert_eq!(encoded, vec![257u16]);
    assert_eq!(bpe.decode(&encoded), b"aaaa");
}

// ── Тест 9: кэш чанков внутри encode (повторяющиеся слова) ─────────────────

#[test]
fn encode_cache_reuses_chunks() {
    let sample = b"hello world hello world hello world";
    let bpe = Bpe::train(sample, 300);

    // Текст с повторяющимся словом
    let text = b"hello hello hello world world";
    let encoded = bpe.encode(text);
    let decoded = bpe.decode(&encoded);
    assert_eq!(decoded, text);
}

// ── Тест 10: load отвергает ссылку вперёд ──────────────────────────────────

#[test]
fn load_rejects_forward_reference() {
    let path = std::env::temp_dir().join("ggrs_bpe_forward_ref.txt");
    // vocab_size=258 → 2 слияния.
    // Первое слияние "300 65": 300 >= 256+0 — ссылка вперёд, невалид.
    // Второе слияние "65 66" валидно, но до него не дойдём.
    std::fs::write(&path, b"258\n300 65\n65 66\n").unwrap();
    assert!(Bpe::load(&path).is_err(), "forward reference must be rejected");
    let _ = std::fs::remove_file(&path);

    // Дополнительно: проверка при 257 (одно слияние) с референсом на несуществующий токен
    let path2 = std::env::temp_dir().join("ggrs_bpe_forward_ref2.txt");
    // Одно слияние i=0 → max_valid=256. left=255 валидно (байт), right=300 нет (≥256).
    std::fs::write(&path2, b"257\n255 300\n").unwrap();
    assert!(Bpe::load(&path2).is_err(), "right must be < 256 for first merge");
    let _ = std::fs::remove_file(&path2);
}
