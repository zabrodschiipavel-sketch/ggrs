//! Формат GGTK (бинарный поток токенов), сэмплирование корпуса для обучения BPE,
//! загрузка потока токенов (TokenBin) и нарезка окон обучения (WindowSampler, val_windows).
//!
//! GGTK — файл потока токенов (little-endian):
//! ```text
//! magic:     4 байта b"GGTK"
//! vocab_size u32
//! n_tokens   u64
//! данные:    n_tokens × u16 (LE)
//! ```

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use ggrs_core::util::Lcg;

const GGTK_MAGIC: &[u8; 4] = b"GGTK";

// ── TokenBin ─────────────────────────────────────────────────────────────────

/// Загруженный поток токенов формата GGTK.
pub struct TokenBin {
    pub tokens: Vec<u16>,
    pub vocab_size: u32,
}

impl TokenBin {
    /// Прочитать файл формата GGTK (magic "GGTK", vocab_size u32, n_tokens u64, данные u16 LE).
    /// Несовпадение magic/размера → io::Error.
    pub fn load(path: &Path) -> io::Result<TokenBin> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != GGTK_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "GGTK: неверный magic"));
        }
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        let vocab_size = u32::from_le_bytes(b4);

        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let n_u64 = u64::from_le_bytes(b8);

        // n_tokens должен влезать в usize (нужно для 32-бит платформ)
        let n: usize = usize::try_from(n_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "GGTK: n_tokens не влезает в usize",
            )
        })?;

        // Предотвращение переполнения при вычислении размера данных
        let data_size = n.checked_mul(2).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "GGTK: переполнение n_tokens * 2",
            )
        })?;

        // Проверка полного размера файла: 4 (magic) + 4 (vocab_size) + 8 (n_tokens) + data_size
        let expected_size = (data_size as u64).checked_add(16).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "GGTK: переполнение полного размера файла",
            )
        })?;
        if file_len != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGTK: размер файла не соответствует n_tokens",
            ));
        }

        let mut bytes = vec![0u8; data_size];
        r.read_exact(&mut bytes)?;

        // Парсинг токенов с проверкой каждого: токен < vocab_size
        let mut tokens = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(2) {
            let t = u16::from_le_bytes([chunk[0], chunk[1]]);
            if (t as u32) >= vocab_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGTK: токен вне словаря",
                ));
            }
            tokens.push(t);
        }

        Ok(TokenBin { tokens, vocab_size })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ── WindowSampler ────────────────────────────────────────────────────────────

/// Сэмплер случайных окон обучения.
pub struct WindowSampler {
    rng: Lcg,
    t: usize,
}

impl WindowSampler {
    pub fn new(seed: u64, t: usize) -> Self {
        WindowSampler {
            rng: Lcg::new(seed),
            t,
        }
    }

    /// Случайное окно длины t: (ids, targets), где ids = tokens[s..s+t] как i32,
    /// targets = tokens[s+1..s+t+1] как i32. Начало s ∈ [0, len-t-1].
    /// Требует bin.len() > t.
    pub fn next_window(&mut self, bin: &TokenBin) -> (Vec<i32>, Vec<i32>) {
        assert!(
            bin.len() > self.t,
            "WindowSampler: корпус короче окна+1"
        );
        let max_start = bin.len() - self.t - 1; // включительно
        // Lcg::next_f32 ∈ [-0.5, 0.5) → u ∈ [0,1) → индекс
        let u = (self.rng.next_f32() + 0.5).clamp(0.0, 0.999_999);
        let mut s = (u * (max_start as f32 + 1.0)) as usize;
        if s > max_start {
            s = max_start;
        }
        let ids: Vec<i32> = bin.tokens[s..s + self.t]
            .iter()
            .map(|&x| x as i32)
            .collect();
        let targets: Vec<i32> = bin.tokens[s + 1..s + self.t + 1]
            .iter()
            .map(|&x| x as i32)
            .collect();
        (ids, targets)
    }

    /// Состояние ГПСЧ сэмплера (для чекпоинта).
    pub fn rng_state(&self) -> u64 { self.rng.0 }
    /// Восстановить состояние ГПСЧ (резюм с чекпоинта).
    pub fn set_rng_state(&mut self, s: u64) { self.rng.0 = s; }
}

// ── val_windows ──────────────────────────────────────────────────────────────

/// n детерминированных валидационных окон с равномерным шагом по корпусу.
/// Стабильны между вызовами (не зависят от порядка запусков). Требует bin.len() > t.
pub fn val_windows(bin: &TokenBin, t: usize, n: usize) -> Vec<(Vec<i32>, Vec<i32>)> {
    assert!(
        bin.len() > t,
        "val_windows: корпус короче окна+1"
    );
    let max_start = bin.len() - t - 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // равномерный шаг; при n==1 берём s=0
        let s = if n <= 1 {
            0
        } else {
            (i * max_start) / (n - 1)
        };
        let ids: Vec<i32> = bin.tokens[s..s + t]
            .iter()
            .map(|&x| x as i32)
            .collect();
        let targets: Vec<i32> = bin.tokens[s + 1..s + t + 1]
            .iter()
            .map(|&x| x as i32)
            .collect();
        out.push((ids, targets));
    }
    out
}

// ── write_token_bin ──────────────────────────────────────────────────────────

/// Записать поток токенов в файл формата GGTK (атомарно: tmp + rename).
pub fn write_token_bin(path: &Path, vocab_size: u32, tokens: &[u16]) -> io::Result<()> {
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(".tmp");
        std::path::PathBuf::from(s)
    };
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(GGTK_MAGIC)?;
        w.write_all(&vocab_size.to_le_bytes())?;
        w.write_all(&(tokens.len() as u64).to_le_bytes())?;
        let mut buf = Vec::with_capacity(tokens.len() * 2);
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        w.write_all(&buf)?;
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}

// ── sample_corpus ────────────────────────────────────────────────────────────

/// Равномерный сэмпл корпуса для обучения BPE: конкатенация каждого step-го блока
/// по 64 КиБ, пока не наберётся ~sample_mib МиБ (или корпус не кончится).
/// step подбирается так, чтобы покрыть весь файл: step = max(1, total_blocks / target_blocks).
pub fn sample_corpus(data: &[u8], sample_mib: usize) -> Vec<u8> {
    const BLOCK: usize = 64 * 1024;
    let target_bytes = sample_mib * 1024 * 1024;

    if data.len() <= target_bytes {
        return data.to_vec();
    }

    let total_blocks = data.len().div_ceil(BLOCK);
    let target_blocks = target_bytes.div_ceil(BLOCK).max(1);
    let step = (total_blocks / target_blocks).max(1);

    let mut out = Vec::with_capacity(target_bytes + BLOCK);
    let mut b = 0;
    while b < total_blocks && out.len() < target_bytes {
        let start = b * BLOCK;
        let end = (start + BLOCK).min(data.len());
        out.extend_from_slice(&data[start..end]);
        b += step;
    }
    out
}
