//! Формат GGTK (бинарный поток токенов) и сэмплирование корпуса для обучения BPE.
//!
//! GGTK — файл потока токенов (little-endian):
//! ```text
//! magic:     4 байта b"GGTK"
//! vocab_size u32
//! n_tokens   u64
//! данные:    n_tokens × u16 (LE)
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const GGTK_MAGIC: &[u8; 4] = b"GGTK";

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
