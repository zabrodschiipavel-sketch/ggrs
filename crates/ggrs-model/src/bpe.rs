//! Byte-level BPE токенизатор для GPT.
//!
//! Базовый словарь — 256 байт, плюс переменное число слияний.
//! Токены кодируются как u16 (vocab ≤ 65536). Претокенизация: разбиение на чанки
//! по пробельным границам (ведущий пробел прилипает к слову). Обучение и
//! кодирование используют одинаковое разбиение, что даёт детерминизм.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Пробельные байты, используемые как границы чанков.
fn is_ws(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
}

/// Разбить текст на чанки. Граница ставится перед байтом `text[i]`,
/// если `is_ws(text[i]) && !is_ws(text[i-1])`.
pub fn chunks(text: &[u8]) -> Vec<&[u8]> {
    if text.is_empty() {
        return vec![];
    }
    let mut starts = vec![0usize];
    for i in 1..text.len() {
        if is_ws(text[i]) && !is_ws(text[i - 1]) {
            starts.push(i);
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for w in starts.windows(2) {
        out.push(&text[w[0]..w[1]]);
    }
    out.push(&text[*starts.last().unwrap()..]);
    out
}

/// Byte-level BPE токенизатор. Приватные поля:
/// - `merges` — список слияний, где merges[i] порождает токен 256+i из пары.
/// - `decode_table` — предразвёрнутые байты каждого токена.
pub struct Bpe {
    merges: Vec<(u16, u16)>,
    decode_table: Vec<Vec<u8>>,
}

impl Bpe {
    /// Обучить на сэмпле: 256 базовых байт + (vocab_size−256) слияний.
    /// vocab_size ∈ [256, 65536]. Если сэмпл кончился (нет пар) — обучение
    /// останавливается досрочно, vocab_size() вернёт фактический размер.
    pub fn train(sample: &[u8], vocab_size: u32) -> Bpe {
        let vocab_size = vocab_size.clamp(256, 65536) as usize;
        let num_merges = vocab_size.saturating_sub(256);

        // Разбиваем на чанки и дедуплицируем с весами (ускоряет подсчёт частот пар).
        let raw_chunks = chunks(sample);
        let mut chunk_counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for c in &raw_chunks {
            *chunk_counts.entry(c.to_vec()).or_default() += 1;
        }

        // Каждый чанк — последовательность байтовых токенов + вес.
        let mut corpus: Vec<(Vec<u32>, u64)> = chunk_counts
            .into_iter()
            .map(|(bytes, cnt)| (bytes.into_iter().map(|b| b as u32).collect(), cnt))
            .collect();

        let mut merges: Vec<(u16, u16)> = Vec::with_capacity(num_merges);

        for _merge_idx in 0..num_merges {
            // Подсчёт пар
            let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
            for (chunk, cnt) in &corpus {
                for j in 0..chunk.len().saturating_sub(1) {
                    *pair_counts.entry((chunk[j], chunk[j + 1])).or_default() += cnt;
                }
            }
            if pair_counts.is_empty() {
                break;
            }

            // Выбор пары: максимум по счётчику, при равенстве — лексикографически меньшая.
            let best_pair = pair_counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0).reverse()))
                .unwrap()
                .0;

            let new_id = 256 + merges.len() as u32;
            merges.push((best_pair.0 as u16, best_pair.1 as u16));

            // Замена во всех чанках (стандартный проход minbpe).
            for (chunk, _cnt) in &mut corpus {
                let mut new_chunk = Vec::with_capacity(chunk.len());
                let mut j = 0;
                while j < chunk.len() {
                    if j + 1 < chunk.len()
                        && chunk[j] == best_pair.0
                        && chunk[j + 1] == best_pair.1
                    {
                        new_chunk.push(new_id);
                        j += 2;
                    } else {
                        new_chunk.push(chunk[j]);
                        j += 1;
                    }
                }
                *chunk = new_chunk;
            }
        }

        // Построение decode_table
        let mut decode_table: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        for &(l, r) in &merges {
            let left_bytes = decode_table[l as usize].clone();
            let right_bytes = decode_table[r as usize].clone();
            let mut merged = Vec::with_capacity(left_bytes.len() + right_bytes.len());
            merged.extend_from_slice(&left_bytes);
            merged.extend_from_slice(&right_bytes);
            decode_table.push(merged);
        }

        Bpe {
            merges,
            decode_table,
        }
    }

    /// Закодировать текст в токены u16 (то же разбиение на чанки, что и при обучении).
    pub fn encode(&self, text: &[u8]) -> Vec<u16> {
        if text.is_empty() {
            return vec![];
        }

        // Индекс ранга: пара → индекс в merges (чем меньше, тем раньше слито).
        let rank: HashMap<(u16, u16), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();

        let mut cache: HashMap<Vec<u8>, Vec<u16>> = HashMap::new();
        let mut result = Vec::new();

        for chunk in chunks(text) {
            if let Some(tokens) = cache.get(chunk) {
                result.extend_from_slice(tokens);
                continue;
            }

            let mut ids: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();

            loop {
                // Ищем пару с наименьшим рангом.
                let mut best_rank: Option<usize> = None;
                let mut best_pair: Option<(u32, u32)> = None;
                for j in 0..ids.len().saturating_sub(1) {
                    let a = ids[j];
                    let b = ids[j + 1];
                    if a > u16::MAX as u32 || b > u16::MAX as u32 {
                        continue;
                    }
                    if let Some(&r) = rank.get(&(a as u16, b as u16))
                        && best_rank.is_none_or(|br| r < br)
                    {
                        best_rank = Some(r);
                        best_pair = Some((a, b));
                    }
                }

                if best_rank.is_none() {
                    break;
                }

                let (a, b) = best_pair.unwrap();
                let new_id = 256 + best_rank.unwrap() as u32;

                // Слить ВСЕ вхождения этой пары (один проход).
                let mut new_ids = Vec::with_capacity(ids.len());
                let mut j = 0;
                while j < ids.len() {
                    if j + 1 < ids.len() && ids[j] == a && ids[j + 1] == b {
                        new_ids.push(new_id);
                        j += 2;
                    } else {
                        new_ids.push(ids[j]);
                        j += 1;
                    }
                }
                ids = new_ids;
            }

            let tokens: Vec<u16> = ids.into_iter().map(|id| id as u16).collect();
            cache.insert(chunk.to_vec(), tokens.clone());
            result.extend(tokens);
        }

        result
    }

    /// Декодировать токены обратно в байты (O(выход), без рекурсии).
    pub fn decode(&self, ids: &[u16]) -> Vec<u8> {
        let mut result = Vec::new();
        for &id in ids {
            result.extend_from_slice(&self.decode_table[id as usize]);
        }
        result
    }

    /// Сохранить BPE в текстовый файл: vocab_size, затем пары left right.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let tmp = {
            let mut s = path.as_os_str().to_os_string();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            writeln!(w, "{}", self.vocab_size())?;
            for &(left, right) in &self.merges {
                writeln!(w, "{} {}", left, right)?;
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    /// Загрузить BPE из текстового файла (формат save).
    pub fn load(path: &Path) -> io::Result<Bpe> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        let first_line = first_line.trim();
        let vocab_size: usize = first_line.parse().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "BPE: неверный vocab_size")
        })?;

        if !(256..=65536).contains(&vocab_size) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BPE: vocab_size вне [256, 65536]",
            ));
        }

        let num_merges = vocab_size - 256;
        let mut merges: Vec<(u16, u16)> = Vec::with_capacity(num_merges);

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "BPE: ожидалось два числа в строке слияния",
                ));
            }
            let left: u16 = parts[0].parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "BPE: неверный left в слиянии")
            })?;
            let right: u16 = parts[1].parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "BPE: неверный right в слиянии")
            })?;
            merges.push((left, right));
        }

        if merges.len() != num_merges {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BPE: ожидалось {} слияний, прочитано {}",
                    num_merges,
                    merges.len()
                ),
            ));
        }

        // Построение decode_table
        let mut decode_table: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        for &(l, r) in &merges {
            let left_bytes = decode_table[l as usize].clone();
            let right_bytes = decode_table[r as usize].clone();
            let mut merged = Vec::with_capacity(left_bytes.len() + right_bytes.len());
            merged.extend_from_slice(&left_bytes);
            merged.extend_from_slice(&right_bytes);
            decode_table.push(merged);
        }

        Ok(Bpe {
            merges,
            decode_table,
        })
    }

    /// Размер словаря: 256 базовых + количество слияний.
    pub fn vocab_size(&self) -> u32 {
        256 + self.merges.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Границы чанков по пробелам: ведущий пробел прилипает к слову.
    #[test]
    fn chunks_leading_space_attaches() {
        let input = b"hello world\n foo";
        let cs = chunks(input);
        let result = cs.to_vec();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], b"hello");
        assert_eq!(result[1], b" world");
        assert_eq!(result[2], b"\n foo");
    }

    /// Пустой ввод → пустой список чанков.
    #[test]
    fn chunks_empty() {
        assert!(chunks(b"").is_empty());
    }

    /// Только пробелы: один чанк (все пробелы склеены).
    #[test]
    fn chunks_all_ws() {
        let cs = chunks(b"   \n\t");
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0], b"   \n\t");
    }
}
