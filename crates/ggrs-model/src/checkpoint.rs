//! Бинарный формат чекпоинтов GGRS1 (little-endian).
//!
//! Раскладка файла:
//! ```text
//! magic:      4 байта b"GGRS"
//! version:    u32 = 1
//! n_tensors:  u32
//! на тензор:  name_len u32 | name (utf8) | dtype u32 (0=F32,2=I32) | ne [u64;4] | данные (сырые LE)
//! extra:      step u64 | rng u64 | n_opt u32
//! на opt:     name_len u32 | name | m_len u64 | m ([f32] LE) | v_len u64 | v ([f32] LE)
//! ```
//! Данные тензоров и m/v пишутся/читаются побитово (to_le_bytes/from_le_bytes).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use ggrs_core::{Context, DType, TensorId};

const MAGIC: &[u8; 4] = b"GGRS";
const VERSION: u32 = 1;
/// Максимальная длина имени тензора/параметра (защита от OOM).
const MAX_NAME_LEN: u64 = 4096;

/// Дополнительное состояние тренировки, сохраняемое рядом с весами.
pub struct CheckpointExtra {
    pub step: u64,
    pub rng: u64,
    /// Состояние оптимизатора: (имя параметра, момент m, момент v).
    pub opt: Vec<(String, Vec<f32>, Vec<f32>)>,
}

fn dtype_tag(d: DType) -> io::Result<u32> {
    match d {
        DType::F32 => Ok(0),
        DType::I32 => Ok(2),
        DType::F16 => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "GGRS1: F16 в чекпоинте не поддержан в Фазе 3 (GGUF — Фазы 4/5)",
        )),
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

// ── низкоуровневые писатели ────────────────────────────────────────────────
fn w_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn w_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn w_name<W: Write>(w: &mut W, name: &str) -> io::Result<()> {
    w_u32(w, name.len() as u32)?;
    w.write_all(name.as_bytes())
}
fn w_f32s<W: Write>(w: &mut W, s: &[f32]) -> io::Result<()> {
    let mut buf = Vec::with_capacity(s.len() * 4);
    for &x in s {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf)
}
fn w_i32s<W: Write>(w: &mut W, s: &[i32]) -> io::Result<()> {
    let mut buf = Vec::with_capacity(s.len() * 4);
    for &x in s {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf)
}

// ── низкоуровневые читатели ────────────────────────────────────────────────
fn r_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn r_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Прочитать имя с проверкой на разумный лимит и остаток файла.
fn r_name_checked<R: Read>(r: &mut R, remaining: u64) -> io::Result<(String, u64)> {
    let len_u32 = r_u32(r)?;
    let len = len_u32 as u64;
    // 1. Лимит разумности
    if len > MAX_NAME_LEN {
        return Err(invalid("GGRS1: имя тензора слишком длинное"));
    }
    // 2. Не больше оставшихся байт
    if remaining < 4 || len > remaining - 4 {
        return Err(invalid("GGRS1: недостаточно данных для имени"));
    }
    let len_usize: usize = usize::try_from(len).map_err(|_| invalid("GGRS1: имя не влезает в usize"))?;
    let mut b = vec![0u8; len_usize];
    r.read_exact(&mut b)?;
    let s = String::from_utf8(b).map_err(|_| invalid("GGRS1: имя не UTF-8"))?;
    Ok((s, 4 + len))
}

/// Прочитать f32-вектор с проверкой остатка файла.
fn r_f32s_checked<R: Read>(r: &mut R, n: usize, remaining: u64) -> io::Result<(Vec<f32>, u64)> {
    let nbytes = n.checked_mul(4).ok_or_else(|| invalid("GGRS1: переполнение размера f32-данных"))?;
    if nbytes as u64 > remaining {
        return Err(invalid("GGRS1: недостаточно данных для f32-вектора"));
    }
    let mut bytes = vec![0u8; nbytes];
    r.read_exact(&mut bytes)?;
    let data = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((data, nbytes as u64))
}

/// Прочитать i32-вектор с проверкой остатка файла.
fn r_i32s_checked<R: Read>(r: &mut R, n: usize, remaining: u64) -> io::Result<(Vec<i32>, u64)> {
    let nbytes = n.checked_mul(4).ok_or_else(|| invalid("GGRS1: переполнение размера i32-данных"))?;
    if nbytes as u64 > remaining {
        return Err(invalid("GGRS1: недостаточно данных для i32-вектора"));
    }
    let mut bytes = vec![0u8; nbytes];
    r.read_exact(&mut bytes)?;
    let data = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((data, nbytes as u64))
}

/// Сохранить именованные тензоры + extra в файл формата GGRS1 (атомарно: tmp + rename).
pub fn save_checkpoint(
    path: &Path,
    ctx: &Context,
    named: &[(&str, TensorId)],
    extra: &CheckpointExtra,
) -> io::Result<()> {
    // временный файл-сосед, затем атомарный rename
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(".tmp");
        std::path::PathBuf::from(s)
    };
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w_u32(&mut w, VERSION)?;
        w_u32(&mut w, named.len() as u32)?;
        for &(name, id) in named {
            let t = ctx.t(id);
            let tag = dtype_tag(t.dtype)?;
            w_name(&mut w, name)?;
            w_u32(&mut w, tag)?;
            for d in 0..4 {
                w_u64(&mut w, t.ne[d] as u64)?;
            }
            match t.dtype {
                DType::F32 => w_f32s(&mut w, ctx.data_f32(id))?,
                DType::I32 => w_i32s(&mut w, ctx.data_i32(id))?,
                DType::F16 => return Err(dtype_tag(DType::F16).unwrap_err()),
            }
        }
        // extra
        w_u64(&mut w, extra.step)?;
        w_u64(&mut w, extra.rng)?;
        w_u32(&mut w, extra.opt.len() as u32)?;
        for (name, m, v) in &extra.opt {
            w_name(&mut w, name)?;
            w_u64(&mut w, m.len() as u64)?;
            w_f32s(&mut w, m)?;
            w_u64(&mut w, v.len() as u64)?;
            w_f32s(&mut w, v)?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Временное хранилище данных тензора для транзакционной загрузки.
enum PendingTensor {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

/// Загрузить чекпоинт: данные пишутся в тензоры из `named` (формы/имена/dtype обязаны
/// совпасть — иначе Err). Возвращает extra.
/// Транзакционность: ВСЕ данные читаются и валидируются, и только после успешного
/// полного чтения записываются в Context. Частично повреждённый файл не трогает Context.
pub fn load_checkpoint(
    path: &Path,
    ctx: &mut Context,
    named: &[(&str, TensorId)],
) -> io::Result<CheckpointExtra> {
    load_checkpoint_validated(path, ctx, named, |_| Ok(()))
}

/// Позволяет тренировочному циклу проверить extra до изменения весов.
pub(crate) fn load_checkpoint_validated(
    path: &Path,
    ctx: &mut Context,
    named: &[(&str, TensorId)],
    validate_extra: impl FnOnce(&CheckpointExtra) -> io::Result<()>,
) -> io::Result<CheckpointExtra> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut remaining = file_len;
    let mut r = BufReader::new(file);

    // --- заголовок ---
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    remaining -= 4;
    if &magic != MAGIC {
        return Err(invalid("GGRS1: неверный magic"));
    }
    let version = r_u32(&mut r)?;
    remaining -= 4;
    if version != VERSION {
        return Err(invalid("GGRS1: неподдерживаемая версия"));
    }
    let n = r_u32(&mut r)? as usize;
    remaining -= 4;
    if n != named.len() {
        return Err(invalid("GGRS1: число тензоров не совпадает с ожидаемым"));
    }

    // --- читаем все тензоры во временное хранилище (транзакционность) ---
    let mut pending: Vec<PendingTensor> = Vec::with_capacity(n);

    for &(exp_name, id) in named {
        let (name, consumed) = r_name_checked(&mut r, remaining)?;
        remaining -= consumed;

        if name != exp_name {
            return Err(invalid("GGRS1: имя тензора не совпадает"));
        }
        let tag = r_u32(&mut r)?;
        remaining -= 4;
        let mut ne = [0usize; 4];
        for d in ne.iter_mut() {
            *d = usize::try_from(r_u64(&mut r)?)
                .map_err(|_| invalid("GGRS1: размерность не влезает в usize"))?;
            remaining -= 8;
        }
        let t = ctx.t(id);
        let exp_tag = dtype_tag(t.dtype)?;
        if tag != exp_tag {
            return Err(invalid("GGRS1: dtype тензора не совпадает"));
        }
        if ne != t.ne {
            return Err(invalid("GGRS1: форма тензора не совпадает"));
        }
        let nelem = t.nelements();
        match t.dtype {
            DType::F32 => {
                let (data, consumed_bytes) = r_f32s_checked(&mut r, nelem, remaining)?;
                remaining -= consumed_bytes;
                pending.push(PendingTensor::F32(data));
            }
            DType::I32 => {
                let (data, consumed_bytes) = r_i32s_checked(&mut r, nelem, remaining)?;
                remaining -= consumed_bytes;
                pending.push(PendingTensor::I32(data));
            }
            DType::F16 => return Err(dtype_tag(DType::F16).unwrap_err()),
        }
    }

    // --- читаем extra ---
    let step = r_u64(&mut r)?;
    remaining -= 8;
    let rng = r_u64(&mut r)?;
    remaining -= 8;
    let n_opt = r_u32(&mut r)? as usize;
    remaining -= 4;
    // Даже пустая запись содержит name_len (4), m_len (8) и v_len (8).
    // Проверяем счётчик до выделения памяти по недоверенному заголовку.
    if n_opt as u64 > remaining / 20 {
        return Err(invalid("GGRS1: число состояний оптимизатора превышает размер файла"));
    }
    let mut opt = Vec::with_capacity(n_opt);
    for _ in 0..n_opt {
        let (name, consumed) = r_name_checked(&mut r, remaining)?;
        remaining -= consumed;

        let m_len_u64 = r_u64(&mut r)?;
        remaining -= 8;
        let m_len: usize = usize::try_from(m_len_u64)
            .map_err(|_| invalid("GGRS1: m_len не влезает в usize"))?;
        let (m, consumed_bytes) = r_f32s_checked(&mut r, m_len, remaining)?;
        remaining -= consumed_bytes;

        let v_len_u64 = r_u64(&mut r)?;
        remaining -= 8;
        let v_len: usize = usize::try_from(v_len_u64)
            .map_err(|_| invalid("GGRS1: v_len не влезает в usize"))?;
        let (v, consumed_bytes) = r_f32s_checked(&mut r, v_len, remaining)?;
        remaining -= consumed_bytes;

        opt.push((name, m, v));
    }

    if remaining != 0 {
        return Err(invalid("GGRS1: лишние данные в конце файла"));
    }
    let extra = CheckpointExtra { step, rng, opt };
    validate_extra(&extra)?;

    // --- транзакционная запись: все данные прочитаны успешно ---
    for (i, &(_, id)) in named.iter().enumerate() {
        match &pending[i] {
            PendingTensor::F32(data) => ctx.set_f32(id, data),
            PendingTensor::I32(data) => ctx.set_i32(id, data),
        }
    }

    Ok(extra)
}
