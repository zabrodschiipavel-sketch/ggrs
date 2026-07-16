use ggrs_core::{Context, DType, TensorId};
use ggrs_model::{load_checkpoint, save_checkpoint, CheckpointExtra};

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

/// Round-trip: F32 + I32 тензоры и extra сохраняются и читаются бит-в-бит.
#[test]
fn roundtrip_f32_i32_extra() {
    let path = tmp_path("ggrs_ckpt_roundtrip.ggrs");
    // источник
    let mut ctx = Context::new(1 << 20);
    let w = ctx.new_tensor_2d(DType::F32, 16, 8); // 128 f32 (>4КБ данных суммарно с ниже)
    let big = ctx.new_tensor_1d(DType::F32, 2000); // >4КБ — проверка буферизации
    let ids = ctx.new_tensor_1d(DType::I32, 5);
    let wv: Vec<f32> = (0..128).map(|i| i as f32 * 0.5 - 3.0).collect();
    let bv: Vec<f32> = (0..2000).map(|i| (i as f32).sin()).collect();
    ctx.set_f32(w, &wv);
    ctx.set_f32(big, &bv);
    ctx.set_i32(ids, &[7, -1, 0, 42, -999]);

    let extra = CheckpointExtra {
        step: 12345,
        rng: 0xDEADBEEF,
        opt: vec![
            ("w".to_string(), vec![0.1, 0.2, 0.3], vec![1.0, 2.0, 3.0]),
            ("big".to_string(), vec![-0.5; 4], vec![9.9; 4]),
        ],
    };
    let named: [(&str, TensorId); 3] = [("w", w), ("big", big), ("ids", ids)];
    save_checkpoint(&path, &ctx, &named, &extra).unwrap();

    // приёмник — свежий ctx той же структуры, данные нулевые до загрузки
    let mut ctx2 = Context::new(1 << 20);
    let w2 = ctx2.new_tensor_2d(DType::F32, 16, 8);
    let big2 = ctx2.new_tensor_1d(DType::F32, 2000);
    let ids2 = ctx2.new_tensor_1d(DType::I32, 5);
    let named2: [(&str, TensorId); 3] = [("w", w2), ("big", big2), ("ids", ids2)];
    let got = load_checkpoint(&path, &mut ctx2, &named2).unwrap();

    assert_eq!(ctx2.data_f32(w2), wv.as_slice());
    assert_eq!(ctx2.data_f32(big2), bv.as_slice());
    assert_eq!(ctx2.data_i32(ids2), &[7, -1, 0, 42, -999]);
    assert_eq!(got.step, 12345);
    assert_eq!(got.rng, 0xDEADBEEF);
    assert_eq!(got.opt.len(), 2);
    assert_eq!(got.opt[0].0, "w");
    assert_eq!(got.opt[0].1, vec![0.1, 0.2, 0.3]);
    assert_eq!(got.opt[1].2, vec![9.9; 4]);
    let _ = std::fs::remove_file(&path);
}

/// Повреждённый magic → Err (не паника).
#[test]
fn corrupt_magic_errors() {
    let path = tmp_path("ggrs_ckpt_badmagic.ggrs");
    std::fs::write(&path, b"XXXX\x01\x00\x00\x00").unwrap();
    let mut ctx = Context::new(1 << 16);
    let a = ctx.new_tensor_1d(DType::F32, 2);
    let named: [(&str, TensorId); 1] = [("a", a)];
    assert!(load_checkpoint(&path, &mut ctx, &named).is_err());
    let _ = std::fs::remove_file(&path);
}

/// Несовпадение формы при загрузке → Err.
#[test]
fn shape_mismatch_errors() {
    let path = tmp_path("ggrs_ckpt_shape.ggrs");
    let mut ctx = Context::new(1 << 16);
    let a = ctx.new_tensor_1d(DType::F32, 4);
    ctx.set_f32(a, &[1.0, 2.0, 3.0, 4.0]);
    let named: [(&str, TensorId); 1] = [("a", a)];
    save_checkpoint(&path, &ctx, &named, &CheckpointExtra { step: 0, rng: 0, opt: vec![] }).unwrap();

    let mut ctx2 = Context::new(1 << 16);
    let b = ctx2.new_tensor_1d(DType::F32, 8); // другая форма
    let named2: [(&str, TensorId); 1] = [("a", b)];
    assert!(load_checkpoint(&path, &mut ctx2, &named2).is_err());
    let _ = std::fs::remove_file(&path);
}

/// Транзакционность: обрезанный файл не меняет Context.
#[test]
fn load_is_transactional() {
    let path = tmp_path("ggrs_ckpt_tx.ggrs");
    let trunc_path = tmp_path("ggrs_ckpt_tx_trunc.ggrs");

    // Сохраняем два F32-тензора
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_1d(DType::F32, 4);
    let b = ctx.new_tensor_1d(DType::F32, 4);
    ctx.set_f32(a, &[1.0, 2.0, 3.0, 4.0]);
    ctx.set_f32(b, &[5.0, 6.0, 7.0, 8.0]);
    let named: [(&str, TensorId); 2] = [("a", a), ("b", b)];
    let extra = CheckpointExtra {
        step: 100,
        rng: 200,
        opt: vec![],
    };
    save_checkpoint(&path, &ctx, &named, &extra).unwrap();

    // Читаем файл и обрезаем до 60%
    let bytes = std::fs::read(&path).unwrap();
    let truncated_len = (bytes.len() as f64 * 0.6) as usize;
    let truncated = &bytes[..truncated_len];
    std::fs::write(&trunc_path, truncated).unwrap();

    // Свежий ctx с маркерными данными (все 7.0)
    let mut ctx2 = Context::new(1 << 20);
    let a2 = ctx2.new_tensor_1d(DType::F32, 4);
    let b2 = ctx2.new_tensor_1d(DType::F32, 4);
    ctx2.set_f32(a2, &[7.0; 4]);
    ctx2.set_f32(b2, &[7.0; 4]);
    let named2: [(&str, TensorId); 2] = [("a", a2), ("b", b2)];

    // Загрузка обрезанного файла → Err
    let result = load_checkpoint(&trunc_path, &mut ctx2, &named2);
    assert!(result.is_err(), "truncated checkpoint must produce error");

    // Context НЕ изменился (всё ещё 7.0)
    assert_eq!(
        ctx2.data_f32(a2),
        &[7.0; 4],
        "ctx data changed despite failed load"
    );
    assert_eq!(
        ctx2.data_f32(b2),
        &[7.0; 4],
        "ctx data changed despite failed load"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&trunc_path);
}

/// Абсурдная длина имени → Err без OOM.
#[test]
fn load_rejects_absurd_name_len() {
    let path = tmp_path("ggrs_ckpt_absurd_name.ggrs");

    // Собираем файл вручную: magic + version + n_tensors=1 + name_len=0xFFFFFFFF
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGRS");
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u32.to_le_bytes()); // 1 тензор
    buf.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // name_len = 4ГБ — абсурд
                                                        // остальное неважно, упадёт на name_len
    std::fs::write(&path, &buf).unwrap();

    let mut ctx = Context::new(1 << 16);
    let x = ctx.new_tensor_1d(DType::F32, 2);
    let named: [(&str, TensorId); 1] = [("x", x)];
    let result = load_checkpoint(&path, &mut ctx, &named);
    assert!(result.is_err(), "absurd name_len must be rejected");

    let _ = std::fs::remove_file(&path);
}
