//! Тесты тренировочного цикла: loss уменьшается, резюм бит-в-бит, NaN-guard.
//!
//! Используем синтетический TokenBin с детерминированным псевдопаттерном:
//! tokens = (0..4096).map(|i| ((i * 7) % 60 + 1) as u16).collect()
//! vocab_size = 65.

use std::sync::atomic::{AtomicU64, Ordering};

use ggrs_core::Context;
use ggrs_model::{
    build_gpt, train, GptConfig, TokenBin, TrainConfig,
};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Уникальный каталог для теста.
fn test_dir(name: &str) -> std::path::PathBuf {
    let c = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ggrs_{}_{}", name, c))
}

/// Синтетический train-бин: 4096 токенов, vocab=65.
fn synth_train() -> TokenBin {
    let tokens: Vec<u16> = (0..4096).map(|i| ((i * 7) % 60 + 1) as u16).collect();
    TokenBin {
        tokens,
        vocab_size: 65,
    }
}

/// Синтетический val-бин: 512 токенов, тот же паттерн.
fn synth_val() -> TokenBin {
    let tokens: Vec<u16> = (0..512).map(|i| ((i * 7) % 60 + 1) as u16).collect();
    TokenBin {
        tokens,
        vocab_size: 65,
    }
}

/// Тест 1: loss уменьшается вдвое за 200 шагов.
#[test]
fn loss_halves_in_200_steps() {
    let bin_train = synth_train();
    let bin_val = synth_val();

    let mut ctx = Context::new(1 << 26);
    let cfg = GptConfig::tiny();
    let gpt = build_gpt(&mut ctx, &cfg);

    let out_dir = test_dir("loss_halves");

    let train_cfg = TrainConfig {
        steps: 200,
        total_steps: 200,
        grad_accum: 2,
        lr: 1e-3,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 0,
        eval_windows: 8,
        ckpt_every: 0,
        threads: 1,
        out_dir,
        seed: 1,
    };

    let report = train(&mut ctx, &gpt, &bin_train, &bin_val, &train_cfg, None)
        .expect("train не должен паниковать");

    // Стартовый loss ~ln(65) ≈ 4.17
    let ln65 = (65.0f32).ln();
    assert!(
        report.final_train_loss < 0.5 * ln65,
        "final_train_loss={:.4}, ожидалось < {:.4}",
        report.final_train_loss,
        0.5 * ln65
    );
    assert_eq!(
        report.skipped_steps, 0,
        "skipped_steps должно быть 0, got {}",
        report.skipped_steps
    );

    eprintln!(
        "loss_halves: final_train_loss={:.6}, final_val_loss={:.6}, tokens_seen={}, skipped={}",
        report.final_train_loss, report.final_val_loss, report.tokens_seen, report.skipped_steps
    );
}

/// Тест 2: резюм бит-в-бит.
#[test]
fn resume_bitwise() {
    let bin_train = synth_train();
    let bin_val = synth_val();

    // --- Прогон A: 200 шагов одним заходом ---
    let out_a = test_dir("resume_a");
    // Не удаляем: каталог свежий

    let mut ctx_a = Context::new(1 << 26);
    let cfg = GptConfig::tiny();
    let gpt_a = build_gpt(&mut ctx_a, &cfg);

    let train_cfg_a = TrainConfig {
        steps: 200,
        total_steps: 200, // общий горизонт для всех сегментов
        grad_accum: 2,
        lr: 1e-3,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 0,
        eval_windows: 8,
        ckpt_every: 0,
        threads: 1,
        out_dir: out_a,
        seed: 1,
    };

    let report_a = train(&mut ctx_a, &gpt_a, &bin_train, &bin_val, &train_cfg_a, None)
        .expect("train A не должен паниковать");

    // --- Прогон Б: 100 шагов → чекпоинт → свежий ctx → resume до 200 ---
    let out_b = test_dir("resume_b");

    // Первые 100 шагов
    {
        let mut ctx_b = Context::new(1 << 26);
        let gpt_b = build_gpt(&mut ctx_b, &cfg);

        let train_cfg_b = TrainConfig {
            steps: 100,
            total_steps: 200, // ТОТ ЖЕ горизонт, что и в A
            grad_accum: 2,
            lr: 1e-3,
            warmup_frac: 0.02,
            warmdown_frac: 0.33,
            clip: 1.0,
            eval_every: 0,
            eval_windows: 8,
            ckpt_every: 0, // чекпоинт только в конце (последний шаг)
            threads: 1,
            out_dir: out_b.clone(),
            seed: 1,
        };

        let report_b = train(&mut ctx_b, &gpt_b, &bin_train, &bin_val, &train_cfg_b, None)
            .expect("train B (100 шагов) не должен паниковать");

        eprintln!("train B done: loss={:.6}", report_b.final_train_loss);
    }

    // Убедимся, что чекпоинт существует (создан на последнем шаге цикла B)
    let ckpt_path = out_b.join("ckpt.ggrs");
    assert!(
        ckpt_path.exists(),
        "Чекпоинт не создан: {:?}",
        ckpt_path
    );

    // Резюм: свежий Context + build_gpt + resume
    let mut ctx_c = Context::new(1 << 26);
    let gpt_c = build_gpt(&mut ctx_c, &cfg);

    let train_cfg_c = TrainConfig {
        steps: 200,
        total_steps: 200, // ТОТ ЖЕ горизонт, что и в A/B
        grad_accum: 2,
        lr: 1e-3,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 0,
        eval_windows: 8,
        ckpt_every: 0,
        threads: 1,
        out_dir: out_b.clone(),
        seed: 1,
    };

    let report_c = train(
        &mut ctx_c,
        &gpt_c,
        &bin_train,
        &bin_val,
        &train_cfg_c,
        Some(&ckpt_path),
    )
    .expect("train C (resume) не должен паниковать");

    // Сверка бит-в-бит
    assert_eq!(
        report_a.final_train_loss.to_bits(),
        report_c.final_train_loss.to_bits(),
        "final_train_loss не совпадает бит-в-бит: A={:.10} C={:.10}",
        report_a.final_train_loss,
        report_c.final_train_loss
    );
    assert_eq!(
        report_a.final_val_loss.to_bits(),
        report_c.final_val_loss.to_bits(),
        "final_val_loss не совпадает бит-в-бит: A={:.10} C={:.10}",
        report_a.final_val_loss,
        report_c.final_val_loss
    );

    // Веса первого параметра (emb) бит-в-бит
    let emb_a = ctx_a.data_f32(gpt_a.params[0].1);
    let emb_c = ctx_c.data_f32(gpt_c.params[0].1);
    assert_eq!(emb_a.len(), emb_c.len(), "длина emb не совпадает");
    for i in 0..emb_a.len() {
        assert_eq!(
            emb_a[i].to_bits(),
            emb_c[i].to_bits(),
            "emb[{}] не совпадает бит-в-бит: A={:08x} C={:08x}",
            i,
            emb_a[i].to_bits(),
            emb_c[i].to_bits()
        );
    }

    eprintln!(
        "resume_bitwise: OK — A loss={:.6}, C loss={:.6}, skipped A={}, C={}",
        report_a.final_train_loss,
        report_c.final_train_loss,
        report_a.skipped_steps,
        report_c.skipped_steps
    );
}

/// Тест 3: NaN-guard не паникует при большом LR.
#[test]
fn nan_guard_no_panic() {
    let bin_train = synth_train();
    let bin_val = synth_val();

    let mut ctx = Context::new(1 << 26);
    let cfg = GptConfig::tiny();
    let gpt = build_gpt(&mut ctx, &cfg);

    let out_dir = test_dir("nan_guard");

    let train_cfg = TrainConfig {
        lr: 1e6, // огромный LR → градиенты взрываются → NaN
        steps: 10,
        total_steps: 10,
        grad_accum: 2,
        warmup_frac: 0.02,
        warmdown_frac: 0.33,
        clip: 1.0,
        eval_every: 0,
        eval_windows: 8,
        ckpt_every: 0,
        threads: 1,
        out_dir,
        seed: 1,
    };

    let report = train(&mut ctx, &gpt, &bin_train, &bin_val, &train_cfg, None)
        .expect("train не должен паниковать при NaN");

    assert!(
        report.skipped_steps > 0,
        "ожидались пропущенные шаги (NaN), got skipped_steps={}",
        report.skipped_steps
    );

    eprintln!(
        "nan_guard: skipped_steps={}/{}, final_loss={:.6}",
        report.skipped_steps, train_cfg.steps, report.final_train_loss
    );
}

#[test]
fn invalid_resume_optimizer_state_leaves_weights_unchanged() {
    use ggrs_model::{save_checkpoint, CheckpointExtra};

    let dir = test_dir("invalid_resume_state");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("invalid.ggrs");
    let mut source = Context::new(1 << 26);
    let cfg = GptConfig::tiny();
    let source_gpt = build_gpt(&mut source, &cfg);
    let named: Vec<_> = source_gpt.params.iter().map(|(name, id)| (name.as_str(), *id)).collect();
    let valid_opt: Vec<_> = named.iter().map(|&(name, id)| {
        let n = source.t(id).nelements();
        (name.to_string(), vec![0.0; n], vec![0.0; n])
    }).collect();
    let bin_train = synth_train();
    let bin_val = synth_val();
    let train_cfg = TrainConfig {
        steps: 1,
        total_steps: 1,
        grad_accum: 1,
        lr: 1e-3,
        warmup_frac: 0.0,
        warmdown_frac: 0.0,
        clip: 1.0,
        eval_every: 0,
        eval_windows: 1,
        ckpt_every: 0,
        threads: 1,
        out_dir: dir.join("output"),
        seed: 1,
    };

    for case in ["missing", "extra", "name", "m_length", "v_length"] {
        let mut opt = valid_opt.clone();
        match case {
            "missing" => { opt.pop(); }
            "extra" => { opt.push(opt[0].clone()); }
            "name" => { opt[0].0 = "wrong_name".to_string(); }
            "m_length" => { opt[0].1.pop(); }
            "v_length" => { opt[0].2.pop(); }
            _ => unreachable!(),
        }
        let extra = CheckpointExtra { step: 0, rng: 0, opt };
        save_checkpoint(&path, &source, &named, &extra).unwrap();

        let mut target = Context::new(1 << 26);
        let target_gpt = build_gpt(&mut target, &cfg);
        for &(_, id) in &target_gpt.params {
            target.data_f32_mut(id).fill(7.0);
        }
        let error = train(
            &mut target, &target_gpt, &bin_train, &bin_val, &train_cfg, Some(&path),
        ).err().expect("invalid optimizer state must return an error");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData, "{case}");
        for (name, id) in &target_gpt.params {
            assert!(target.data_f32(*id).iter().all(|&x| x == 7.0), "{case}: {name} changed");
        }
        assert!(!train_cfg.out_dir.exists());
    }
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
}
