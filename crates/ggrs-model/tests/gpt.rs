use ggrs_core::{
    build_backward, build_forward, compute, Context, TensorId,
};
use ggrs_model::{build_gpt, GptConfig};

// ═══════════════════════════════════════════════════════════════════════════
// Вспомогательные функции
// ═══════════════════════════════════════════════════════════════════════════

/// Заполнить ids циклическим паттерном: 0,1,2,...,vocab-1,0,1,...
fn fill_ids_cyclic(ctx: &mut Context, ids: TensorId, vocab: usize) {
    let t = ctx.t(ids).ne[0];
    let vals: Vec<i32> = (0..t).map(|i| (i % vocab) as i32).collect();
    ctx.set_i32(ids, &vals);
}

/// Заполнить targets: one-hot следующего токена (wrapping vocab).
/// Для ids = [a0, a1, ..., a_{t-1}], targets[:, i] = one-hot(a_{i+1})
/// с зацикливанием: a_t ≡ a_0.
fn fill_targets_next(ctx: &mut Context, targets: TensorId, ids: TensorId, vocab: usize) {
    let t = ctx.t(ids).ne[0];
    let ids_vals = ctx.data_i32(ids);
    let mut tv = vec![0.0f32; vocab * t];
    for i in 0..t {
        let next = ids_vals[(i + 1) % t] as usize;
        tv[i * vocab + next] = 1.0;
    }
    ctx.set_f32(targets, &tv);
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 1: n_params совпадает с суммой nelements параметров
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn n_params_matches() {
    for cfg in [GptConfig::tiny(), GptConfig::d10m()] {
        let mem = 1 << 28; // 256 МиБ
        let mut ctx = Context::new(mem);
        let gpt = build_gpt(&mut ctx, &cfg);

        let nelements_sum: usize = gpt
            .params
            .iter()
            .map(|(_name, id)| ctx.t(*id).nelements())
            .sum();

        assert_eq!(
            cfg.n_params(),
            nelements_sum,
            "n_params() не совпадает с суммой nelements параметров для n_params={}",
            cfg.n_params()
        );
    }

    // Проверка, что d10m в разумном диапазоне (формула: vocab*d + layers*(4*d*d + 3*d*d_ff);
    // при d=256, vocab=4096, layers=8, d_ff=704 → ~7.5M).
    let d10m_params = GptConfig::d10m().n_params();
    assert!(
        (7_000_000..=12_000_000).contains(&d10m_params),
        "d10m n_params={} вне диапазона [7M, 12M]",
        d10m_params
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 2: forward loss в разумном диапазоне на случайных весах
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn forward_loss_in_range() {
    let cfg = GptConfig::micro();
    let mut ctx = Context::new(1 << 20);
    let gpt = build_gpt(&mut ctx, &cfg);

    fill_ids_cyclic(&mut ctx, gpt.ids, cfg.vocab);
    fill_targets_next(&mut ctx, gpt.targets, gpt.ids, cfg.vocab);

    let gf = build_forward(&ctx, gpt.loss);
    compute(&mut ctx, &gf, 1);
    let loss_val = ctx.data_f32(gpt.loss)[0];
    assert!(loss_val.is_finite(), "loss не конечен: {loss_val}");

    let ln_vocab = (cfg.vocab as f32).ln();
    assert!(
        loss_val >= 0.5 * ln_vocab && loss_val <= 1.5 * ln_vocab,
        "loss={loss_val} вне [{}, {}]",
        0.5 * ln_vocab,
        1.5 * ln_vocab
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 3: паритет вычислений между 1 и 8 потоками
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn threads_parity() {
    let cfg = GptConfig::tiny();

    // Строим два одинаковых контекста
    let mut c1 = Context::new(1 << 26);
    let gpt1 = build_gpt(&mut c1, &cfg);
    fill_ids_cyclic(&mut c1, gpt1.ids, cfg.vocab);
    fill_targets_next(&mut c1, gpt1.targets, gpt1.ids, cfg.vocab);
    let gf1 = build_forward(&c1, gpt1.loss);

    let mut c8 = Context::new(1 << 26);
    let gpt8 = build_gpt(&mut c8, &cfg);
    fill_ids_cyclic(&mut c8, gpt8.ids, cfg.vocab);
    fill_targets_next(&mut c8, gpt8.targets, gpt8.ids, cfg.vocab);
    let gf8 = build_forward(&c8, gpt8.loss);

    compute(&mut c1, &gf1, 1);
    compute(&mut c8, &gf8, 8);

    let loss1 = c1.data_f32(gpt1.loss)[0];
    let loss8 = c8.data_f32(gpt8.loss)[0];
    assert_eq!(loss1, loss8, "loss не совпадает: 1 поток={loss1}, 8 потоков={loss8}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 4: память для d10m < 2 ГиБ
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mem_used_d10m_under_2gib() {
    let cfg = GptConfig::d10m();
    let arena = 2usize << 30; // 2 ГиБ
    let mut ctx = Context::new(arena);
    let gpt = build_gpt(&mut ctx, &cfg);

    let gf = build_forward(&ctx, gpt.loss);
    let bw = build_backward(&mut ctx, &gf, gpt.loss);
    let _groot = build_forward(&ctx, bw.root);

    let used = ctx.mem_used();
    println!("d10m mem_used = {} байт ({:.2} МиБ)", used, used as f64 / (1 << 20) as f64);
    assert!(
        used < 2 * 1024 * 1024 * 1024,
        "d10m mem_used={} >= 2 ГиБ",
        used
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Тест 5: сквозной gradcheck micro-модели по ВСЕМ параметрам
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gradcheck_gpt_micro_all_params() {
    let cfg = GptConfig::micro();
    let mut ctx = Context::new(1 << 24); // 16 МиБ — micro крошечная
    let gpt = build_gpt(&mut ctx, &cfg);

    // ids: [1, 4, 1, 9] в пределах vocab=11
    ctx.set_i32(gpt.ids, &[1, 4, 1, 9]);
    fill_targets_next(&mut ctx, gpt.targets, gpt.ids, cfg.vocab);

    let gf = build_forward(&ctx, gpt.loss);

    // Вычисляем forward один раз
    compute(&mut ctx, &gf, 1);

    let bw = build_backward(&mut ctx, &gf, gpt.loss);
    let groot = build_forward(&ctx, bw.root);

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut any_fail = false;

    for (name, param_id) in &gpt.params {
        let grad_id = match bw.grads.get(param_id) {
            Some(&g) => g,
            None => {
                panic!("gradcheck: градиент для параметра '{name}' не найден в backward");
            }
        };

        let data = ctx.data_f32(*param_id).to_vec();
        for (i, &x) in data.iter().enumerate() {
            let eps = 1e-3;

            // x + eps
            ctx.data_f32_mut(*param_id)[i] = x + eps;
            compute(&mut ctx, &gf, 1);
            let l1 = ctx.data_f32(gpt.loss)[0];

            // x - eps
            ctx.data_f32_mut(*param_id)[i] = x - eps;
            compute(&mut ctx, &gf, 1);
            let l2 = ctx.data_f32(gpt.loss)[0];

            // восстановить
            ctx.data_f32_mut(*param_id)[i] = x;

            let fd = (l1 - l2) / (2.0 * eps);

            // аналитический градиент
            compute(&mut ctx, &gf, 1);
            compute(&mut ctx, &groot, 1);
            let analytic = ctx.data_f32(grad_id)[i];

            let abs_err = (analytic - fd).abs();
            let rel_err = abs_err / (fd.abs() + 1e-4);
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);

            let abs_tol = 2e-3;
            let rel_tol = 6e-2;
            if abs_err >= abs_tol && rel_err >= rel_tol {
                any_fail = true;
                eprintln!(
                    "FAIL param='{name}'[{i}] analytic={analytic:.6} fd={fd:.6} abs_err={abs_err:.6} rel_err={rel_err:.6}",
                );
            }
        }
    }

    println!(
        "gradcheck micro: max_abs_err={max_abs_err:.8} max_rel_err={max_rel_err:.8}"
    );

    assert!(
        !any_fail,
        "gradcheck_gpt_micro_all_params: некоторые параметры не прошли проверку (см. stderr)"
    );
}
