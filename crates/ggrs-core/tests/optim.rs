use std::collections::HashMap;

use ggrs_core::backward::Backward;
use ggrs_core::dtype::DType;
use ggrs_core::tensor::TensorId;
use ggrs_core::AdamW;
use ggrs_core::Context;
use ggrs_core::GradAccum;
use ggrs_core::LrSchedule;

/// Создать простой градиентный тензор в контексте (F32, 1D длины n).
fn make_grad(ctx: &mut Context, n: usize, vals: &[f32]) -> TensorId {
    let t = ctx.new_tensor_1d(DType::F32, n);
    ctx.set_f32(t, vals);
    t
}

/// Построить Backward с одним параметром и ручным градиентом.
fn make_backward(param: TensorId, grad: TensorId) -> Backward {
    let mut grads_map = HashMap::new();
    grads_map.insert(param, grad);
    Backward {
        grads: grads_map,
        root: grad,
    }
}

#[test]
fn adamw_single_step_math() {
    let mut ctx = Context::new(1 << 20);
    // Параметр [1.0, -2.0]
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);
    // Градиент [0.5, -0.5]
    let grad = make_grad(&mut ctx, 2, &[0.5, -0.5]);

    let backward = make_backward(param, grad);

    let mut opt = AdamW::new(&[param], &ctx, 0.1);
    opt.clip_global_norm = 0.0; // клиппинг выключен

    let (norm, nan) = opt.step(&mut ctx, &backward);
    assert!(!nan, "NaN не должен быть");

    // норма = sqrt(0.5^2 + (-0.5)^2) = sqrt(0.5) ≈ 0.70710678
    let expected_norm = (0.5_f32 * 0.5 + 0.5_f32 * 0.5).sqrt();
    assert!(
        (norm - expected_norm).abs() < 1e-6,
        "норма не совпадает: got {}, expected {}",
        norm,
        expected_norm
    );

    // Ручной расчёт с t=1:
    // beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0
    // m[0] = 0.9*0 + 0.1*0.5 = 0.05
    // v[0] = 0.999*0 + 0.001*0.25 = 0.00025
    // m_hat[0] = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
    // v_hat[0] = 0.00025 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
    // update[0] = 0.1 * (0.5 / (sqrt(0.25) + 1e-8)) ≈ 0.1 * 0.5/0.5 = 0.1
    // p[0] = 1.0 - 0.1 = 0.9
    //
    // m[1] = 0.9*0 + 0.1*(-0.5) = -0.05
    // v[1] = 0.999*0 + 0.001*0.25 = 0.00025
    // m_hat[1] = -0.05 / 0.1 = -0.5
    // v_hat[1] = 0.00025 / 0.001 = 0.25
    // update[1] = 0.1 * (-0.5 / 0.5) = -0.1
    // p[1] = -2.0 - (-0.1) = -1.9

    let result = ctx.data_f32(param);
    let expected = [0.9_f32, -1.9_f32];
    for i in 0..2 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-6,
            "param[{}]: got {}, expected {}",
            i,
            result[i],
            expected[i]
        );
    }
}

#[test]
fn adamw_nan_guard() {
    let mut ctx = Context::new(1 << 20);
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);

    // Градиент с NaN
    let grad = make_grad(&mut ctx, 2, &[f32::NAN, 0.5]);

    let backward = make_backward(param, grad);

    let mut opt = AdamW::new(&[param], &ctx, 0.1);
    opt.clip_global_norm = 0.0;

    let (norm, nan) = opt.step(&mut ctx, &backward);
    assert!(nan, "Должен быть NaN");
    assert!(norm.is_nan(), "Норма должна быть NaN");

    // Параметр не изменился
    let result = ctx.data_f32(param);
    assert!(
        (result[0] - 1.0).abs() < 1e-6,
        "param[0] изменился: {}",
        result[0]
    );
    assert!(
        (result[1] - (-2.0)).abs() < 1e-6,
        "param[1] изменился: {}",
        result[1]
    );
}

#[test]
fn adamw_clip() {
    let mut ctx = Context::new(1 << 20);
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, 2.0]);

    // Градиент [3.0, 4.0] — норма 5.0
    let grad = make_grad(&mut ctx, 2, &[3.0, 4.0]);

    let backward = make_backward(param, grad);

    let mut opt = AdamW::new(&[param], &ctx, 0.1);
    opt.clip_global_norm = 1.0; // клиппинг к норме 1

    let (norm, nan) = opt.step(&mut ctx, &backward);
    assert!(!nan, "NaN не должен быть");

    // норма до клиппинга = 5.0
    assert!((norm - 5.0).abs() < 1e-6, "норма должна быть 5.0, got {}", norm);

    // После клиппинга: scale = 1.0/5.0 = 0.2
    // g_eff = [0.6, 0.8]
    // m[0] = 0.9*0 + 0.1*0.6 = 0.06
    // v[0] = 0.999*0 + 0.001*0.36 = 0.00036
    // m_hat[0] = 0.06 / 0.1 = 0.6
    // v_hat[0] = 0.00036 / 0.001 = 0.36
    // update[0] = 0.1 * (0.6 / (sqrt(0.36) + 0)) = 0.1 * 0.6/0.6 = 0.1
    // p[0] = 1.0 - 0.1 = 0.9
    //
    // m[1] = 0.9*0 + 0.1*0.8 = 0.06... wait, 0.9*0 + 0.1*0.8 = 0.08
    // v[1] = 0.999*0 + 0.001*0.64 = 0.00064
    // m_hat[1] = 0.08 / 0.1 = 0.8
    // v_hat[1] = 0.00064 / 0.001 = 0.64
    // update[1] = 0.1 * (0.8 / (sqrt(0.64) + 0)) = 0.1 * 0.8/0.8 = 0.1
    // p[1] = 2.0 - 0.1 = 1.9

    let result = ctx.data_f32(param);
    let expected = [0.9_f32, 1.9_f32];
    for i in 0..2 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-6,
            "param[{}]: got {}, expected {}",
            i,
            result[i],
            expected[i]
        );
    }
}

#[test]
fn adamw_weight_decay() {
    let mut ctx = Context::new(1 << 20);

    // Параметр [1.0, -2.0]
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);

    // Нулевой градиент [0.0, 0.0] — изолирует wd-член
    let grad = make_grad(&mut ctx, 2, &[0.0, 0.0]);

    let backward = make_backward(param, grad);

    let mut opt = AdamW::new(&[param], &ctx, 0.1);
    opt.wd = 0.1;
    opt.clip_global_norm = 0.0; // клиппинг выключен

    let (norm, nan) = opt.step(&mut ctx, &backward);

    assert!(!nan, "NaN не должен быть при нулевом градиенте");
    assert!((norm - 0.0).abs() < 1e-6, "норма должна быть 0.0, got {}", norm);

    // При нулевом градиенте: m=0, v=0 → m_hat=0, v_hat=0
    // update = lr * (0/(sqrt(0)+eps) + wd * p) = lr * wd * p
    //        = 0.1 * 0.1 * p = 0.01 * p
    // p -= 0.01 * p → p *= (1 - 0.01) = 0.99
    // p[0] = 1.0 * 0.99 = 0.99
    // p[1] = -2.0 * 0.99 = -1.98

    let result = ctx.data_f32(param);
    let expected = [0.99_f32, -1.98_f32];
    for i in 0..2 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-6,
            "param[{}]: got {}, expected {}",
            i,
            result[i],
            expected[i]
        );
    }
}

#[test]
fn lr_schedule_shape() {
    let sched = LrSchedule {
        base: 1.0,
        warmup_frac: 0.1,
        warmdown_frac: 0.33,
    };

    let total = 100;

    // at(0) == 0.0
    let a0 = sched.at(0, total);
    assert!((a0 - 0.0).abs() < 1e-6, "at(0) != 0.0, got {}", a0);

    // at(5) == 0.5 (линейно: 5 / 10 * 1.0)
    let a5 = sched.at(5, total);
    assert!((a5 - 0.5).abs() < 1e-6, "at(5) != 0.5, got {}", a5);

    // at(10) == 1.0 (конец warmup)
    let a10 = sched.at(10, total);
    assert!((a10 - 1.0).abs() < 1e-6, "at(10) != 1.0, got {}", a10);

    // at(50) == 1.0 (плато)
    let a50 = sched.at(50, total);
    assert!((a50 - 1.0).abs() < 1e-6, "at(50) != 1.0, got {}", a50);

    // at(100) == 0.0
    let a100 = sched.at(100, total);
    assert!((a100 - 0.0).abs() < 1e-6, "at(100) != 0.0, got {}", a100);

    // at(83) между 0 и 1 и меньше at(66)
    // warmdown_start = 100 * (1 - 0.33) = 67
    // at(66) = 1.0 (плато)
    // at(83) = 1.0 * (100-83) / (100-67) = 17/33 ≈ 0.51515
    let a66 = sched.at(66, total);
    let a83 = sched.at(83, total);
    assert!(
        a83 > 0.0 && a83 < 1.0,
        "at(83) должен быть между 0 и 1, got {}",
        a83
    );
    assert!(
        a83 < a66,
        "at(83)={} должен быть меньше at(66)={}",
        a83,
        a66
    );
}

// ===== НОВЫЕ ТЕСТЫ (Фаза 3: GradAccum + step_accum + state/restore_state) =====

/// step_accum после усреднения двух микроградиентов даёт тот же результат,
/// что и step по одному среднему градиенту.
#[test]
fn step_accum_equals_step_on_mean() {
    let mut ctx = Context::new(1 << 20);

    // Параметр [1.0, -2.0]
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);

    // Два микроградиента
    let g1_vals = [0.4_f32, -0.2_f32];
    let g2_vals = [0.6_f32, -0.8_f32];

    // Средний градиент
    let mean_vals = [0.5_f32, -0.5_f32];

    // ===== Вариант А: свежий AdamW + step по среднему градиенту =====
    let mut ctx_a = Context::new(1 << 20);
    let param_a = ctx_a.new_tensor_1d(DType::F32, 2);
    ctx_a.set_f32(param_a, &[1.0, -2.0]);

    let grad_mean_t = ctx_a.new_tensor_1d(DType::F32, 2);
    ctx_a.set_f32(grad_mean_t, &mean_vals);

    let mut opt_a = AdamW::new(&[param_a], &ctx_a, 0.1);
    opt_a.clip_global_norm = 0.0;
    opt_a.wd = 0.0;

    let bw_a = make_backward(param_a, grad_mean_t);
    let (_norm_a, _nan_a) = opt_a.step(&mut ctx_a, &bw_a);

    // ===== Вариант Б: свежий AdamW + GradAccum, add(g1), add(g2), step_accum =====
    let mut ctx_b = Context::new(1 << 20);
    let param_b = ctx_b.new_tensor_1d(DType::F32, 2);
    ctx_b.set_f32(param_b, &[1.0, -2.0]);

    let g1_t = ctx_b.new_tensor_1d(DType::F32, 2);
    ctx_b.set_f32(g1_t, &g1_vals);
    let g2_t = ctx_b.new_tensor_1d(DType::F32, 2);
    ctx_b.set_f32(g2_t, &g2_vals);

    let mut opt_b = AdamW::new(&[param_b], &ctx_b, 0.1);
    opt_b.clip_global_norm = 0.0;
    opt_b.wd = 0.0;

    let mut acc = GradAccum::new(&[param_b], &ctx_b);

    let bw1 = make_backward(param_b, g1_t);
    acc.add(&ctx_b, &bw1);

    let bw2 = make_backward(param_b, g2_t);
    acc.add(&ctx_b, &bw2);

    assert_eq!(acc.count(), 2, "счётчик должен быть 2 после двух add");

    let (_norm_b, _nan_b) = opt_b.step_accum(&mut ctx_b, &acc);

    // Параметры после А и Б равны бит-в-бит
    let result_a = ctx_a.data_f32(param_a);
    let result_b = ctx_b.data_f32(param_b);

    assert_eq!(
        result_a.len(),
        result_b.len(),
        "длины параметров не совпадают"
    );
    for i in 0..result_a.len() {
        assert_eq!(
            result_a[i].to_bits(),
            result_b[i].to_bits(),
            "параметр[{}] не совпадает бит-в-бит: A={}, B={}",
            i,
            result_a[i],
            result_b[i]
        );
    }
}

/// state/restore_state: после 3 шагов → снятие state (t, m, v и параметров) →
/// новый AdamW → restore_state + восстановление параметров → 4-й шаг в обоих —
/// параметры и (t, m, v) равны бит-в-бит.
///
/// Примечание: state/restore_state управляет только внутренним состоянием
/// оптимизатора (t, m, v). Параметры тензоров хранятся в Context — их
/// чекпоинтинг/восстановление — ответственность вызывающего кода (как в
/// реальных фреймворках PyTorch и т.п.).
#[test]
fn state_restore_bitwise_resume() {
    let mut ctx = Context::new(1 << 20);

    // Параметр [1.0, -2.0]
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);

    // Фиксированные градиенты для 4 шагов
    let grad_vals = [
        [0.5_f32, -0.3_f32],
        [0.2_f32, 0.1_f32],
        [-0.4_f32, 0.6_f32],
        [0.3_f32, -0.5_f32],
    ];

    // ===== AdamW #1: 3 шага, затем снятие state + параметров =====
    let mut ctx1 = Context::new(1 << 20);
    let param1 = ctx1.new_tensor_1d(DType::F32, 2);
    ctx1.set_f32(param1, &[1.0, -2.0]);

    let mut opt1 = AdamW::new(&[param1], &ctx1, 0.1);
    opt1.clip_global_norm = 0.0;
    opt1.wd = 0.01;

    // Шаги 1-3
    for &gv in grad_vals.iter().take(3) {
        let g_t = ctx1.new_tensor_1d(DType::F32, 2);
        ctx1.set_f32(g_t, &gv);
        let bw = make_backward(param1, g_t);
        opt1.step(&mut ctx1, &bw);
    }

    // Сохраняем state оптимизатора (t, m, v) + значения параметров
    let (t_saved, state_ref) = opt1.state();
    let state_cloned: Vec<(TensorId, Vec<f32>, Vec<f32>)> = state_ref
        .iter()
        .map(|(p, m, v)| (*p, m.clone(), v.clone()))
        .collect();
    let param_vals_saved: Vec<f32> = ctx1.data_f32(param1).to_vec();

    // 4-й шаг на opt1 (продолжаем с t=3 → t=4)
    let g4_1 = ctx1.new_tensor_1d(DType::F32, 2);
    ctx1.set_f32(g4_1, &grad_vals[3]);
    let bw4_1 = make_backward(param1, g4_1);
    let _ = opt1.step(&mut ctx1, &bw4_1);

    let params1: Vec<f32> = ctx1.data_f32(param1).to_vec();
    let (t1, state1_ref) = opt1.state();
    let t1_val = t1;
    let state1_cloned: Vec<(TensorId, Vec<f32>, Vec<f32>)> = state1_ref
        .iter()
        .map(|(p, m, v)| (*p, m.clone(), v.clone()))
        .collect();

    // ===== AdamW #2: restore_state → восстановление параметров → 4-й шаг =====
    let mut ctx2 = Context::new(1 << 20);
    let param2 = ctx2.new_tensor_1d(DType::F32, 2);
    // Восстанавливаем параметры до значений после 3 шагов
    ctx2.set_f32(param2, &param_vals_saved);

    let mut opt2 = AdamW::new(&[param2], &ctx2, 0.1);
    opt2.clip_global_norm = 0.0;
    opt2.wd = 0.01;

    // Восстанавливаем state оптимизатора (t=3, m, v)
    opt2.restore_state(t_saved, state_cloned);

    // 4-й шаг на opt2
    let g4_2 = ctx2.new_tensor_1d(DType::F32, 2);
    ctx2.set_f32(g4_2, &grad_vals[3]);
    let bw4_2 = make_backward(param2, g4_2);
    let _ = opt2.step(&mut ctx2, &bw4_2);

    let params2: Vec<f32> = ctx2.data_f32(param2).to_vec();
    let (t2, state2_ref) = opt2.state();
    let state2_cloned: Vec<(TensorId, Vec<f32>, Vec<f32>)> = state2_ref
        .iter()
        .map(|(p, m, v)| (*p, m.clone(), v.clone()))
        .collect();

    // Параметры равны бит-в-бит
    assert_eq!(params1.len(), params2.len());
    for i in 0..params1.len() {
        assert_eq!(
            params1[i].to_bits(),
            params2[i].to_bits(),
            "param[{}] не совпадает бит-в-бит после 4-го шага: {} vs {}",
            i,
            params1[i],
            params2[i]
        );
    }

    // t равны
    assert_eq!(t1_val, t2, "t не совпадают: {} vs {}", t1_val, t2);

    // (m, v) равны бит-в-бит
    assert_eq!(
        state1_cloned.len(),
        state2_cloned.len(),
        "количество параметров в state не совпадает"
    );
    for i in 0..state1_cloned.len() {
        assert_eq!(
            state1_cloned[i].0, state2_cloned[i].0,
            "TensorId на индексе {} не совпадает",
            i
        );
        assert_eq!(
            state1_cloned[i].1.len(),
            state2_cloned[i].1.len(),
            "длина m на индексе {} не совпадает",
            i
        );
        assert_eq!(
            state1_cloned[i].2.len(),
            state2_cloned[i].2.len(),
            "длина v на индексе {} не совпадает",
            i
        );
        for j in 0..state1_cloned[i].1.len() {
            assert_eq!(
                state1_cloned[i].1[j].to_bits(),
                state2_cloned[i].1[j].to_bits(),
                "m[{}][{}] не совпадает бит-в-бит: {} vs {}",
                i,
                j,
                state1_cloned[i].1[j],
                state2_cloned[i].1[j]
            );
            assert_eq!(
                state1_cloned[i].2[j].to_bits(),
                state2_cloned[i].2[j].to_bits(),
                "v[{}][{}] не совпадает бит-в-бит: {} vs {}",
                i,
                j,
                state1_cloned[i].2[j],
                state2_cloned[i].2[j]
            );
        }
    }
}

/// step_accum на свежем GradAccum без add паникует с сообщением, содержащим "count".
#[test]
#[should_panic(expected = "count")]
fn step_accum_zero_count_panics() {
    let mut ctx = Context::new(1 << 20);
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);

    let mut opt = AdamW::new(&[param], &ctx, 0.1);
    let acc = GradAccum::new(&[param], &ctx);

    // count == 0 → должна быть паника
    let _ = opt.step_accum(&mut ctx, &acc);
}

/// После reset счётчик 0, повторный цикл add → step_accum работает.
#[test]
fn grad_accum_reset() {
    let mut ctx = Context::new(1 << 20);

    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);

    let g1_t = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(g1_t, &[0.4_f32, -0.2_f32]);

    let mut opt = AdamW::new(&[param], &ctx, 0.1);
    opt.clip_global_norm = 0.0;

    let mut acc = GradAccum::new(&[param], &ctx);

    // add → reset → счётчик 0
    let bw1 = make_backward(param, g1_t);
    acc.add(&ctx, &bw1);
    assert_eq!(acc.count(), 1, "счётчик должен быть 1 после add");

    acc.reset();
    assert_eq!(acc.count(), 0, "счётчик должен быть 0 после reset");

    // Повторный цикл: add → step_accum работает
    let g2_t = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(g2_t, &[0.6_f32, -0.8_f32]);

    let bw2 = make_backward(param, g2_t);
    acc.add(&ctx, &bw2);
    assert_eq!(acc.count(), 1, "счётчик должен быть 1 после повторного add");

    let (norm, nan) = opt.step_accum(&mut ctx, &acc);
    assert!(!nan, "NaN не должен быть");
    assert!(norm.is_finite(), "норма должна быть finite");
}
