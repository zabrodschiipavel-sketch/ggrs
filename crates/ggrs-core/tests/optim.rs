use std::collections::HashMap;

use ggrs_core::backward::Backward;
use ggrs_core::dtype::DType;
use ggrs_core::tensor::TensorId;
use ggrs_core::AdamW;
use ggrs_core::Context;
use ggrs_core::LrSchedule;

/// Создать простой градиентный тензор в контексте (F32, 1D длины n).
fn make_grad(ctx: &mut Context, n: usize, vals: &[f32]) -> TensorId {
    let t = ctx.new_tensor_1d(DType::F32, n);
    ctx.set_f32(t, vals);
    t
}

#[test]
fn adamw_single_step_math() {
    let mut ctx = Context::new(1 << 20);
    // Параметр [1.0, -2.0]
    let param = ctx.new_tensor_1d(DType::F32, 2);
    ctx.set_f32(param, &[1.0, -2.0]);
    // Градиент [0.5, -0.5]
    let grad = make_grad(&mut ctx, 2, &[0.5, -0.5]);

    let mut grads_map = HashMap::new();
    grads_map.insert(param, grad);
    let backward = Backward {
        grads: grads_map,
        root: grad,
    };

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

    let mut grads_map = HashMap::new();
    grads_map.insert(param, grad);
    let backward = Backward {
        grads: grads_map,
        root: grad,
    };

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

    let mut grads_map = HashMap::new();
    grads_map.insert(param, grad);
    let backward = Backward {
        grads: grads_map,
        root: grad,
    };

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
    // m[1] = 0.9*0 + 0.1*0.8 = 0.08
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

    let mut grads_map = HashMap::new();
    grads_map.insert(param, grad);
    let backward = Backward {
        grads: grads_map,
        root: grad,
    };

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
