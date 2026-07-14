use crate::backward::Backward;
use crate::context::Context;
use crate::tensor::TensorId;

/// Оптимизатор AdamW с клиппингом глобальной нормы градиента и NaN-guard.
///
/// Схема:
///   1. t += 1
///   2. Сбор градиентов, вычисление глобальной нормы (f64)
///   3. NaN-guard: если норма не finite → возврат (норма, true), параметры не трогать
///   4. Клиппинг: если clip_global_norm > 0 и норма > clip_global_norm → масштаб clip/norm
///   5. Для каждого параметра: m,v → m_hat,v_hat → p -= lr*(m_hat/(sqrt(v_hat)+eps) + wd*p)
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub clip_global_norm: f32, // 0.0 = клиппинг выключен
    state: Vec<(TensorId, Vec<f32>, Vec<f32>)>, // (param, m, v)
    t: u64, // счётчик шагов, стартует с 0
}

impl AdamW {
    /// Создать AdamW с параметрами по умолчанию.
    /// beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, clip=1.0
    pub fn new(params: &[TensorId], ctx: &Context, lr: f32) -> Self {
        let mut state = Vec::with_capacity(params.len());
        for &p in params {
            let n = ctx.t(p).nelements();
            state.push((p, vec![0.0; n], vec![0.0; n]));
        }
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd: 0.0,
            clip_global_norm: 1.0,
            state,
            t: 0,
        }
    }

    /// Выполнить один шаг оптимизации.
    ///
    /// Возвращает (глобальная_норма_до_клиппинга, NaN_флаг).
    /// При NaN все параметры остаются нетронутыми (t уже инкрементирован — это ок,
    /// зафиксировано для будущего восстановления шага).
    pub fn step(&mut self, ctx: &mut Context, grads: &Backward) -> (f32, bool) {
        self.t = self.t.wrapping_add(1);

        // Шаг 2: собрать градиенты из Backward
        let grads_flat: Vec<Vec<f32>> = self
            .state
            .iter()
            .map(|&(param, _, _)| {
                let grad_id = grads.grads[&param];
                ctx.data_f32(grad_id).to_vec()
            })
            .collect();

        self.apply(ctx, &grads_flat)
    }

    /// Шаг по накопленным градиентам: g_eff = bufs / count (усреднение микробатчей).
    /// Семантика идентична step (клип по глобальной норме, NaN-guard, wd, t += 1);
    /// возвращает (grad_global_norm ДО клипа, пропущен_ли_шаг_из-за_NaN).
    ///
    /// acc должен быть создан по тому же срезу params, что и оптимизатор —
    /// соответствие TensorId проверяется assert'ом по индексам.
    ///
    /// Паника при count == 0 с внятным сообщением.
    pub fn step_accum(&mut self, ctx: &mut Context, acc: &GradAccum) -> (f32, bool) {
        assert!(
            acc.count > 0,
            "AdamW::step_accum: count == 0 — нет накопленных градиентов"
        );

        self.t = self.t.wrapping_add(1);

        // Проверка соответствия параметров
        for (i, (param, _, _)) in self.state.iter().enumerate() {
            assert_eq!(
                *param, acc.params[i],
                "AdamW::step_accum: несовпадение TensorId на индексе {}: AdamW {:?} != GradAccum {:?}",
                i, param, acc.params[i]
            );
        }

        // Усреднение микробатчей: g_eff = bufs / count
        let inv_count = 1.0 / acc.count as f32;
        let grads_flat: Vec<Vec<f32>> = acc
            .bufs
            .iter()
            .map(|buf| buf.iter().map(|&x| x * inv_count).collect())
            .collect();

        self.apply(ctx, &grads_flat)
    }

    /// Приватное ядро шага: градиенты уже собраны в grads_flat[i] для params[i].
    /// t уже инкрементирован к моменту вызова.
    fn apply(&mut self, ctx: &mut Context, grads_flat: &[Vec<f32>]) -> (f32, bool) {
        // Вычисление глобальной нормы (f64)
        let mut global_norm_sq: f64 = 0.0;
        for g in grads_flat {
            for &gi in g.iter() {
                global_norm_sq += (gi as f64) * (gi as f64);
            }
        }
        let norm = global_norm_sq.sqrt() as f32;

        // NaN-guard
        if !norm.is_finite() {
            return (norm, true);
        }

        // Клиппинг
        let scale: f32 = if self.clip_global_norm > 0.0 && norm > self.clip_global_norm {
            self.clip_global_norm / norm
        } else {
            1.0
        };

        // Bias correction
        let inv_beta1_t = 1.0 / (1.0 - self.beta1.powi(self.t as i32));
        let inv_beta2_t = 1.0 / (1.0 - self.beta2.powi(self.t as i32));

        // Обновление параметров
        for ((param, m, v), g) in self.state.iter_mut().zip(grads_flat.iter()) {
            let p_data = ctx.data_f32_mut(*param);

            for i in 0..p_data.len() {
                let gi = g[i] * scale;

                // m = beta1*m + (1-beta1)*g
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * gi;
                // v = beta2*v + (1-beta2)*g*g
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * gi * gi;

                let m_hat = m[i] * inv_beta1_t;
                let v_hat = v[i] * inv_beta2_t;

                // p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
                let update = self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.wd * p_data[i]);
                p_data[i] -= update;
            }

            // ЗАДЕЛ (ставка №2, Фаза 7): ECO error-feedback (arXiv 2601.22101) — при
            // квантованном хранении параметров здесь после записи p ошибка
            // (p_фактический − p_идеальный) инжектится обратно в момент m.
        }

        (norm, false)
    }

    /// Состояние для чекпоинта: (t, срез (param, m, v)).
    #[allow(clippy::type_complexity)]
    pub fn state(&self) -> (u64, &[(TensorId, Vec<f32>, Vec<f32>)]) {
        (self.t, &self.state)
    }

    /// Восстановление состояния (t и m/v). Паника при несовпадении длин/параметров.
    pub fn restore_state(&mut self, t: u64, mv: Vec<(TensorId, Vec<f32>, Vec<f32>)>) {
        assert_eq!(
            mv.len(),
            self.state.len(),
            "AdamW::restore_state: количество параметров не совпадает: {} vs {}",
            mv.len(),
            self.state.len()
        );
        for (i, (param, m, v)) in mv.iter().enumerate() {
            assert_eq!(
                *param, self.state[i].0,
                "AdamW::restore_state: TensorId не совпадает на индексе {}",
                i
            );
            assert_eq!(
                m.len(),
                self.state[i].1.len(),
                "AdamW::restore_state: длина m не совпадает на индексе {}",
                i
            );
            assert_eq!(
                v.len(),
                self.state[i].2.len(),
                "AdamW::restore_state: длина v не совпадает на индексе {}",
                i
            );
        }
        self.t = t;
        self.state = mv;
    }
}

/// Аккумулятор градиентов для микробатчей: буферы вне графа.
///
/// Параметры должны быть в том же порядке, что и при создании AdamW.
pub struct GradAccum {
    params: Vec<TensorId>,
    bufs: Vec<Vec<f32>>,
    count: u32,
}

impl GradAccum {
    /// Буферы нулей по форме параметров.
    pub fn new(params: &[TensorId], ctx: &Context) -> Self {
        let bufs: Vec<Vec<f32>> = params
            .iter()
            .map(|&p| vec![0.0; ctx.t(p).nelements()])
            .collect();
        GradAccum {
            params: params.to_vec(),
            bufs,
            count: 0,
        }
    }

    /// bufs[i] += data(grads.grads[params[i]]); count += 1.
    /// Паника, если у какого-то параметра нет градиента в grads.
    pub fn add(&mut self, ctx: &Context, grads: &Backward) {
        for (buf, &param) in self.bufs.iter_mut().zip(&self.params) {
            let grad_id = grads.grads[&param];
            let g = ctx.data_f32(grad_id);
            for (b, &gi) in buf.iter_mut().zip(g.iter()) {
                *b += gi;
            }
        }
        self.count += 1;
    }

    /// Обнулить буферы и счётчик.
    pub fn reset(&mut self) {
        for buf in self.bufs.iter_mut() {
            buf.fill(0.0);
        }
        self.count = 0;
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Линейное расписание скорости обучения: warmup → плато → warmdown.
///
/// at(0) = 0, линейный рост до base на warmup_frac*total шагов,
/// плато на base, затем линейный спад до 0 к total.
pub struct LrSchedule {
    pub base: f32,
    pub warmup_frac: f32,
    pub warmdown_frac: f32,
}

impl LrSchedule {
    /// Создать LrSchedule с дефолтными долями: warmup 0.02, warmdown 0.33.
    pub fn new(base: f32) -> Self {
        LrSchedule {
            base,
            warmup_frac: 0.02,
            warmdown_frac: 0.33,
        }
    }

    /// Вычислить LR на шаге `step` при общем числе шагов `total`.
    ///
    /// total > 0. Линейный warmup, плато, линейный warmdown.
    pub fn at(&self, step: u64, total: u64) -> f32 {
        assert!(total > 0, "LrSchedule::at: total должен быть > 0");

        let total_f = total as f32;
        let step_f = step as f32;

        let warmup_end = (self.warmup_frac * total_f) as u64;
        let warmdown_start = (total_f * (1.0 - self.warmdown_frac)) as u64;

        if step <= warmup_end {
            // Линейный рост от 0 до base
            if warmup_end == 0 {
                return self.base;
            }
            self.base * step_f / (warmup_end as f32)
        } else if step >= warmdown_start {
            // Линейный спад от base до 0
            let remaining = total - warmdown_start;
            if remaining == 0 {
                return 0.0;
            }
            self.base * (total - step) as f32 / (remaining as f32)
        } else {
            // Плато
            self.base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backward::Backward;
    use crate::dtype::DType;
    use std::collections::HashMap;

    #[test]
    fn adamw_single_step_math() {
        let mut ctx = Context::new(1 << 20);
        // Параметр [1.0, -2.0]
        let param = ctx.new_tensor_1d(DType::F32, 2);
        ctx.set_f32(param, &[1.0, -2.0]);
        // Градиент [0.5, -0.5]
        let grad = ctx.new_tensor_1d(DType::F32, 2);
        ctx.set_f32(grad, &[0.5, -0.5]);

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
        // норма = sqrt(0.5^2 + (-0.5)^2) = sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.70710678
        assert!((norm - (0.5_f32 * 0.5 + 0.5_f32 * 0.5).sqrt()).abs() < 1e-6,
            "норма не совпадает");

        // Ручной расчёт с t=1:
        // beta1=0.9, beta2=0.999, eps=1e-8
        // m[0] = 0.9*0 + 0.1*0.5 = 0.05
        // v[0] = 0.999*0 + 0.001*0.25 = 0.00025
        // m_hat[0] = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
        // v_hat[0] = 0.00025 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
        // update[0] = 0.1 * (0.5 / (sqrt(0.25) + 1e-8)) = 0.1 * 0.5/0.5 = 0.1
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
                i, result[i], expected[i]
            );
        }
    }

    #[test]
    fn adamw_nan_guard() {
        let mut ctx = Context::new(1 << 20);
        let param = ctx.new_tensor_1d(DType::F32, 2);
        ctx.set_f32(param, &[1.0, -2.0]);

        // Градиент с NaN
        let grad = ctx.new_tensor_1d(DType::F32, 2);
        ctx.set_f32(grad, &[f32::NAN, 0.5]);

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
        assert!((result[0] - 1.0).abs() < 1e-6, "param[0] изменился: {}", result[0]);
        assert!((result[1] - (-2.0)).abs() < 1e-6, "param[1] изменился: {}", result[1]);
    }

    #[test]
    fn adamw_clip() {
        let mut ctx = Context::new(1 << 20);
        let param = ctx.new_tensor_1d(DType::F32, 2);
        ctx.set_f32(param, &[1.0, 2.0]);

        // Градиент [3.0, 4.0] — норма 5.0
        let grad = ctx.new_tensor_1d(DType::F32, 2);
        ctx.set_f32(grad, &[3.0, 4.0]);

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
        // g_eff = [3.0*0.2, 4.0*0.2] = [0.6, 0.8]
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
                i, result[i], expected[i]
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
        assert!((sched.at(0, total) - 0.0).abs() < 1e-6, "at(0) != 0.0");

        // at(5) == 0.5 (линейно: 5 / (0.1*100=10) * 1.0 = 0.5)
        assert!((sched.at(5, total) - 0.5).abs() < 1e-6, "at(5) != 0.5");

        // at(10) == 1.0 (конец warmup)
        assert!((sched.at(10, total) - 1.0).abs() < 1e-6, "at(10) != 1.0");

        // at(50) == 1.0 (плато)
        assert!((sched.at(50, total) - 1.0).abs() < 1e-6, "at(50) != 1.0");

        // at(100) == 0.0
        assert!((sched.at(100, total) - 0.0).abs() < 1e-6, "at(100) != 0.0");

        // at(83) между 0 и 1 и меньше at(66)
        // warmdown_start = 100 * (1 - 0.33) = 67
        // at(66) = 1.0 (плато)
        // at(83) = 1.0 * (100-83) / (100-67) = 17 / 33 ≈ 0.51515
        let a66 = sched.at(66, total);
        let a83 = sched.at(83, total);
        assert!(a83 > 0.0 && a83 < 1.0, "at(83) должен быть между 0 и 1, got {}", a83);
        assert!(a83 < a66, "at(83)={} должен быть меньше at(66)={}", a83, a66);
    }
}
