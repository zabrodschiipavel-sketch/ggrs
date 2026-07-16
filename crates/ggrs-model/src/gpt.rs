//! GPT-подобная модель: multi-head attention + RMSNorm + RoPE + SwiGLU,
//! tied embeddings. Билдер строит forward-граф поверх ядра ggrs-core.
//!
//! Инициализация весов — GPT-2-стиль (normal, гашение остаточных проекций).

use ggrs_core::{Context, DType, TensorId, util::Lcg};

/// Конфигурация GPT-модели.
pub struct GptConfig {
    /// Размер словаря.
    pub vocab: usize,
    /// Размерность модели (d_model).
    pub d: usize,
    /// Число голов attention.
    pub h: usize,
    /// Число слоёв.
    pub layers: usize,
    /// Длина последовательности (контекст).
    pub t: usize,
    /// Размерность скрытого слоя FFN.
    pub d_ff: usize,
    /// База для RoPE (по умолчанию 10000).
    pub rope_base: f32,
    /// Сид для инициализации весов.
    pub seed: u64,
}

impl GptConfig {
    /// Число обучаемых параметров: vocab*d + layers*(4*d*d + 3*d*d_ff).
    pub fn n_params(&self) -> usize {
        self.vocab * self.d + self.layers * (4 * self.d * self.d + 3 * self.d * self.d_ff)
    }

    /// Конфиг ~10M: vocab 4096, d 256, h 8, layers 8, t 256, d_ff 704, base 1e4, seed 1.
    pub fn d10m() -> Self {
        GptConfig {
            vocab: 4096,
            d: 256,
            h: 8,
            layers: 8,
            t: 256,
            d_ff: 704,
            rope_base: 1e4,
            seed: 1,
        }
    }

    /// Конфиг для тестов/смоука: vocab 65, d 16, h 2, layers 2, t 32, d_ff 32.
    pub fn tiny() -> Self {
        GptConfig {
            vocab: 65,
            d: 16,
            h: 2,
            layers: 2,
            t: 32,
            d_ff: 32,
            rope_base: 1e4,
            seed: 1,
        }
    }

    /// Минимальная конфигурация для сквозного gradcheck:
    /// vocab 11, d 8, h 2, layers 1, t 4, d_ff 16, seed 3.
    pub fn micro() -> Self {
        GptConfig {
            vocab: 11,
            d: 8,
            h: 2,
            layers: 1,
            t: 4,
            d_ff: 16,
            rope_base: 1e4,
            seed: 3,
        }
    }
}

/// Построенная GPT-модель: тензоры параметров, входов/выходов.
pub struct Gpt {
    /// Имена стабильны (для чекпоинтов): "emb", "l{i}.wq","l{i}.wk","l{i}.wv","l{i}.wo",
    /// "l{i}.w_up","l{i}.w_gate","l{i}.w_down".
    pub params: Vec<(String, TensorId)>,
    /// I32 [t] — входные токены, данные ставит вызывающий (set_i32).
    pub ids: TensorId,
    /// I32 [t] — позиции, build_gpt заполняет 0..t.
    pub pos: TensorId,
    /// F32 [vocab, t] — one-hot цели, ставит вызывающий.
    pub targets: TensorId,
    /// F32 [vocab, t] — логиты.
    pub logits: TensorId,
    /// F32 [1] — кросс-энтропия.
    pub loss: TensorId,
}

/// Строит forward-граф до loss включительно и инициализирует веса.
///
/// # Panics
/// Паникует, если `cfg.d` не делится на `cfg.h`.
pub fn build_gpt(ctx: &mut Context, cfg: &GptConfig) -> Gpt {
    let hd = cfg.d / cfg.h;
    assert_eq!(cfg.d % cfg.h, 0, "build_gpt: d должно делиться на h");

    let mut rng = Lcg::new(cfg.seed);
    let mut params: Vec<(String, TensorId)> = Vec::new();

    // ── Embedding ────────────────────────────────────────────────────────
    let emb = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.vocab);
    ctx.set_param(emb);
    ctx.fill_normal(emb, 0.0, 0.02, &mut rng);
    params.push(("emb".to_string(), emb));

    // ── Гашение остаточных проекций ──────────────────────────────────────
    let residual_std = 0.02 / (2.0 * cfg.layers as f32).sqrt();

    // ── Входные тензоры ──────────────────────────────────────────────────
    let ids = ctx.new_tensor_1d(DType::I32, cfg.t);
    let pos = ctx.new_tensor_1d(DType::I32, cfg.t);
    {
        let pv: Vec<i32> = (0..cfg.t as i32).collect();
        ctx.set_i32(pos, &pv);
    }
    let targets = ctx.new_tensor_2d(DType::F32, cfg.vocab, cfg.t);

    // ── Forward: начальный вход ──────────────────────────────────────────
    let mut x = ctx.get_rows(emb, ids); // [d, t]

    for l in 0..cfg.layers {
        // ── Параметры attention ──────────────────────────────────────
        let wq = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.d);
        ctx.set_param(wq);
        ctx.fill_normal(wq, 0.0, 0.02, &mut rng);
        params.push((format!("l{l}.wq"), wq));

        let wk = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.d);
        ctx.set_param(wk);
        ctx.fill_normal(wk, 0.0, 0.02, &mut rng);
        params.push((format!("l{l}.wk"), wk));

        let wv = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.d);
        ctx.set_param(wv);
        ctx.fill_normal(wv, 0.0, 0.02, &mut rng);
        params.push((format!("l{l}.wv"), wv));

        let wo = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.d);
        ctx.set_param(wo);
        ctx.fill_normal(wo, 0.0, residual_std, &mut rng);
        params.push((format!("l{l}.wo"), wo));

        // ── RMSNorm → Q/K/V ─────────────────────────────────────────
        let xn = ctx.rms_norm(x, 1e-5); // [d, t]

        let q0 = ctx.mul_mat(wq, xn); // [d, t]
        let k0 = ctx.mul_mat(wk, xn); // [d, t]
        let v0 = ctx.mul_mat(wv, xn); // [d, t]

        // ── Reshape 3D ──────────────────────────────────────────────
        let q1 = ctx.reshape_3d(q0, hd, cfg.h, cfg.t); // [hd, h, t]
        let k1 = ctx.reshape_3d(k0, hd, cfg.h, cfg.t); // [hd, h, t]
        let v1 = ctx.reshape_3d(v0, hd, cfg.h, cfg.t); // [hd, h, t]

        // ── RoPE ────────────────────────────────────────────────────
        let q2 = ctx.rope(q1, pos, hd, cfg.rope_base); // [hd, h, t]
        let k2 = ctx.rope(k1, pos, hd, cfg.rope_base); // [hd, h, t]

        // ── Permute [hd,h,t] → [hd,t,h] + cont ──────────────────────
        let q3 = ctx.permute(q2, [0, 2, 1, 3]);
        let q = ctx.cont(q3); // [hd, t, h]
        let k3 = ctx.permute(k2, [0, 2, 1, 3]);
        let k = ctx.cont(k3); // [hd, t, h]
        let v3 = ctx.permute(v1, [0, 2, 1, 3]);
        let v = ctx.cont(v3); // [hd, t, h]

        // ── Attention scores ────────────────────────────────────────
        let att0 = ctx.mul_mat(k, q); // [t, t, h]
        let scale_factor = 1.0 / (hd as f32).sqrt();
        let att1 = ctx.scale(att0, scale_factor);
        let att2 = ctx.diag_mask_inf(att1); // каузальная, батч по h
        let att = ctx.soft_max(att2); // [t, t, h]

        // ── Value aggregation ───────────────────────────────────────
        let vt0 = ctx.transpose(v); // [t, hd, h]
        let vt = ctx.cont(vt0); // CONT ОБЯЗАТЕЛЕН
        let out0 = ctx.mul_mat(vt, att); // [hd, t, h]

        // ── Обратная сборка голов ───────────────────────────────────
        let out1 = ctx.permute(out0, [0, 2, 1, 3]); // [hd, h, t]
        let out2 = ctx.cont(out1); // CONT ОБЯЗАТЕЛЕН перед reshape
        let out3 = ctx.reshape_2d(out2, cfg.d, cfg.t); // [d, t]

        let att_out = ctx.mul_mat(wo, out3); // [d, t]

        // ── Residual 1 ──────────────────────────────────────────────
        let h_res = ctx.add(x, att_out); // [d, t]

        // ── FFN (SwiGLU) ────────────────────────────────────────────
        let w_up = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.d_ff);
        ctx.set_param(w_up);
        ctx.fill_normal(w_up, 0.0, 0.02, &mut rng);
        params.push((format!("l{l}.w_up"), w_up));

        let w_gate = ctx.new_tensor_2d(DType::F32, cfg.d, cfg.d_ff);
        ctx.set_param(w_gate);
        ctx.fill_normal(w_gate, 0.0, 0.02, &mut rng);
        params.push((format!("l{l}.w_gate"), w_gate));

        let w_down = ctx.new_tensor_2d(DType::F32, cfg.d_ff, cfg.d);
        ctx.set_param(w_down);
        ctx.fill_normal(w_down, 0.0, residual_std, &mut rng);
        params.push((format!("l{l}.w_down"), w_down));

        let hn = ctx.rms_norm(h_res, 1e-5); // [d, t]
        let up = ctx.mul_mat(w_up, hn); // [d_ff, t]
        let gate = ctx.mul_mat(w_gate, hn); // [d_ff, t]
        let gs = ctx.silu(gate); // [d_ff, t]
        let ff0 = ctx.mul(gs, up); // [d_ff, t]
        let ff = ctx.mul_mat(w_down, ff0); // [d, t]

        // ── Residual 2 → вход следующего слоя ───────────────────────
        x = ctx.add(h_res, ff); // [d, t]
    }

    // ── Финальные логиты и loss (tied embeddings) ──────────────────────
    let logits = ctx.mul_mat(emb, x); // [vocab, t]
    let loss = ctx.cross_entropy_loss(logits, targets); // [1]

    Gpt {
        params,
        ids,
        pos,
        targets,
        logits,
        loss,
    }
}
