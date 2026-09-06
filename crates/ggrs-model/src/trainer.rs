//! Тренировочный цикл для GPT: grad-accum, LR-schedule, чекпоинты, CSV-лог.
//!
//! Один граф forward/backward строится статически, данные меняются in-place.

use std::io::Write;
use std::path::PathBuf;

use ggrs_core::{
    build_backward, build_forward, compute, AdamW, Context, GradAccum, Graph, LrSchedule, TensorId,
};

use crate::checkpoint::{load_checkpoint_validated, save_checkpoint, CheckpointExtra};
use crate::dataset::{val_windows, TokenBin, WindowSampler};
use crate::gpt::Gpt;

/// Конфигурация тренировки.
pub struct TrainConfig {
    pub steps: u64,          // число ШАГОВ ОПТИМИЗАТОРА
    /// ИСТИННЫЙ горизонт для LR-расписания (warmup/warmdown считаются от него).
    /// При резюме между сегментами total_steps НЕ меняется (все сегменты используют
    /// одну и ту же шкалу), а steps — граница ЭТОГО вызова цикла.
    pub total_steps: u64,
    pub grad_accum: u32,     // микробатчей на шаг (дефолт 4)
    pub lr: f32,
    pub warmup_frac: f32,    // дефолт 0.02
    pub warmdown_frac: f32,  // дефолт 0.33 (arXiv 2605.25966)
    pub clip: f32,           // дефолт 1.0
    pub eval_every: u64,     // каждые N шагов оптимизатора; 0 = только в конце
    pub eval_windows: usize, // число val-окон (дефолт 8)
    pub ckpt_every: u64,     // каждые N шагов; 0 = только в конце
    pub threads: usize,
    pub out_dir: PathBuf,
    pub seed: u64,           // сид сэмплера окон
}

/// Отчёт о тренировке.
pub struct TrainReport {
    pub final_train_loss: f32, // средний train loss последнего шага оптимизатора
    pub final_val_loss: f32,
    pub tokens_seen: u64,
    pub skipped_steps: u64,    // шаги, пропущенные NaN-guard'ом
}

// ── хелперы ─────────────────────────────────────────────────────────────────

/// Заполнить один батч: токены ids, one-hot цели targets_ids.
fn set_batch(ctx: &mut Context, gpt: &Gpt, ids: &[i32], targets_ids: &[i32], vocab: usize) {
    ctx.set_i32(gpt.ids, ids);
    let t = ids.len();
    let mut tv = vec![0.0f32; vocab * t];
    for (i, &tok) in targets_ids.iter().enumerate() {
        tv[i * vocab + tok as usize] = 1.0;
    }
    ctx.set_f32(gpt.targets, &tv);
}

/// Средний loss по n детерминированным валидационным окнам.
#[allow(clippy::too_many_arguments)]
fn eval_val(
    ctx: &mut Context,
    gpt: &Gpt,
    gf: &Graph,
    bin_val: &TokenBin,
    t: usize,
    n: usize,
    vocab: usize,
    threads: usize,
) -> f32 {
    let ws = val_windows(bin_val, t, n);
    let mut sum = 0.0f64;
    for (ids, tgt) in &ws {
        set_batch(ctx, gpt, ids, tgt, vocab);
        compute(ctx, gf, threads);
        sum += ctx.data_f32(gpt.loss)[0] as f64;
    }
    (sum / n as f64) as f32
}

// ── train ────────────────────────────────────────────────────────────────────

/// Тренировочный цикл.
///
/// Граф строится один раз. Микробатчи аккумулируются через GradAccum.
/// Чекпоинты в формате GGRS1 (save_checkpoint/load_checkpoint).
///
/// Если resume = Some(путь), загружается чекпоинт (веса + шаг + состояние
/// оптимизатора + rng сэмплера) и обучение продолжается с сохранённого шага.
pub fn train(
    ctx: &mut Context,
    gpt: &Gpt,
    bin_train: &TokenBin,
    bin_val: &TokenBin,
    cfg: &TrainConfig,
    resume: Option<&std::path::Path>,
) -> std::io::Result<TrainReport> {
    let params: Vec<TensorId> = gpt.params.iter().map(|(_, id)| *id).collect();
    let vocab = ctx.t(gpt.logits).ne[0]; // размер словаря из формы logits
    let t = ctx.t(gpt.ids).ne[0];        // длина последовательности

    let mut opt = AdamW::new(&params, ctx, cfg.lr);
    opt.clip_global_norm = cfg.clip;
    let mut accum = GradAccum::new(&params, ctx);
    let sched = LrSchedule {
        base: cfg.lr,
        warmup_frac: cfg.warmup_frac,
        warmdown_frac: cfg.warmdown_frac,
    };
    let mut sampler = WindowSampler::new(cfg.seed, t);

    // Графы — один раз
    let gf = build_forward(ctx, gpt.loss);
    let bw = build_backward(ctx, &gf, gpt.loss);
    let groot = build_forward(ctx, bw.root);

    // Резюм
    let mut start_step: u64 = 0;
    if let Some(path) = resume {
        let named: Vec<(&str, TensorId)> =
            gpt.params.iter().map(|(n, id)| (n.as_str(), *id)).collect();
        let sizes: Vec<usize> = named.iter().map(|&(_, id)| ctx.t(id).nelements()).collect();
        let extra = load_checkpoint_validated(path, ctx, &named, |extra| {
            if extra.opt.len() != named.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "GGRS: число состояний оптимизатора не совпадает с числом параметров",
                ));
            }
            for (((name, _), &size), (opt_name, m, v)) in
                named.iter().zip(&sizes).zip(&extra.opt)
            {
                if name != opt_name || m.len() != size || v.len() != size {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("GGRS: состояние оптимизатора не соответствует параметру '{name}'"),
                    ));
                }
            }
            Ok(())
        })?;
        start_step = extra.step;
        sampler.set_rng_state(extra.rng);

        // Структура состояния проверена до применения весов.
        let mv: Vec<(TensorId, Vec<f32>, Vec<f32>)> = gpt
            .params
            .iter()
            .zip(extra.opt.iter())
            .map(|((_, id), (_, m, v))| (*id, m.clone(), v.clone()))
            .collect();
        opt.restore_state(extra.step, mv);
    }

    // CSV-лог
    std::fs::create_dir_all(&cfg.out_dir)?;
    let csv_path = cfg.out_dir.join("log.csv");
    let csv_exists = csv_path.exists();
    let mut csv_file: Box<dyn Write> = if resume.is_some() && csv_exists {
        Box::new(std::fs::OpenOptions::new().append(true).open(&csv_path)?)
    } else {
        let mut f = std::fs::File::create(&csv_path)?;
        writeln!(f, "step,train_loss,val_loss,lr,grad_norm,tok_per_s")?;
        Box::new(f)
    };

    // ── Основной цикл ────────────────────────────────────────────────────
    let mut last_val_loss: Option<f32> = None;
    let mut final_train_loss: f32 = 0.0;
    let mut skipped_steps: u64 = 0;
    let mut tokens_seen: u64 = 0;
    let run_start = std::time::Instant::now();

    for step in start_step..cfg.steps {
        accum.reset();
        let mut sum_loss = 0.0f64;

        for _ in 0..cfg.grad_accum {
            let (ids, tgt) = sampler.next_window(bin_train);
            set_batch(ctx, gpt, &ids, &tgt, vocab);

            compute(ctx, &gf, cfg.threads);
            sum_loss += ctx.data_f32(gpt.loss)[0] as f64;

            compute(ctx, &groot, cfg.threads);
            accum.add(ctx, &bw);
        }

        let avg_loss = (sum_loss / cfg.grad_accum as f64) as f32;
        final_train_loss = avg_loss;

        // LR по total_steps (истинный горизонт расписания), а не по cfg.steps
        opt.lr = sched.at(step + 1, cfg.total_steps);
        let (norm, skipped) = opt.step_accum(ctx, &accum);
        if skipped {
            skipped_steps += 1;
        }

        tokens_seen += (cfg.grad_accum as u64) * t as u64;

        // Кумулятивное среднее с начала ЭТОГО вызова train() (не с начала step —
        // баг был именно тут: cumulative tokens_seen делился на elapsed одного шага,
        // что линейно завышало tok/s с ростом номера шага).
        let elapsed = run_start.elapsed();
        let tok_per_s = if elapsed.as_secs_f64() > 0.0 {
            (tokens_seen as f64 / elapsed.as_secs_f64()) as f32
        } else {
            0.0
        };

        // Eval
        let eval_this = (cfg.eval_every > 0 && step % cfg.eval_every == 0) || step == cfg.steps - 1;
        let val_loss = if eval_this {
            let vl = eval_val(ctx, gpt, &gf, bin_val, t, cfg.eval_windows, vocab, cfg.threads);
            last_val_loss = Some(vl);
            vl
        } else {
            last_val_loss.unwrap_or(f32::NAN)
        };

        if eval_this {
            println!(
                "step {} loss {:.4} val {:.4} lr {:.5} norm {:.3} tok/s {:.0}",
                step, avg_loss, val_loss, opt.lr, norm, tok_per_s
            );
        }

        // CSV-строка: step,avg_train_loss,val_loss(пусто если не мерили),lr,norm,tok_per_s
        let val_str = match last_val_loss {
            Some(v) if eval_this => format!("{:.6}", v),
            Some(v) => format!("{:.6}", v),
            None => String::new(),
        };
        writeln!(
            csv_file,
            "{},{:.6},{},{:.6},{:.6},{:.0}",
            step, avg_loss, val_str, opt.lr, norm, tok_per_s
        )?;

        // Чекпоинт
        let ckpt_this =
            (cfg.ckpt_every > 0 && step % cfg.ckpt_every == 0) || step == cfg.steps - 1;
        if ckpt_this {
            let named: Vec<(&str, TensorId)> =
                gpt.params.iter().map(|(n, id)| (n.as_str(), *id)).collect();
            let (_, opt_state) = opt.state();
            let extra_opt: Vec<(String, Vec<f32>, Vec<f32>)> = gpt
                .params
                .iter()
                .zip(opt_state.iter())
                .map(|((name, _), (_, m, v))| (name.clone(), m.clone(), v.clone()))
                .collect();
            let extra = CheckpointExtra {
                step: step + 1,
                rng: sampler.rng_state(),
                opt: extra_opt,
            };
            save_checkpoint(&cfg.out_dir.join("ckpt.ggrs"), ctx, &named, &extra)?;
        }
    }

    // Финальный val loss (если не измеряли в последнем шаге — маловероятно)
    let final_val_loss = last_val_loss.unwrap_or_else(|| {
        eval_val(ctx, gpt, &gf, bin_val, t, cfg.eval_windows, vocab, cfg.threads)
    });

    Ok(TrainReport {
        final_train_loss,
        final_val_loss,
        tokens_seen,
        skipped_steps,
    })
}
