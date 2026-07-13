use ggrs_core::{
    build_backward, build_forward, compute,
    util::Lcg,
    AdamW, Context, DType, LrSchedule,
};

/// Веха проекта: первое обучение сети движком ggrs.
///
/// Модель (tied embeddings, vocab=8, d=16, T=16):
///   emb [d × vocab] — таблица эмбеддингов
///   w1  [d × d]     — линейный слой
///   x = get_rows(emb, ids)       → [d × T]
///   h = mul_mat(w1, x)           → [d × T]
///   h2 = silu(h)
///   logits = mul_mat(emb, h2)    → [vocab × T]  (tied embeddings)
///   loss = cross_entropy_loss(logits, targets)
///
/// Оверфит на одном батче из 16 токенов.
#[test]
fn train_smoke() {
    const VOCAB: usize = 8;
    const D: usize = 16;
    const T: usize = 16;
    const STEPS: u64 = 300;

    let mut ctx = Context::new(1 << 24);

    // --- тензоры данных (живут в арене) ---

    // ids: индексы токенов, ids[t] = t % 8
    let ids = ctx.new_tensor_1d(DType::I32, T);
    {
        let data = ctx.data_i32_mut(ids);
        for (t, v) in data.iter_mut().enumerate() {
            *v = (t % VOCAB) as i32;
        }
    }

    // targets: one-hot размера [vocab, T], цель — класс (t+1) % 8
    let targets = ctx.new_tensor_2d(DType::F32, VOCAB, T);
    {
        let data = ctx.data_f32_mut(targets);
        data.fill(0.0);
        for t in 0..T {
            let target_class = (t + 1) % VOCAB;
            // тензор [vocab, T] — layout: row-major по ne0 (vocab), затем ne1 (T)
            // индекс: row + col * ne0  =  target_class + t * VOCAB
            data[target_class + t * VOCAB] = 1.0;
        }
    }

    // --- параметры ---

    // emb: таблица эмбеддингов [d × vocab], иниц Lcg(101)
    let emb = ctx.new_tensor_2d(DType::F32, D, VOCAB);
    ctx.set_param(emb);
    {
        let mut rng = Lcg::new(101);
        let data = ctx.data_f32_mut(emb);
        for v in data.iter_mut() {
            *v = rng.next_f32();
        }
    }

    // w1: линейный слой [d × d], иниц Lcg(102)
    let w1 = ctx.new_tensor_2d(DType::F32, D, D);
    ctx.set_param(w1);
    {
        let mut rng = Lcg::new(102);
        let data = ctx.data_f32_mut(w1);
        for v in data.iter_mut() {
            *v = rng.next_f32();
        }
    }

    // --- граф вычислений ---

    // x = get_rows(emb, ids) → [d, T]
    let x = ctx.get_rows(emb, ids);

    // h = mul_mat(w1, x) → [d, T]
    let h = ctx.mul_mat(w1, x);

    // h2 = silu(h)
    let h2 = ctx.silu(h);

    // logits = mul_mat(emb, h2) → [vocab, T]  (tied embeddings)
    let logits = ctx.mul_mat(emb, h2);

    // loss = cross_entropy_loss(logits, targets) → [1]
    let loss = ctx.cross_entropy_loss(logits, targets);

    // Строим forward-граф
    let gf = build_forward(&ctx, loss);

    // Строим backward
    let bw = build_backward(&mut ctx, &gf, loss);

    // Граф для backward (collect градиентов параметров)
    let groot = build_forward(&ctx, bw.root);

    // --- оптимизатор и расписание ---

    let mut opt = AdamW::new(&[emb, w1], &ctx, 0.0);
    let sched = LrSchedule::new(0.05);

    // --- цикл обучения ---

    let mut loss0: f32 = 0.0;
    let mut final_loss: f32 = 0.0;
    let mut final_norm: f32 = 0.0;
    let mut loss_at_150: f32 = 0.0;
    let mut any_skipped = false;

    for step in 0..STEPS {
        // Установить LR по расписанию
        opt.lr = sched.at(step, STEPS);

        // Прямой проход
        compute(&ctx, &gf, 1);

        // Фиксируем loss на шаге 0 (после compute gf, до opt.step)
        if step == 0 {
            loss0 = ctx.data_f32(loss)[0];
        }

        // Обратный проход (вычисляем градиенты)
        compute(&ctx, &groot, 1);

        // Шаг оптимизатора
        let (norm, skipped) = opt.step(&mut ctx, &bw);
        if skipped {
            any_skipped = true;
        }

        // Логируем промежуточные значения
        if step == 149 {
            loss_at_150 = ctx.data_f32(loss)[0];
        }
        if step == STEPS - 1 {
            final_loss = ctx.data_f32(loss)[0];
            final_norm = norm;
        }
    }

    // --- ассерты вехи ---

    // Шаг 0: случайная инициализация ≈ равномерное распределение → loss ≈ ln(vocab)
    let ln_vocab = (VOCAB as f32).ln();
    assert!(
        (loss0 - ln_vocab).abs() < 0.4,
        "loss0 = {:.4}, ожидалось около ln(8) = {:.4}",
        loss0,
        ln_vocab
    );

    // Финальный loss < 0.2 (оверфит на одном батче)
    assert!(
        final_loss < 0.2,
        "финальный loss = {:.4}, ожидалось < 0.2",
        final_loss
    );

    // Грубая монотонность: loss на шаге 150 < loss0
    assert!(
        loss_at_150 < loss0,
        "loss на шаге 150 = {:.4} не меньше loss0 = {:.4}",
        loss_at_150,
        loss0
    );

    // Ни одного skipped
    assert!(!any_skipped, "обнаружен skipped-шаг (NaN в градиентах)");

    // Отчёт вехи
    eprintln!(
        "train_smoke: loss {:.4} -> {:.4} за {} шагов, норма последнего градиента {:.4}",
        loss0, final_loss, STEPS, final_norm
    );
}
