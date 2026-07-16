# Фаза 3: модель — BPE, GPT multi-head, тренер, char-Шекспир, TinyStories — Implementation Plan

> **For agentic workers:** исполнение по docs/process/workforce.md (flash/flash:thinking/pro
> через dev_agent.py + cargo-раннер; Haiku/Sonnet через Agent tool; ревью — архитектор).
> Каждая задача АТОМАРНА: свой тест-цикл, свой коммит, независимая проверка. Чекбоксы `- [ ]`.

**Goal:** Полный претрейн-пайплайн на CPU: byte-level BPE (vocab 4096), GPT ~10M
(multi-head attention, RMSNorm+RoPE+SwiGLU, tied embeddings), загрузчик данных,
тренер с grad-accumulation/чекпоинтами/телеметрией, сэмплер. Смоук — char-Шекспир
(byte-level = BPE без merges). **ВЕХА: TinyStories-10M, связные истории, ≤48ч на
ноутбучном CPU** + зафиксированный протокол спидрана (вход Фазы 6 и ставки №1).

**Architecture:** ядро (ggrs-core) дорастает до multi-head обучения четырьмя точечными
расширениями (normal-init, DiagMaskInf, 3D mulmat/out_prod backward, GradAccum+доступ
к состоянию AdamW). Всё модельное — в новом крейте `crates/ggrs-model` (zero deps,
использует ТОЛЬКО публичный API ядра): bpe / dataset / gpt / trainer / checkpoint +
четыре bin-а (train_bpe, pretokenize, train, generate). Батч — микробатчи grad-accum
вне графа (граф статичен, B=1, данные in-place — арена не растёт с батчем).

## Global Constraints

- Атомарность: одна задача = один аспект = свои тесты = свой коммит (метка исполнителя).
  Задача, трогающая >3 файлов кода, возвращается архитектору на распил.
- Zero deps — закон (mmap не берём: `std::fs::read` + окна по индексам).
- clippy `-D warnings`; тесты обязаны ИСПОЛНЯТЬ графы (build_forward/backward + compute).
- Новые backward-контракты — только через gradcheck (порог Фазы 2: |an−fd|/(|fd|+1e-4) < 2e-2, eps=1e-3).
- Многопоточная эквивалентность: каждый новый/расширенный kernel — тест 1==4 потока бит-в-бит.
- Существующие 81 тест (55 в tests/ + 26 unit в src/ на момент старта фазы) не трогать
  (кроме случаев, явно указанных в задаче). ПРИМЕЧАНИЕ: README/аудит называют «105» —
  это невоспроизводимая цифра, фактический `#[test]`-счётчик в HEAD = 81; отчёты сверять
  с реальным выводом cargo, а не с этим числом.
- Ветка `phase-3-model` от master.
- Решения фазы (зафиксированы, в брифах не пересматривать): маска — op DiagMaskInf
  (НЕ broadcast-add: Add backward ассертит same_shape — так и оставляем); RMSNorm без
  обучаемой гаммы (прецедент modded-nanogpt; broadcast-Mul backward не нужен); bias нет
  (LLaMA-стиль); rope NORM; embeddings tied (untied — бэклог спидрана); F32 обучение.

## Интерфейсы фазы (фиксированы, использовать дословно)

```rust
// ── ggrs-core ──────────────────────────────────────────────────────────────
// util.rs (T1)
impl Lcg {
    /// Box-Muller по двум next_f32 (u = x + 0.5, guard u > 0), spare отбрасывается.
    pub fn next_normal(&mut self, mean: f32, std: f32) -> f32;
}
// context.rs (T1)
pub fn fill_normal(&mut self, id: TensorId, mean: f32, std: f32, rng: &mut Lcg); // F32 contiguous
pub fn fill_uniform(&mut self, id: TensorId, lo: f32, hi: f32, rng: &mut Lcg);   // F32 contiguous
// context.rs (T4)
pub fn mem_used(&self) -> usize; // = arena_used

// context.rs + op.rs + kernels/rows.rs (T2)
/// Каузальная маска: a — [tk, tq, h(,1)]; dst[i0,i1,i2] = if i0 > i1 { f32::NEG_INFINITY } else { a[..] }.
/// backward: градиент проходит на не-маскированных, на маскированных = 0 (Op::DiagMaskInfBack не нужен —
/// то же ядро DiagMaskInf с заменой -inf на 0.0 по op_params[0]==1).
pub fn diag_mask_inf(&mut self, a: TensorId) -> TensorId;

// backward.rs + context.rs::out_prod + kernels/outprod.rs (T3)
// out_prod: снять 2D-assert → батч: x[Dx,R,B], y[Dy,R,B] → dst[Dx,Dy,B]; потоки по (iy,B); ne[3]==1.
// MulMat backward: разрешить a.ne[2]==b.ne[2] (равные батчи, БЕЗ broadcast; ne[3]==1 — assert
// с сообщением "mulmat backward: 4D/broadcast — Фаза 5/6"); формулы прежние:
// ∂a = out_prod(b, g); ∂b = mul_mat(cont(transpose(a)), g) — transpose/cont/mul_mat уже 3D-способны.

// optim.rs (T5)
pub struct GradAccum { /* params: Vec<TensorId>, bufs: Vec<Vec<f32>>, count: u32 */ }
impl GradAccum {
    pub fn new(params: &[TensorId], ctx: &Context) -> Self;
    pub fn add(&mut self, ctx: &Context, grads: &Backward); // bufs[i] += data(grads[params[i]]); count += 1
    pub fn reset(&mut self);                                // bufs = 0, count = 0
    pub fn count(&self) -> u32;
}
impl AdamW {
    /// Шаг по НАКОПЛЕННЫМ градиентам, усреднение bufs/count внутри. Семантика (клип, NaN-guard,
    /// wd, возврат (норма, пропущен)) — как у step. step и step_accum делят приватное ядро.
    pub fn step_accum(&mut self, ctx: &mut Context, acc: &GradAccum) -> (f32, bool);
    /// Доступ к состоянию для чекпоинтов: (t, [(param, m, v)]).
    pub fn state(&self) -> (u64, &[(TensorId, Vec<f32>, Vec<f32>)]);
    pub fn restore_state(&mut self, t: u64, mv: Vec<(TensorId, Vec<f32>, Vec<f32>)>);
}

// ── crates/ggrs-model (новый крейт, lib + bins) ────────────────────────────
// bpe.rs (T7)
pub struct Bpe { /* merges: Vec<(u32, u32)>, decode-таблица */ }
impl Bpe {
    pub fn train(sample: &[u8], vocab_size: u32) -> Bpe;  // 256 байтов + (vocab_size-256) merges
    pub fn encode(&self, text: &[u8]) -> Vec<u16>;
    pub fn decode(&self, ids: &[u16]) -> Vec<u8>;
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()>;
    pub fn load(path: &std::path::Path) -> std::io::Result<Bpe>;
    pub fn vocab_size(&self) -> u32;
}

// dataset.rs (T9)
pub struct TokenBin { pub tokens: Vec<u16>, pub vocab_size: u32 } // load(path) формата pretokenize (T8)
pub struct WindowSampler { /* rng: Lcg, t: usize */ }
impl WindowSampler {
    pub fn new(seed: u64, t: usize) -> Self;
    /// Случайное окно: (ids[t], targets[t]) — targets = ids сдвинутые на 1.
    pub fn next_window(&mut self, bin: &TokenBin) -> (Vec<i32>, Vec<i32>);
}
/// Детерминированные val-окна: n окон с равномерным шагом, без пересечения порядка запусков.
pub fn val_windows(bin: &TokenBin, t: usize, n: usize) -> Vec<(Vec<i32>, Vec<i32>)>;

// gpt.rs (T10)
pub struct GptConfig {
    pub vocab: usize, pub d: usize, pub h: usize, pub layers: usize,
    pub t: usize, pub d_ff: usize, pub rope_base: f32, pub seed: u64,
}
impl GptConfig {
    pub fn n_params(&self) -> usize; // vocab*d + layers*(4*d*d + 3*d*d_ff)
    pub fn d10m() -> Self;           // vocab 4096, d 256, h 8, layers 8, t 256, d_ff 704, base 1e4
    pub fn tiny() -> Self;           // vocab 256, d 64, h 2, layers 2, t 64, d_ff 128 — для тестов/смоука
}
pub struct Gpt {
    pub params: Vec<(String, TensorId)>, // имена стабильны: "emb", "l{i}.wq" ... "l{i}.w_down"
    pub ids: TensorId, pub pos: TensorId, pub targets: TensorId, // входные буферы (данные in-place)
    pub logits: TensorId, pub loss: TensorId,
}
/// Строит граф до loss включительно и инициализирует веса: normal(0, 0.02),
/// проекции в residual (wo, w_down) — normal(0, 0.02/sqrt(2*layers)). pos заполняется 0..t.
pub fn build_gpt(ctx: &mut Context, cfg: &GptConfig) -> Gpt;

// checkpoint.rs (T6) — формат GGRS1, little-endian:
// magic "GGRS" u32, version=1 u32, n_tensors u32,
// затем на тензор: name_len u32 + utf8, dtype u32 (0=F32,1=F16,2=I32), ne [u64;4], данные;
// затем extra: step u64, rng u64, n_opt u32, на параметр: name + m:[f32] + v:[f32] (длины = nelements).
pub struct CheckpointExtra { pub step: u64, pub rng: u64, pub opt: Vec<(String, Vec<f32>, Vec<f32>)> }
pub fn save_checkpoint(path: &std::path::Path, ctx: &Context,
    named: &[(&str, TensorId)], extra: &CheckpointExtra) -> std::io::Result<()>;
/// Формы/имена/dtype обязаны совпасть с named — иначе Err. Данные пишутся в существующие тензоры.
pub fn load_checkpoint(path: &std::path::Path, ctx: &mut Context,
    named: &[(&str, TensorId)]) -> std::io::Result<CheckpointExtra>;

// trainer.rs (T11)
pub struct TrainConfig {
    pub steps: u64, pub grad_accum: u32, pub lr: f32,
    pub warmup_frac: f32, pub warmdown_frac: f32,      // дефолт 0.02 / 0.33 (arXiv 2605.25966)
    pub clip: f32,                                     // дефолт 1.0
    pub eval_every: u64, pub eval_windows: usize, pub ckpt_every: u64,
    pub threads: usize, pub out_dir: std::path::PathBuf, pub seed: u64,
}
pub struct TrainReport { pub final_train_loss: f32, pub final_val_loss: f32, pub tokens_seen: u64 }
/// Цикл: окно → set_i32(ids/targets-one-hot) → compute(fwd+bwd) → GradAccum::add →
/// (каждые grad_accum) step_accum с lr из LrSchedule → лог CSV (step,loss,val,lr,norm,tok_s) →
/// eval/чекпоинт по расписанию. Резюм: --resume ckpt → шаг/оптимизатор/rng восстановлены.
pub fn train(ctx: &mut Context, gpt: &Gpt, bin_train: &TokenBin, bin_val: &TokenBin,
    cfg: &TrainConfig) -> TrainReport;
```

**Схема attention-блока** (T10, эталон — обобщение tests/transformer_forward.rs; hd = d/h):

```text
x [d,t] → rms_norm → q0,k0,v0 = mul_mat(w*, xn) [d,t]
q,k: reshape [hd,h,t] → rope(pos, n_dims=hd) → permute [hd,t,h] → cont
v:   reshape [hd,h,t] → permute [hd,t,h] → cont
att = mul_mat(k[hd,t,h], q[hd,t,h]) [t,t,h] → scale(1/√hd) → diag_mask_inf → soft_max
out = mul_mat(cont(transpose(v)) [t,hd,h], att) [hd,t,h] → permute [hd,h,t] → cont → reshape [d,t]
→ mul_mat(wo) → +x (residual) → rms_norm → SwiGLU(w_up,w_gate,w_down) → +residual
логиты: mul_mat(emb, h_final) [vocab,t] (tied) → cross_entropy_loss(one-hot targets)
```

---

### T1 (flash): normal-инициализация

**Files:** `crates/ggrs-core/src/util.rs`, `crates/ggrs-core/src/context.rs`, тест `crates/ggrs-core/tests/init.rs`.

- [ ] `Lcg::next_normal` (Box-Muller, интерфейс выше), `Context::fill_normal` / `fill_uniform`
  (assert F32 + contiguous).
- [ ] Тесты: детерминизм (два Lcg(42) → 100 одинаковых значений); статистика на 10 000 сэмплов
  N(0, 0.02): |mean| < 0.002, std в пределах ±5%; fill_uniform границы [lo, hi).
- [ ] `cargo test -p ggrs-core` зелёный; clippy чист. Commit: `feat(core): normal-init (Box-Muller) + fill_normal/fill_uniform [flash]`.

### T2 (flash:thinking): Op::DiagMaskInf + backward + gradcheck

**Files:** `crates/ggrs-core/src/op.rs`, `context.rs`, `kernels/rows.rs` (или новый `kernels/mask.rs`),
`backward.rs`, тест `tests/ops_mask.rs`.

- [ ] Ядро: по строкам (i1,i2), элементы i0 > i1 → `-inf` (forward, op_params[0]==0) или `0.0`
  (grad-режим, op_params[0]==1 — Op тот же, билдер приватного grad-режима вызывает только backward).
- [ ] backward.rs: ветка Op::DiagMaskInf — ∂a = diag_mask(g, режим 0.0).
- [ ] Тесты: forward-эталон 4×4 вручную; строка q=0 после softmax = one-hot (валиден ровно 1 элемент);
  gradcheck цепочки diag_mask→soft_max→ce_loss (FD по немаскированным ≠ 0, по маскированным = 0);
  1==4 потока бит-в-бит.
- [ ] Commit: `feat(core): DiagMaskInf (каузальная маска) + backward + gradcheck [flash:thinking]`.

### T3 (pro): 3D mulmat/out_prod backward

**Files:** `crates/ggrs-core/src/context.rs` (out_prod), `kernels/outprod.rs`, `backward.rs`,
тесты `tests/ops_outprod.rs` (расширить), `tests/backward_mulmat.rs` (расширить).

- [ ] out_prod 3D: контракт из Интерфейсов; naive-эталон в тесте; потоки по (iy, batch), 1==4 бит-в-бит.
- [ ] backward MulMat: снять 2D-assert по контракту выше (равные ne[2], assert ne[3]==1 и
  запрет broadcast с внятным сообщением).
- [ ] Gradcheck: mul_mat 3D [4,3,2]×[4,5,2] по обоим входам через sum_all-лосс; цепочка
  «attention-мини» (mul_mat 3D → scale → diag_mask → softmax → mul_mat 3D → loss) — gradcheck по q,k,v.
- [ ] Commit: `feat(core): 3D mulmat/out_prod backward + gradcheck [pro]`.

### T4 (flash): mem_used + rustdoc-долги аудита

**Files:** `crates/ggrs-core/src/context.rs`, `backward.rs` (только rustdoc), тест в `tests/init.rs` (дописать).

- [ ] `pub fn mem_used(&self)`; rustdoc на `build_backward`: «вызывать не более одного раза на ctx
  (повторный вызов дублирует узлы)» (долг аудита №4).
- [ ] Тест: mem_used монотонно растёт и учитывает выравнивание (два тензора 1 байт → смещения кратны 32).
- [ ] Commit: `feat(core): Context::mem_used + rustdoc build_backward [flash]`.

### T5 (flash:thinking): GradAccum + AdamW::step_accum + доступ к состоянию

**Files:** `crates/ggrs-core/src/optim.rs`, тест `tests/optim.rs` (дописать).

- [ ] Интерфейсы выше; step/step_accum делят приватное ядро (рефактор без изменения семантики step —
  существующие тесты optim.rs должны пройти БЕЗ правок).
- [ ] Тесты: (а) 2×add(g/… нет: add(g₁), add(g₂), step_accum == step по (g₁+g₂)/2 — сверка параметров
  бит-в-бит на ручном примере; (б) state→restore_state → следующий шаг бит-в-бит совпадает с непрерывным
  прогоном; (в) step_accum с count==0 — паника с сообщением.
- [ ] Commit: `feat(core): GradAccum, AdamW::step_accum, state()/restore_state [flash:thinking]`.

### T6 (flash): крейт ggrs-model + чекпоинты GGRS1

**Files:** `Cargo.toml` (workspace member), `crates/ggrs-model/Cargo.toml`, `src/lib.rs`,
`src/checkpoint.rs`, тест `crates/ggrs-model/tests/checkpoint.rs`.

- [ ] Каркас крейта (zero deps, зависимость только ggrs-core path); формат и функции — дословно
  из Интерфейсов.
- [ ] Тесты: round-trip f32+f16+i32 (сохранить → загрузить в свежий ctx той же структуры →
  данные бит-в-бит, extra совпал); повреждённый magic/несовпавшая форма → Err (не паника);
  файл читается после записи >4КБ (буферизация BufWriter/BufReader).
- [ ] Commit: `feat(model): крейт ggrs-model + чекпоинт-формат GGRS1 [flash]`.

### T7 (pro): byte-level BPE

**Files:** `crates/ggrs-model/src/bpe.rs`, тест `tests/bpe.rs`.

- [ ] `train`: minbpe-алгоритм (итеративно: подсчёт пар по corpus-Vec<u32> → самая частая → merge,
  перезапись вектора; ties — лексикографически меньшая пара, для детерминизма). Сэмпл ≤ 16 MiB —
  ответственность вызывающего (bin T8 сэмплирует сам).
- [ ] `encode`: жадно по рангу merges (наименьший ранг первым), чанкование по пробельным границам +
  HashMap-кэш чанк→токены (корпуса повторяются — кэш решает скорость); `decode` — конкатенация таблицы.
- [ ] save/load: текстовый формат (заголовок vocab_size; по строке на merge: `left right`).
- [ ] Тесты: encode∘decode == id на UTF-8 (русский + emoji + бинарные байты); детерминизм train;
  классика "aaabdaaabac" (vocab 259 → 3 ожидаемых merge); vocab_size==256 → encode == сами байты
  (это же — char-режим смоука T13); скорость: encode 1 MiB < 2 с в debug-профиле НЕ проверять
  (замеры — release, вне тестов).
- [ ] Commit: `feat(model): byte-level BPE train/encode/decode [pro]`.

### T8 (flash): bin-ы train_bpe + pretokenize

**Files:** `crates/ggrs-model/src/bin/train_bpe.rs`, `src/bin/pretokenize.rs`, формат .bin в
`src/dataset.rs` (заголовок: magic "GGTK" u32, vocab_size u32, n_tokens u64, далее u16-токены LE),
тест `tests/pretokenize.rs`.

- [ ] `train_bpe <corpus.txt> <vocab.bpe> [vocab_size=4096] [sample_mib=16]` — равномерный сэмпл
  из файла (каждый k-й блок 64КиБ до лимита), train, save, печать топ-20 merges.
- [ ] `pretokenize <corpus.txt> <vocab.bpe> <out_train.bin> <out_val.bin> [val_frac=0.01]` —
  потоковое чтение кусками по границам строк, encode, split хвостом (последние val_frac — в val).
- [ ] Тест: на игрушечном корпусе (fixtures, 100 КБ) — заголовок корректен, n_tokens сходится,
  decode(train.bin[..100]) — валидный UTF-8-префикс корпуса.
- [ ] Commit: `feat(model): train_bpe + pretokenize CLI, формат GGTK [flash]`.

### T9 (flash): датасет — окна

**Files:** `crates/ggrs-model/src/dataset.rs`, тест `tests/dataset.rs`.

- [ ] TokenBin::load (формат T8), WindowSampler / val_windows — интерфейсы дословно.
- [ ] Тесты: сид → детерминизм; окна в границах (fuzz 1000 окон на bin из 300 токенов, t=64);
  targets[i] == ids[i+1]; val_windows: n непересекающихся по началу окон, стабильны между вызовами.
- [ ] Commit: `feat(model): TokenBin + WindowSampler [flash]`.

### T10 (pro): GPT-билдер

**Files:** `crates/ggrs-model/src/gpt.rs`, тест `tests/gpt.rs`.
**Depends:** T1, T2, T3.

- [ ] GptConfig/Gpt/build_gpt — интерфейсы и схема блока дословно (эталон формы —
  tests/transformer_forward.rs, обобщить на h голов через 3D + DiagMaskInf; one-hot targets
  заполняет трейнер, здесь — только тензор).
- [ ] Тесты (конфиг tiny): n_params == посчитанному вручную; forward: loss конечен и в
  [0.5·ln(vocab), 1.5·ln(vocab)] на случайных весах; 1==8 потоков бит-в-бит; h=1 конфиг
  эквивалентен по построению старому тесту (лосс конечен); mem_used после build_forward+backward
  на d10m() печатается и < 2 GiB (assert — ранний сигнал долга аудита №3).
- [ ] Gradcheck НЕ здесь: покрыт по-операционно (T2, T3) — сквозной хэппи-пас проверит T11 обучением.
- [ ] Commit: `feat(model): GPT multi-head билдер + init [pro]`.

### T11 (flash:thinking): тренер

**Files:** `crates/ggrs-model/src/trainer.rs`, `src/bin/train.rs`, тест `tests/trainer.rs`.
**Depends:** T5, T6, T9, T10.

- [ ] TrainConfig/train — интерфейс и цикл дословно; CSV `out_dir/log.csv`
  (step,loss,val_loss,lr,grad_norm,tok_per_s), чекпоинт `out_dir/ckpt.ggrs` (перезапись через
  tmp+rename), консольная строка раз в eval_every.
- [ ] bin train: аргументы `--data-train --data-val --out [--resume ckpt] [--steps N] [--threads N]
  [--config d10m|tiny] [--lr X] [--grad-accum G]`.
- [ ] Тесты (tiny, синтетический bin из повторяющегося паттерна ~4К токенов): (а) 60 шагов →
  train loss упал минимум вдвое от ln(vocab); (б) резюм: 30 шагов + save → load + 30 шагов ==
  60 шагов непрерывно (loss на шаге 60 бит-в-бит; требует восстановления rng сэмплера через
  CheckpointExtra.rng); (в) NaN-guard: подсунуть lr=1e6 → шаги пропускаются, паники нет.
- [ ] Commit: `feat(model): тренер (grad-accum, LR-schedule, чекпоинты, CSV) [flash:thinking]`.

### T12 (flash): сэмплер/generate

**Files:** `crates/ggrs-model/src/sample.rs`, `src/bin/generate.rs`, тест `tests/sample.rs`.
**Depends:** T6, T7, T10.

- [ ] `pub fn sample_next(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Lcg) -> usize`
  (t=0 → argmax; softmax по f64-аккумулятору); bin generate: `--ckpt --vocab --prompt --n --temp --top-k`
  — полный forward на каждый токен (KV-кэш — Фаза 5), позиции 0..len, контекст обрезается до cfg.t.
- [ ] Тесты: t=0 — argmax детерминирован; top_k=1 == argmax; распределение при t=1 на logits
  [0, ln2] → частоты ~[1/3, 2/3] ±10% на 10 000 сэмплов.
- [ ] Commit: `feat(model): сэмплер + bin generate [flash]`.

### T13 (Sonnet): смоук char-Шекспир

**Files:** `crates/ggrs-model/tests/shakespeare_smoke.rs`, фикстура
`crates/ggrs-model/tests/fixtures/shakespeare_64k.txt` (public-domain фрагмент tinyshakespeare ≤64 КиБ,
закоммитить), `docs/runs/2026-XX-XX-shakespeare.md` (отчёт).
**Depends:** T7–T12.

- [ ] Смоук-тест (в CI): byte-level (Bpe vocab=256), конфиг tiny, 200 шагов, grad_accum 4 —
  assert: final_train_loss < 0.55·ln(256); время теста < 5 мин.
- [ ] `#[ignore]`-прогон подлиннее (2000 шагов) — запускается руками, отчёт с генерацией 200 симв.
  (видимая структура английского) в docs/runs/.
- [ ] Commit: `test(model): char-Шекспир смоук + отчёт [sonnet]`.

### T14 (архитектор): ВЕХА — TinyStories + протокол спидрана

**Files:** `docs/speedrun.md`, `docs/runs/2026-XX-XX-tinystories-10m.md`, README, project-graph.
**Depends:** всё выше. Данные: TinyStories (HF, CDLA-Sharing-1.0) скачивает Павел —
`TinyStories-train.txt` + `TinyStories-valid.txt`, пути передаются bin-ам.

- [ ] Подготовка: train_bpe (4096, сэмпл 16 MiB) → pretokenize → train d10m, замер tok/s первых
  1000 шагов → расчёт бюджета шагов под ≤48ч → полный прогон с телеметрией.
- [ ] Приёмка вехи: (а) val loss монотонно падает и < baseline-цели (цель фиксируется по факту
  бейзлайна — первый прогон И ЕСТЬ baseline спидрана); (б) generate выдаёт связные истории
  (субъективная оценка, примеры в отчёт); (в) прогон воспроизводим по сид+конфиг.
- [ ] `docs/speedrun.md` — протокол ставки №1 (формат modded-nanogpt): фикс. данные/токенизатор/
  val-набор, цель val loss, метрика wall-clock, эталонный CPU, один скрипт, таблица-лидерборд
  с первой строкой-бейзлайном. Бэклог трюков: Muon (Polar Express) — первый кандидат (Фаза 3.5),
  untied embs, QK-norm, CPU-форк Parameter Golf.
- [ ] README/project-graph: Фаза 3 ✅, веха задокументирована.

---

## Порядок и параллелизм

```
волна 1 (независимы): T1, T2, T4, T5, T6   ── flash×3, flash:thinking×2 последовательно через dev_agent
волна 2: T3 (pro), T7 (pro), T8→T9 (flash)  ── T3 после T2 (использует diag_mask в gradcheck-цепочке)
волна 3: T10 (pro) ← T1,T2,T3
волна 4: T11 (flash:thinking) ← T5,T6,T9,T10;  T12 (flash) ← T6,T7,T10
волна 5: T13 (Sonnet) → T14 (архитектор)
```
Два пишущих исполнителя параллельно НЕ работают (правило workforce); «волны» — порядок, не параллелизм.

## Риски и стоп-краны

1. **Память d10m** (долг аудита №3): оценка ~0.5 ГиБ на fwd+bwd (B=1, grad-accum держит арену
   плоской); T10 ассертит < 2 GiB. Провал → приоритет планировщику арены (из Фазы 6 в 3.5).
2. **Скорость шага**: если по замеру T14 бюджет 48ч не сходится — уменьшать модель/ctx, не мутить
   ядра (перф — Фаза 6); профиль писать compute_profiled (телеметрия уже есть).
3. **BPE encode на 2 ГБ корпуса**: кэш чанков должен дать ≥5 МиБ/с в release; провал → сэмплировать
   корпус (первый спидран не обязан видеть все токены TinyStories).
4. **DiagMaskInf + gradcheck**: маскированные позиции дают FD==analytic==0 (маска — константа
   графа); если исполнитель упрётся в inf-inf — ошибка в ядре, не в методике.
5. **Эскалация** — по workforce: 2 провала → ступень выше; 3 — бриф чинит архитектор.

## Definition of Done (Фаза 3)

- Все 81 старый + новые тесты зелёные (ubuntu+windows CI), clippy `-D warnings` чист, zero deps.
- Новые backward (DiagMaskInf, 3D mulmat/out_prod) покрыты gradcheck + 1==4 потока.
- Пайплайн работает end-to-end из CLI: train_bpe → pretokenize → train (резюм с чекпоинта
  бит-в-бит) → generate.
- Смоук char-Шекспир в CI; отчёты прогонов в docs/runs/.
- ВЕХА: TinyStories-10M обучен ≤48ч, связные истории, docs/speedrun.md с бейзлайном-лидербордом.
- README/project-graph обновлены; долги аудита №1 (чекпоинты), №2 (init), №4 (rustdoc) закрыты,
  №3 (память) — замерен и под assert.
