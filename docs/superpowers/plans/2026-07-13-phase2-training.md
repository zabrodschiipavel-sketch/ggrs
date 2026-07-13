# Фаза 2: обучение — backward, OutProd, gradcheck, AdamW — Implementation Plan

> **For agentic workers:** исполнение по docs/process/workforce.md (flash/flash:thinking/pro через dev_agent.py, ревью — архитектор). Каждая задача АТОМАРНА: свой тест-цикл, свой коммит, независимая проверка ревьюером. Чекбоксы `- [ ]`.

**Goal:** Полный тренировочный цикл на CPU: backward-граф по ggml-модели (градиенты как узлы того же графа), gradcheck конечными разностями для каждой операции, AdamW с клиппингом/NaN-guard/LR-расписанием (warmdown 33% — дефолт из исследования) и крючком error-feedback (ECO, ставка №2). Веха: мини-LM (embeddings+2×mulmat+silu+CE) сходится с ln(8) до <0.2 на циклическом паттерне токенов.

**Architecture:** `build_backward(ctx, forward_graph, loss) -> Backward { grads: HashMap<TensorId, TensorId> }` — обратный проход по узлам forward-графа; каждый вклад градиента строится СУЩЕСТВУЮЩИМИ билдерами (add/mul/scale/mul_mat/out_prod/*_back-ядра); аккумуляция вкладов — узлом Add. Оптимизатор — вне графа: читает grads по data_f32, пишет в параметры.

## Global Constraints

- Атомарность: одна задача = один аспект = свой набор тестов = свой коммит. Задача, трогающая >3 файлов, должна быть возвращена архитектору на распил.
- Backward Фазы 2 — только 2D для MulMat (3D/multi-head — Фаза 3, assert с понятным сообщением).
- Broadcast в Add/Mul backward НЕ поддерживаем (assert same_shape) — расширение в Фазе 3 (маска+bias могут подождать).
- Каждая операция с backward обязана пройти gradcheck: |analytic − FD| / (|FD|+1e-4) < 2e-2 при eps=1e-3, f32.
- Gradcheck-харнесс исполняет граф ЗАНОВО через compute после каждого шевеления входа (пересборка не нужна — данные меняются in-place, граф статичен).
- Zero deps; clippy -D warnings; тесты обязаны ИСПОЛНЯТЬ графы.
- Ветка phase-2-training; коммит после каждой задачи с меткой исполнителя.

## Интерфейсы фазы (фиксированы, использовать дословно)

```rust
// context.rs
pub fn set_param(&mut self, id: TensorId);              // is_param = true
pub fn out_prod(&mut self, x: TensorId, y: TensorId) -> TensorId; // см. T2

// новый crates/ggrs-core/src/backward.rs
pub struct Backward { pub grads: std::collections::HashMap<TensorId, TensorId> }
pub fn build_backward(ctx: &mut Context, gf: &Graph, loss: TensorId) -> Backward;
// контракт: grads[loss] — тензор из единиц формы loss; для каждого узла с
// is_param или лежащего на пути к параметрам — grads[id] заполнен; view-узлы
// (Reshape/Permute) прозрачны (T9). Расширенный граф исполняется build_forward
// от каждого grads[param] ИЛИ одним графом: build_forward(ctx, объединённый
// корень) — см. T1: вводим ctx.collect(vec![...]) -> TensorId (op=Collect,
// no-op ядро, до 4 src) для склейки корней. НЕТ: проще — Backward хранит
// также pub root: TensorId — цепочка Collect-узлов над всеми grads параметров.

// новый crates/ggrs-core/src/optim.rs (T7)
pub struct AdamW {
    pub lr: f32, pub beta1: f32, pub beta2: f32, pub eps: f32, pub wd: f32,
    pub clip_global_norm: f32,           // 0.0 = выключен
    state: Vec<(TensorId, Vec<f32>, Vec<f32>)>, // (param, m, v)
    t: u64,
}
impl AdamW {
    pub fn new(params: &[TensorId], ctx: &Context, lr: f32) -> Self; // дефолты: 0.9/0.999/1e-8/0.0/1.0
    /// Возвращает (grad_global_norm, применённый шаг был ли пропущен из-за NaN)
    pub fn step(&mut self, ctx: &mut Context, grads: &Backward) -> (f32, bool);
}
pub struct LrSchedule { pub base: f32, pub warmup_frac: f32, pub warmdown_frac: f32 } // дефолт warmdown 0.33 (arXiv 2605.25966)
impl LrSchedule { pub fn at(&self, step: u64, total: u64) -> f32 } // линейный warmup → плато → линейный warmdown до 0
```

---

### T1 (flash:thinking): backward-инфраструктура + Add/Mul/Scale + gradcheck-харнесс

Files: `backward.rs` (новый), `context.rs` (set_param, collect, ones-заливка), `op.rs` (Op::Collect), `kernels/mod.rs` (Collect => no-op), тест `tests/backward_basic.rs`.

Backward-правила задачи (аккумуляция: если grads[src] уже есть — grads[src] = ctx.add(старый, вклад)):
- Add(a,b): ∂a += g; ∂b += g (assert same_shape в build_backward).
- Mul(a,b): ∂a += mul(g, b); ∂b += mul(g, a).
- Scale(a,s): ∂a += scale(g, s).
- Op::None/входы: остановка.
Gradcheck-харнесс (в тесте, pub-хелпер не нужен): fn gradcheck(ctx, loss, param, eps) — для каждого элемента param: x±eps → два compute полного forward-графа → FD; сравнить с data_f32(grads[param]) после compute(backward.root).
Тест: z = scale(mul(add(a,b), c), 0.5), loss = «сумма» — ВНИМАНИЕ: суммирующего op нет; для скалярного лосса в тестах T1 использовать cross_entropy_loss? Он ещё без backward. Решение T1: ввести Op::SumAll (простое ядро: dst[1] = Σ всех элементов src, ith==0; backward: ∂src += repeat(g)... repeat-а нет — backward SumAll: вклад = scale(ones_like(src), g_value)? g_value — тензор [1]; НЕТ доступа к значению при построении графа. Правильно: backward SumAll = ctx.sum_all_back(g, src_ne) — новый op SumAllBack, ядро: заполнить dst значением g[0]. Оба ядра тривиальны. Это добавляет 2 op — принято, они нужны и для нормировок в будущем.
Gradcheck: a,b,c из util::Lcg, формы [6,3].

### T2 (flash): OutProd — forward

Files: `op.rs` (+OutProd), `context.rs` (билдер), `kernels/outprod.rs` (новый), `kernels/mod.rs`, тест `tests/ops_outprod.rs`.
Семантика (фиксирована): x ne=[Dx, R], y ne=[Dy, R] (2D, F32) → dst ne=[Dx, Dy]; dst[ix, iy] = Σ_{r=0..R} x[ix, r] * y[iy, r]. Параллелизм по строкам dst (iy), внутри — vec_dot НЕЛЬЗЯ (страйды по r не единичные): наивный цикл с аккумулятором f32 (SIMD — Фаза 6). Тест: против тройного цикла на Lcg-данных [5,7]×[4,7]; и симметрия: out_prod(x,x) симметрична.

### T3 (flash:thinking): MulMat backward (2D)

Files: `backward.rs`, тест `tests/backward_mulmat.rs`.
Математика (наша конвенция: a ne=[K,M], b ne=[K,N], dst=g ne=[M,N]):
- ∂a = out_prod(b, g)  → ne=[K, M] ✓
- ∂b = mul_mat(cont(transpose(a)), g) → ne=[K?]: transpose(a) ne=[M,K], cont → плотный, mul_mat(aT, g): dst[i0 over K, i1 over N] = Σ_m aT[m,i0]·g[m,i1] ✓ ne=[K,N].
assert: a.ne[2..]==1 и b.ne[2..]==1, иначе panic "mulmat backward: 3D — Фаза 3".
Gradcheck: [K=4,M=3]×[K=4,N=2], лосс = SumAll(mul_mat(a,b)); проверить ∂a и ∂b.

### T4 (flash): SiluBack + GeluBack

Files: `op.rs`, `context.rs` (билдеры silu_back(g, x), gelu_back(g, x)), `kernels/elementwise.rs` (+2 ядра: dst = g ⊙ f'(x)), `backward.rs` (правила Silu/Gelu), тест `tests/backward_unary.rs`.
silu'(x) = σ(x)·(1 + x·(1−σ(x))), σ(x)=1/(1+e^−x). gelu (tanh-аппрокс): u = √(2/π)(x+0.044715x³); gelu'(x) = 0.5(1+tanh u) + 0.5x·(1−tanh²u)·√(2/π)(1+3·0.044715x²). Gradcheck обоих: цепочка SumAll(silu(mul_mat(a,b))) и отдельно gelu.

### T5 (flash): CrossEntropyLoss backward

Files: `op.rs` (+CrossEntropyLossBack), `context.rs`, `kernels/loss.rs` (+ядро: dst[v,r] = (softmax(logits)[v,r] − targets[v,r]) · g0 / nrows; g0 — скаляр grads[loss][0], передаётся тензором src[2]), `backward.rs`, тест `tests/backward_loss.rs` (gradcheck по logits на [vocab=5, rows=3], one-hot targets; targets градиента не имеют).

### T6 (flash:thinking): GetRows backward (embeddings!)

Files: `op.rs` (+GetRowsBack), `context.rs`, `kernels/copy.rs` (+ядро: dst (форма таблицы, F32) заливается нулями, затем for ir: строка dst[ids[ir]] += g[ir]; ith==0 only — ids могут повторяться, гонки недопустимы), `backward.rs` (правило: ∂table += get_rows_back(g, ids, table_ne); для F16-таблиц backward НЕ поддержан — assert, обучаемые embeddings в Фазе 2 только F32), тест: gradcheck таблицы [4, 6] с ids=[2,0,2] (повтор! аккумуляция обязана сработать), лосс = SumAll(get_rows(...)).

### T7 (flash:thinking): AdamW + клиппинг + NaN-guard + LrSchedule

Files: `optim.rs` (новый), `lib.rs`, тест `tests/optim.rs`.
step(): (1) собрать все grads параметров (data_f32), глобальная норма; (2) если !finite → (norm, true), параметры не трогать; (3) clip: если norm > clip_global_norm > 0 → масштаб; (4) AdamW классический с bias-correction: m=β1m+(1−β1)g; v=β2v+(1−β2)g²; m̂=m/(1−β1^t); v̂=v/(1−β2^t); p −= lr·(m̂/(√v̂+eps) + wd·p). Комментарий-крючок в step(): «// ECO error-feedback (ставка №2): здесь после квантованной записи p ошибка p_факт − p_идеал добавляется в m (arXiv 2601.22101); активируется в Фазе 7».
Тесты: (а) один шаг на параметре [2] с рукой посчитанными m,v,p (точные числа в тесте); (б) NaN в градиенте → параметры не изменились, skipped=true; (в) клиппинг: norm 10, clip 1 → шаг как с g/10; (г) LrSchedule: warmup 0.1/warmdown 0.33/total 100: at(0)=0? (линейный от 0), at(10)=base, at(66)=base, at(100)=0, монотонность на краях.

### T8 (pro, thinking) ВЕХА: мини-LM сходится

Files: тест `tests/train_smoke.rs` (только тест!).
Модель: vocab=8, d=16; emb=new_tensor_2d(F32,[d,vocab]) (set_param), w1 [d, 4d]? — компактно: logits = mul_mat(emb, silu(mul_mat(w1, get_rows(emb, ids)))) — tied embeddings; w1 ne=[d,d]. Данные: циклическая последовательность токенов (i+1)%8, батч из T=16 позиций: ids[t]=t%8, targets one-hot (t+1)%8. Цикл: 300 шагов, AdamW lr=0.05, LrSchedule warmdown 0.33; каждый шаг: compute(gf) → compute(backward.root) → step() → зафиксировать loss. Ассерты: loss[0] ≈ ln(8)±0.3; loss падает монотонно в среднем (loss[последний] < 0.2); нет skipped-шагов. И одна строка eprintln с итоговым loss и норм для отчёта.
ВНИМАНИЕ: статический граф → данные ids/targets переливаются set_i32/set_f32 в СУЩЕСТВУЮЩИЕ тензоры на каждом шаге не нужны (паттерн фиксированный — один батч, оверфит на нём и есть тест сходимости).

### T9 (flash:thinking): комплект backward для Фазы 3

Files: `op.rs` (+SoftMaxBack, +RmsNormBack; RopeBack — НЕ новый op: rope с op_params[2]=1 → sin со знаком минус), `kernels/rows.rs` (+2 ядра), `kernels/rope.rs` (знак), `backward.rs` (правила SoftMax/RmsNorm/Rope/Reshape/Permute/Cont), тест `tests/backward_rows.rs`.
- SoftMaxBack(g, y): dx = y ⊙ (g − rowsum(g⊙y)) — ядро построчное.
- RmsNormBack(g, x, eps): r=(mean(x²)+eps)^−1/2; dx_j = r·g_j − x_j·r³·(Σ_i g_i x_i)/ne0 — ядро построчное.
- Rope backward: ∂x = rope_back(g, pos, n_dims, base) (обратное вращение).
- Reshape: ∂src += reshape(g, src.ne); Permute: ∂src += cont(permute(g, inverse_axes)); Cont: ∂src += g, если src contiguous, иначе временно panic (кейс cont(transpose) в градиентном пути — Фаза 3 c CopyBack).
Gradcheck всех: softmax [5,3]; rms_norm [8,2] eps=1e-5; rope [4,2,3] pos=[0,1,2] (лосс SumAll; вход после reshape в 2D для SumAll — проверяет и view-backward).
Inverse_axes: perm_inv[axes[i]] = i.

### T10 (архитектор): второй круг

Полный прогон, построчное ревью каждого дифа, живой тренинг-смоук с GGRS_PROFILE=1 (профиль обучения!), обновление спеки/графа/леджера, merge в master, push.

## Definition of Done
Все op с backward прошли gradcheck; train_smoke: ln(8)→<0.2 за ≤300 шагов; clippy -D warnings; 2D-ограничения и отложенные кейсы задокументированы assert-сообщениями; телеметрия работает в тренировочном цикле.
