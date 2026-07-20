# ggrs — граф проекта: компоненты, зависимости, фазы

Обновлено: 2026-07-20 (Фаза 3 закрыта: ВЕХА TinyStories-10M взята).
Легенда: ✅ готово · 🔄 второй круг · ⬜ впереди · 🔴 обнаруженная скрытая зависимость.

```mermaid
flowchart TB
    subgraph P1["Фаза 1 — ядро ✅ (2-й круг ✅, ретрофит 1.5 ✅: traits/F16/телеметрия)"]
        arena["Context/арена + TensorId ✅<br/>🔄 F1: выравнивание u64<br/>🔄 F6: checked_add"]
        tensor["Tensor ne/nb + views ✅<br/>reshape/permute/transpose"]
        graph_["Граф build_forward ✅<br/>🔄 F6: итеративный DFS"]
        kernels["Forward-ядра ✅<br/>add/mul/scale/silu/gelu<br/>mulmat/softmax/rmsnorm<br/>getrows/cont/rope/xent<br/>🔄 F3: softmax -inf guard"]
        simd["SIMD AVX2/FMA ✅<br/>vec_add/mul/scale/dot"]
        threads["Потоки + барьеры ✅"]
        fixtures["numpy-эталоны ✅<br/>🔄 F2: 3D/multi-head тесты"]
        arena --> tensor --> graph_ --> kernels
        simd --> kernels
        threads --> kernels
        kernels --> fixtures
    end

    subgraph P2["Фаза 2 — обучение ✅ (2026-07-13): autograd+gradcheck, OutProd, AdamW+warmdown33, ВЕХА мини-LM 2.02→0.00"]
        outprod["🔴 OUT_PROD op<br/>(∇W у MulMat — найдено ревью F4)"]
        backward["build_backward_expand<br/>+ backward каждой op"]
        gradcheck["Gradcheck конечными<br/>разностями (test-grad0)"]
        adamw["AdamW фьюзнутый AVX2<br/>+ clip по норме + NaN-guard"]
        mlp["Веха: MLP сходится<br/>на игрушечной задаче"]
        outprod --> backward --> gradcheck --> adamw --> mlp
    end

    subgraph P3["Фаза 3 — модель ✅ (2026-07-20): BPE 4096, GPT multi-head, тренер (побитовый резюм), сэмплер; ВЕХА: TinyStories-10M за 43.7ч, val 8.32→1.78, связные истории — бейзлайн speedrun.md"]
        bpe["BPE-токенизатор<br/>(словарь 4096)"]
        gpt["GPT ~10M: RMSNorm+RoPE<br/>+SwiGLU, multi-head"]
        dataload["Датасет: предтокенизация<br/>+ mmap батчей"]
        shake["Смоук: char-Шекспир"]
        tiny["ВЕХА: TinyStories 10M<br/>связные истории ≤48ч"]
        bpe --> dataload --> shake --> tiny
        gpt --> shake
    end

    subgraph P4["Фаза 4 — GGUF ⬜ (next)"]
        gguf_w["GGUF-запись чекпоинтов"]
        gguf_r["GGUF-чтение"]
        compat["ВЕХА: наш GGUF грузится<br/>в настоящий llama.cpp"]
        gguf_w --> compat
        gguf_r --> compat
    end

    subgraph P5["Фаза 5 — квантование/inference ⬜"]
        f16["dtype F16/BF16"]
        q80["Q8_0 → Q4_0 → Q4_K"]
        viewoff["🔴 view-с-оффсетом<br/>(нужен KV-кэшу — найдено ревью)"]
        kv["KV-кэш + генерация"]
        foreign["ВЕХА: inference чужих GGUF<br/>(Qwen-0.5B с диска)"]
        f16 --> q80 --> foreign
        viewoff --> kv --> foreign
    end

    subgraph P6["Фаза 6 — производительность ⬜"]
        bench["ggrs-bench vs llama.cpp"]
        par_a["F5: mulmat параллель<br/>по строкам a"]
        fuse["F6: фьюз scale,<br/>инкрементальный theta в rope,<br/>exp/tanh в AVX2"]
        galloc["Планировщик памяти графа<br/>(аналог ggml_gallocr)"]
        goal["ЦЕЛЬ: ≥50% скорости llama.cpp"]
        bench --> par_a --> goal
        bench --> fuse --> goal
        galloc --> goal
    end

    subgraph P7["Фаза 7 — исследования (из доков) ⬜"]
        lora["LoRA/DoRA-файнтюн"]
        muon["Muon (Newton-Schulz)"]
        dpo["DPO с единым KV-кэшем<br/>(блочно-диагональная маска)"]
        qat["QAT: ∂L/∂scale,offset<br/>для Q4"]
        relora["ReLoRA/EWC<br/>continual learning"]
        lora --> relora
        muon --> qat
    end

    P1 --> P2
    P2 --> P3
    P1 --> P4
    P3 --> tiny_dep[" "]
    tiny --> P4
    P4 --> P5
    P3 --> P6
    P5 --> P6
    P2 --> P7
    P5 --> P7
    kv -.->|"единый KV-кэш"| dpo
    q80 -.->|"квант-формат"| qat
    adamw -.->|"опт-интерфейс"| muon
    gguf_w -.->|"сайдкар train_state"| P7
    backward -.->|"градиенты"| lora
    style tiny_dep fill:none,stroke:none
```

## Стратегическая цель: шум на рынке опенсорса

Три ставки (все зависят от Фаз 2–3 — backward + бейзлайн TinyStories):

| Ставка | Что это | Ниша занята? (по [исследованию](research/2026-07-13-research-base.md)) | Шанс на резонанс |
|--------|---------|--------------|------------------|
| **CPU-спидран** | Лидерборд «TinyStories до val loss X за N минут на ноутбучном CPU», формат modded-nanogpt | **Пуста** — подтверждено поиском; GPU-спидран даёт готовый бэклог трюков | Высокий |
| **Master-weight-free обучение в GGUF-квантах** | ECO-стиль (arXiv 2601.22101: ошибка реквантизации → момент) + наши F4-градиенты scale/offset для block-wise Q4_K, на CPU | Концепция доказана статьёй ECO (01.2026), **реализаций нет**: ни кода, ни block-wise, ни CPU | Средний-высокий: теория валидирована, мы — первый код |
| **Self-evolution демон** | AZR-архитектура (Proposer=Solver + код-верификатор, arXiv 2505.03335) как локальный инструмент | Наука зрелая (GPU, 3B+); инструментом — пуста; на 1–100M эффект никем не доказан | Средний |

Спидран — главная: дешёвый на нашем железе, воспроизводимый, и площадка для скрининга остальных идей. По итогам исследования Muon (вариант Polar Express) повышен до обязательного второго оптимизатора сразу после AdamW-бейзлайна: доказан от 124M (спидран) до 1T (Kimi K2, MuonClip), состоит из одних matmul — идеально для наших AVX2-ядер. Интерфейс оптимизатора Фазы 2 закладывается с крючком error-feedback под ставку №2.

Пятая фаза исследования ([sources-sweep](research/2026-07-13-sources-sweep.md)) добавила якоря:
- **[OpenAI Parameter Golf](https://github.com/openai/parameter-golf)** — официальный бенчмарк (16MB артефакт, FineWeb, лидерборд); их «выигрышные техники» = наш список (Muon WD, Polar Express, Warmdown, Z-Loss). **CPU-форк Parameter Golf — второй формат спидрана** рядом с TinyStories.
- Площадки: NeurIPS 2026 Evaluations&Datasets track (бенчмарк «CPU Training Speedrun» как сабмишн) и Competition track.
- Риски из рецензий: ECO не рецензирован (валидируем сами); Hadamard из QuEST дорог на CPU (альтернатива — FWHT-подходы типа Fast-TurboQuant); GaLore на 1–100M может не давать экономии (ранг близок к полному).
- Кандидаты в бэклог Фазы 7: ZeroQAT (forward-only градиенты — без памяти под активации), квантованный Muon (4-bit GRASP/8-bit), LC-QAT (2 бита на 0.1–10% данных), TinyStories-33M как эталонный бейзлайн с известными гиперпараметрами.

## Критические пути

1. **К первой обученной модели**: арена(F1) → OUT_PROD → backward → gradcheck → AdamW → BPE → TinyStories. Самый длинный и самый ценный путь; всё остальное может ждать.
2. **К совместимости**: GGUF-запись → загрузка в llama.cpp. Не зависит от Фазы 3 по коду, но бессмысленна без обученного чекпоинта.
3. **К inference чужих моделей**: F16 → K-кванты → view-offset → KV-кэш. Самая объёмная фаза по числу форматов.

## Реестр находок ревью (второй круг Фазы 1)

| # | Находка | Тяжесть | Куда |
|---|---------|---------|------|
| F1 | Арена vec\<u8\> — выравнивание f32 не гарантировано (UB) | Серьёзная | 2-й круг Ф1 |
| F2 | 3D mulmat / multi-head rope не тестированы | Серьёзная | 2-й круг Ф1 |
| F3 | SoftMax → NaN на строке из -inf | Средняя | 2-й круг Ф1 |
| F4 | OUT_PROD отсутствовал в плане Фазы 2 | Планирование | план Ф2 |
| F5 | MulMat параллелит по строкам b (T), а не a | Перф | Фаза 6 |
| F6 | Рекурсивный DFS; alloc без checked_add; powf в rope; scale без фьюза; нет view-offset | Мелкие | 2-й круг Ф1 / Ф6 / Ф5 |
