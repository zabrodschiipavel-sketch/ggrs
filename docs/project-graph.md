# ggrs — граф проекта: компоненты, зависимости, фазы

Обновлено: 2026-07-12 (после второго круга ревью Фазы 1).
Легенда: ✅ готово · 🔄 второй круг · ⬜ впереди · 🔴 обнаруженная скрытая зависимость.

```mermaid
flowchart TB
    subgraph P1["Фаза 1 — ядро ✅ (второй круг 🔄)"]
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

    subgraph P2["Фаза 2 — обучение ⬜"]
        outprod["🔴 OUT_PROD op<br/>(∇W у MulMat — найдено ревью F4)"]
        backward["build_backward_expand<br/>+ backward каждой op"]
        gradcheck["Gradcheck конечными<br/>разностями (test-grad0)"]
        adamw["AdamW фьюзнутый AVX2<br/>+ clip по норме + NaN-guard"]
        mlp["Веха: MLP сходится<br/>на игрушечной задаче"]
        outprod --> backward --> gradcheck --> adamw --> mlp
    end

    subgraph P3["Фаза 3 — модель ⬜"]
        bpe["BPE-токенизатор<br/>(словарь 4096)"]
        gpt["GPT ~10M: RMSNorm+RoPE<br/>+SwiGLU, multi-head"]
        dataload["Датасет: предтокенизация<br/>+ mmap батчей"]
        shake["Смоук: char-Шекспир"]
        tiny["ВЕХА: TinyStories 10M<br/>связные истории ≤48ч"]
        bpe --> dataload --> shake --> tiny
        gpt --> shake
    end

    subgraph P4["Фаза 4 — GGUF ⬜"]
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
