# Ландшафт открытого кода (третья фаза исследования, 2026-07-13)

Провенанс: живой веб-поиск + DeepWiki (ответы, привязанные к коду репозиториев).
Вопрос: кто уже написал код там, где мы заявляем ниши.

## Матрица: обучение LLM вне GPU-мейнстрима

| Проект | Язык | Обучение | CPU | Наш вывод |
|--------|------|----------|-----|-----------|
| [llm.c](https://github.com/karpathy/llm.c) (Karpathy) | C/CUDA | Претрейн GPT-2/3 | Есть fp32-референс (~1000 строк), но: «6× медленнее PyTorch CPU», позиционирован как учебный | **Прецедент, не конкурент.** Признаём как prior art в README; наша цель — производительный движок, а не референс |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) master 2026 | C/C++ | ✅ Вернулось: `llama_opt_init/epoch` (ggml-opt), AdamW+SGD, бинарь `llama-finetune`, бэкенды через sched | Да | **Поправка нашей спеки**: «обучение выпилили» устарело — базовый файнтюн есть. Ограничения: token_embd/rope_freqs с FIXME (не обучаются), только файнтюн-сценарий, без телеметрии/квантованного обучения/претрейн-пайплайна |
| [candle](https://github.com/huggingface/candle) | Rust | Есть (llms-from-scratch-rs учит GPT на нём) | CPU-бэкенд с MKL | Rust-конкурент по обучению в целом, но GPU-first и без ggml-архитектуры/GGUF-обучения. Остаётся нашим эталоном для сверки |
| [burn](https://github.com/tracel-ai/burn) | Rust | Есть | Медленный CPU (в одном сравнении ~26× медленнее candle на шаге) | Не конкурент по CPU-перфу |
| [bitnet.cpp](https://github.com/microsoft/BitNet) | C++ | ❌ только inference тернарных моделей | Да, быстрый | Подтверждает: низкобитный CPU-**inference** занят Microsoft, **обучение** — нет |
| [QuEST](https://github.com/IST-DASLab/QuEST) | PyTorch (+CUDA/Triton) | ✅ квант-обучение (Адамар + trust estimator), конфиги ~213M Llama-стиля | Только pytorch-fallback для базовых операций | **Оракул для Фазы 7**: реализация квантователей и trust-градиентов, с которой сверяем наш backward (как numpy для forward) |
| [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt) | PyTorch | Спидран GPT-2 | ❌ 8×H100 | Формат лидерборда и трюки — заимствуем; CPU-версии нет |
| llm.c-порты на Rust: [ToJen/llm.rs](https://github.com/ToJen/llm.rs), [Steboss/llm.rust](https://github.com/Steboss/llm.rust), [yijunyu/llm.rs](https://github.com/yijunyu/llm.rs) (c2rust), [wassemgtk/llm-training-rust](https://github.com/wassemgtk/llm-training-rust) | Rust | Порт train_gpt2.c | fp32, по описаниям не оптимизированы (⚠️ проверить при бенчмарке) | **Ближайший prior art по «Rust CPU training»**: базовые порты без квантов/GGUF/оптимизаторов. Взять как бейзлайны в бенчмарк Фазы 6 рядом с llm.c |
| torch.optim.Muon (PyTorch 2.12) | PyTorch | Оптимизатор в стандартной библиотеке | Технически есть, неэффективен | Muon стал мейнстримом; Rust/CPU-эффективной реализации по-прежнему нет |

## Что это меняет

1. **Поправка в спеку**: формулировку «тренировочную часть выпилили из upstream»
   уточняем — старые train-from-scratch/finetune удалили (PR #8669), но в 2026 в
   master живёт новый ggml-opt-файнтюн. Наше отличие от llama.cpp-обучения:
   полный претрейн-пайплайн (токенизатор→данные→спидран-телеметрия), обучаемые
   embeddings, квантованное обучение (ставка №2), Muon, Rust.
2. **Позиционирование спидрана**: llm.c CPU-путь — «референс, не движок» (сам
   Карпаты: 6× медленнее PyTorch CPU); это отличная точка отсчёта для нашего
   бенчмарка Фазы 6 — обгон llm.c CPU и PyTorch CPU на равной модели и есть
   первый публичный результат.
3. **Фаза 7 получает оракула**: QuEST-репозиторий (pytorch-бэкенд) — эталон для
   gradcheck квантованного обучения.
4. Ниши ставок после проверки кодом: спидран на CPU — пусто; master-weight-free
   обучение в коде — пусто (ECO без кода, QuEST — с master-весами в fp);
   self-evolution демон — пусто.
