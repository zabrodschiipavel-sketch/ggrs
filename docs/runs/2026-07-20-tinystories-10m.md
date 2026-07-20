# ВЕХА: TinyStories-10M — первый полноразмерный претрейн на ноутбучном CPU

Дата: 2026-07-18 → 2026-07-20. Задача: T14 плана Фазы 3
(`docs/superpowers/plans/2026-07-14-phase3-model.md`), приёмка архитектора.
Прогон = **бейзлайн спидрана** (первая строка лидерборда в [speedrun.md](../speedrun.md)).

## Вердикт приёмки: ВЕХА ВЗЯТА ✅

| Критерий плана | Результат |
|---|---|
| (а) val loss монотонно падает и < цели (цель фиксируется этим прогоном) | ✅ монотонно на каждой 2000-шаговой вехе, финал **1.7764** |
| (б) generate выдаёт связные истории | ✅ с честными оговорками — см. «Оценка качества» ниже |
| (в) воспроизводимость сид+конфиг | ✅ seed=1, команда зафиксирована, движок побитово детерминирован по потокам |
| Бюджет ≤48ч | ✅ **43ч 44м** (2026-07-18 15:39 → 2026-07-20 11:23) |

## Конфигурация

- Модель: `GptConfig::d10m()` — vocab 4096, d 256, h 8, layers 8, t 256, d_ff 704,
  RoPE base 1e4, tied embeddings, RMSNorm без гаммы, без bias; **7 471 104 параметра**.
- Данные: TinyStoriesV2-GPT4 (HF `roneneldan/TinyStories`, CDLA-Sharing-1.0), BPE-4096;
  494 967 388 train / 4 996 871 val токенов (официальный valid-сплит). Подготовка и
  калибровка — [2026-07-17-tinystories-calibration.md](2026-07-17-tinystories-calibration.md).
- Тренер: steps=29 160 (=total_steps), grad_accum=4 (⇒ 1024 ток/шаг), lr=**6e-4**
  (выбран преflight-процедурой: 300 шагов, критерий loss≤5.0 — фактич. 4.77 против 5.14
  у 3e-4 на равном шаге), warmup 2%, warmdown 33% (arXiv 2605.25966), clip 1.0,
  eval_every 50 (8 val-окон), ckpt_every 200, threads 8, seed 1.
- Команда (воспроизведение бейзлайна):
  ```
  train --data-train ts_train.bin --data-val ts_val.bin --out <dir> \
        --steps 29160 --threads 8 --config d10m --lr 6e-4 --grad-accum 4 --seed 1
  ```
- Железо: AMD Ryzen 7 7730U (8C/16T, AVX2+FMA), Windows 11, питание от сети; прогон шёл
  детачем на 8 потоках, машина оставалась в обычном пользовании.

## Результат

- **final_val_loss = 1.7764**, final_train_loss = 1.7957 (шум микробатча; сглаженный
  train ~1.8-2.0 в конце).
- tokens_seen = **29 859 840** (~6.0% корпуса, одно случайное покрытие окнами — полной
  эпохи нет и не требовалось).
- Wall-clock **157 340 с ≈ 43.7 ч**, устойчивые **190 ток/с** (калибровка обещала 192 —
  сошлось; телеметрия tok_per_s к этому прогону уже исправлена, см. `8bd9d98`).
- **skipped_steps = 0** — NaN-guard не сработал ни разу за 29 160 шагов; grad norm
  стабильно 1.6–1.9 всю дорогу; `console.err` пуст. Обрывов не было, резюм не понадобился.

Кривая (вехи каждые 2000 шагов, из log.csv):

| step | train | val | lr |
|---|---|---|---|
| 0 | 8.3184 | 8.3183 | 0.00000 |
| 2000 | 3.6585 | 3.1940 | 0.00060 |
| 4000 | 3.1646 | 2.7107 | 0.00060 |
| 6000 | 2.6609 | 2.4822 | 0.00060 |
| 8000 | 2.4146 | 2.3462 | 0.00060 |
| 10000 | 2.3906 | 2.2657 | 0.00060 |
| 12000 | 2.1814 | 2.2203 | 0.00060 |
| 14000 | 2.2601 | 2.1803 | 0.00060 |
| 16000 | 2.1964 | 2.1694 | 0.00060 |
| 18000 | 2.0408 | 2.1552 | 0.00060 |
| 20000 | 2.1561 | 2.0278 | 0.00057 |
| 22000 | 2.1151 | 1.9814 | 0.00045 |
| 24000 | 1.9330 | 1.9075 | 0.00032 |
| 26000 | 1.7896 | 1.8710 | 0.00020 |
| 28000 | 2.0031 | 1.7938 | 0.00007 |
| 29159 | 1.7957 | **1.7764** | 0.00000 |

Старт 8.3184 = ln(4096) точно (санити инициализации). Плато на lr=6e-4 к шагу ~16-19k
(val ~2.16), warmdown с шага ~19 537 дожал val на −0.38 — треть всего прогресса после
старта плато; расписание warmdown-33 себя оправдало.

## Генерация: образцы (bin generate, top_k 40, seed 42, по 300 токенов)

**temp=0.7, prompt "Once upon a time":**

> Once upon a time there was a little boy named Tim. Tim loved to play with his toy gun.
> One day, he went to the park with his mom. He saw a big hill and wanted to climb it.
> As Tim climbed, he heard a tiny voice say, "Hello!" Tim looked around and saw a little
> bird stuck in the tree. The bird was scared of Tim. Tim wanted to help the bird. He
> thought of a plan to get the bird out of the tree.
> Tim went to his mom and asked, "Mom, can I help the bird?" His mom smiled and said,
> "Yes, Tim. But be gentle and don't be scared." She gave the bird a small piece of the
> bird to go away. Tim was happy and thanked the bird. […]

**temp=0.7, prompt "The little dog was sad because":**

> The little dog was sad because he could not find his toy. The dog wanted to help the
> boy, so he went to find his toy and gave it to the boy. The boy was very happy. […]
> From that day on, the boy and the dog became good friends. They played together every
> day. […] They learned that helping others can make you happy too.

**temp=0.9, prompt "Once upon a time" (фрагмент):**

> […] The bear thought about it and realized that he was not naughty or bad. From that
> day on, the bear and the squirrel were best friends. They played together every day
> and had lots of fun. And they lived happily ever after in the woods.

Полный набор (5 промптов × temp {0.7, 0.9}) — в артефактах прогона (`samples.txt` рядом
с чекпоинтом, вне репо).

## Оценка качества (честная)

**Что модель умеет** (устойчиво по всем 10 образцам):
- Жанр и регистр TinyStories: простая лексика, короткие предложения, детские сюжеты.
- Грамматика на уровне предложения почти безошибочна: времена, диалоговая пунктуация,
  атрибуция реплик («His mom smiled and said, …»).
- Нарративная арка: завязка → событие → развязка → мораль («They learned that helping
  others can make you happy too», «And they lived happily ever after»).
- Диалоги осмысленно связаны с сюжетом; выучен разделитель документов `<|endoftext|>`
  (после него начинается свежая история).

**Чего не умеет** (типично для ~10M-класса, фиксируем без прикрас):
- Дрейф имён/кореференции: история начинается про Tom — продолжается про Tim; Sara и
  Ben превращаются в Tim и Sam. Классический провал маленьких моделей.
- Локальные нелепицы в грамматически правильных предложениях: «She gave the bird a small
  piece of the bird», «built a castle with her castle», «The small bird and the small
  bird played together».
- Редкие сбои местоимений («things that are not hers» про мальчика) и логические
  самопротиворечия (история про горку: «not fun … They wanted to go on the slide»).
- temp=0.9 заметно рыхлее 0.7 (ожидаемо); 0.7 — рабочая температура для демо.

Это ровно полоса качества, которую оригинальная работа TinyStories описывает для моделей
~10M: беглый английский и простые сюжеты при нестабильной кореференции. Критерий вехи
«связные истории» — выполнен.

## Что это значит для проекта

Полный претрейн-пайплайн (BPE → pretokenize → 43-часовой train с чекпоинтами → generate)
прожил двое суток непрерывной работы на ноутбуке без единого сбоя, паники или NaN — на
движке, где каждый backward сверен конечными разностями. Бейзлайн спидрана зафиксирован
([speedrun.md](../speedrun.md)); первый кандидат на его снятие — Muon/Polar Express
(«Фаза 3.5» в бэклоге). Фаза 3 закрыта; следующая — Фаза 4 (GGUF, совместимость
с llama.cpp), для которой этот чекпоинт — первый реальный артефакт экспорта.
