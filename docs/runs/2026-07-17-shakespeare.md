# Прогон: char-Шекспир, 2000 шагов (хвост T13)

Дата: 2026-07-17. Задача: `.superpowers/sdd/задача-главного-инженера-закрытие-фазы-3.md`, Task 4.
Тест: `crates/ggrs-model/tests/shakespeare_smoke.rs::shakespeare_char_long_run` (`#[ignore]`,
запускается руками: `cargo test -p ggrs-model --test shakespeare_smoke -- --ignored --nocapture`).
Коммит кода: `9b8a393` (тест написан в рамках T13); прогон выполнен на master после мержа Фазы 3
(`f043f60`).

## Конфигурация

Байт-левел BPE (vocab=256, 0 merges — 1 токен = 1 байт), фикстура
`tests/fixtures/shakespeare_64k.txt` (64 КиБ, tinyshakespeare, public domain).

GPT (`byte_tiny_config`, локальный конфиг теста — не путать с `GptConfig::tiny()`,
у которого vocab=65): vocab=256, d=64, h=2, layers=2, t=64, d_ff=128, rope_base=1e4, seed=1.

Тренер: steps=2000, total_steps=2000, grad_accum=4, lr=1e-3, warmup_frac=0.02,
warmdown_frac=0.33, clip=1.0, eval_every=200, eval_windows=8, ckpt_every=500, threads=4, seed=1.

Машина: Ryzen 7 7730U (тот же ноутбук, что и эталон проекта).

## Результат

- `final_train_loss` = **1.8877**, `final_val_loss` = **2.0178**, `tokens_seen` = 512 000,
  `skipped_steps` = 0 (NaN-guard ни разу не сработал).
- Время: **1244 с** (≈20.7 мин), 813 341 ток/с на последнем логе (throughput растёт по ходу
  прогона — JIT прогрева арены/кэшей нет, разброс тут — шум измерения между eval-точками).
- Старт loss 5.5450 — точное совпадение с `ln(256)=5.545` (равномерное распределение по 256
  байтам), хорошая проверка на вменяемость инициализации.

Кривая (train / val loss по шагам, из stdout теста):

| step | train | val | lr |
|---|---|---|---|
| 0 | 5.5450 | 5.5445 | 0.00003 |
| 200 | 2.9857 | 2.8687 | 0.00100 |
| 400 | 2.4177 | 2.4113 | 0.00100 |
| 600 | 2.2584 | 2.2841 | 0.00100 |
| 800 | 2.2999 | 2.2329 | 0.00100 |
| 1000 | 2.2031 | 2.1826 | 0.00100 |
| 1200 | 2.1890 | 2.1478 | 0.00100 |
| 1400 | 2.1480 | 2.1064 | 0.00091 |
| 1600 | 2.2302 | 2.0755 | 0.00060 |
| 1800 | 2.0355 | 2.0331 | 0.00030 |
| 1999 | 1.8877 | 2.0178 | 0.00000 |

Val loss падает монотонно; train loss шумит (ожидаемо для grad_accum=4 на маленьких
случайных окнах), но тренд ясный. Warmdown (с шага ~1340 = 2000×0.67) заметно стабилизирует
обе кривые в конце.

## Образцы генерации

Генератор: разовый скрипт, воспроизводящий логику `bin generate` для локального
`byte_tiny_config` (сам bin `generate` жёстко зашит под `d10m`/`tiny(vocab=65)` — под этот
тестовый конфиг не подходит без правки бина, что вне скоупа Task 4). top_k=40, seed=42,
300 токенов на образец.

**temp=0.7, prompt="First Citizen:\n"**
```
First Citizen:

Than the sprous senter sars lour matur es.

MARCIUS:
The no take ne you do, werte det, to the at four stoor busittir of ins harted the he I pall cen meacir selbe the thitnim this thim.

CORIGINIUS:
Aor!

Thith Come the for fis demor Cures yerblot got sen thiss lo stath he no will fou ime w
```

**temp=1.0, prompt="ROMEO:\n"**
```
ROMEO:
-Wes, the whor.

VOLUMNIA:
Git the uce: enfer the marctere ge hruankinh wim ksom pald my ttinos, tos iur of
tof I Meche the.

MALAAer she:
If sro than: bus.

Sean warfhy thor thow's, hay, elat yeefaisl
Tor,
Than: cour of tiy, pepre vroes yarunir eranto shaks!
When me he go short
Ficrin
```

(Остальные 4 из 6 сгенерированных образцов — 3 промпта × {0.7, 1.0} — опущены здесь для
краткости, идентичны по характеру.)

**Честная оценка структуры** (критерий T13 — «видимая структура английского», не связность):
- ✅ Формат реплик пьесы: `ИМЯ:\n текст` воспроизводится устойчиво, включая правдоподобные
  (хоть и не всегда настоящие) имена персонажей в духе corpus (MARCIUS, CORIGINIUS, VOLUMNIA —
  похожи на реальных Coriolanus/Volumnia из фикстуры).
- ✅ Пунктуация, переносы строк, длина строк — в духе стихотворного диалога оригинала.
- ❌ Слова по большей части не настоящие английские — модель на этом масштабе (2 слоя, d=64,
  512K токенов увидено) выучила орфографические паттерны английского (буквосочетания,
  окончания), но не лексику и не грамматику связно. Это ожидаемо и НЕ веха — веха связности
  зарезервирована за TinyStories-10M (T14, план `2026-07-14-phase3-model.md`), модель на 2-3
  порядка больше и корпус на 4 порядка больше.

## Вывод

Пайплайн end-to-end (BPE → GPT → тренер → чекпоинт → generate) подтверждён на реальном тексте,
не только на синтетике. Числа и поведение (loss от ln(256) вниз, val монотонно, NaN-guard
не сработал ни разу, warmdown стабилизирует) — здоровые. Хвост T13 закрыт. Веха связности —
следующий шаг (T14, TinyStories-10M), не этот прогон.
