# ggrs

**A Rust port of ggml with first-class CPU training.** Train language models on a
laptop — no GPU, no Python, zero dependencies.

[![CI](https://github.com/zabrodschiipavel-sketch/ggrs/actions/workflows/ci.yml/badge.svg)](https://github.com/zabrodschiipavel-sketch/ggrs/actions)

## Status (July 2026)

| Phase | Status | Result |
|---|---|---|
| 1 — Core | ✅ + retrofit | Arena context, tensors/strides/views, static graph, forward kernels (scalar + AVX2/FMA), ggml-style threading, type traits for block-quantized formats, F16, per-op telemetry |
| 2 — Training | ✅ | Full autograd on the static graph, finite-difference gradcheck for **every** op, OutProd, AdamW (grad clipping, NaN guard, warmdown-33 LR schedule), **milestone: a tied-embeddings mini-LM trained from loss 2.0168 → 0.0000 in 300 steps** |
| 3 — Model | next | BPE tokenizer, GPT (multi-head), char-Shakespeare smoke, **TinyStories-10M CPU speedrun** |
| 4 — GGUF | planned | Checkpoints loadable by real llama.cpp |
| 5 — Quantized inference | planned | Q8_0/Q4_K, KV cache, run foreign GGUF models |
| 6 — Performance | planned | Benchmarks vs llm.c / PyTorch CPU; SIMD everything |
| 7 — Research | planned | Master-weight-free training in GGUF quants (ECO-style), Muon/Polar Express, self-evolution loop |

Honest positioning: phases 1–2 are solid engineering, not novel science. The
research bets live in phases 3+ and the groundwork for them (ECO hook in the
optimizer, quant-ready type traits, telemetry) is already built in. See
[docs/audits/](docs/audits/) for the unvarnished self-audit.

## Why

Our research sweep ([docs/research/](docs/research/), 6 reports, 4 independent
methods) found the niche empty: llama.cpp removed/limits training, llm.c's CPU
path is a 6×-slower-than-PyTorch teaching reference, candle/burn are GPU-first,
and nobody speedruns training on CPUs. Meanwhile r/LocalLLaMA keeps asking for
exactly this and being told it's impossible.

## Quick start

```bash
cargo test -p ggrs-core                 # 105 tests, all green
GGRS_PROFILE=1 cargo test -p ggrs-core --test train_smoke -- --nocapture
# watch a model actually train, with per-op timing
```

Requires stable Rust (edition 2024). No other dependencies — none.

## Design in one paragraph

ggml's architecture, in Rust: a bump-arena `Context` owns all tensors
(`TensorId` handles, `ne`/`nb` strides, zero-copy views), ops are lazy graph
nodes, `compute()` executes the topological order across N threads with
barriers, and results are bit-identical for any thread count. Training builds
gradients *as nodes of the same static graph* (`build_backward`), every op's
backward is verified against finite differences, and `AdamW::step` carries an
explicit hook for ECO-style error feedback (arXiv 2601.22101) — the path to
training directly in quantized weights, no fp32 master copy.

## Project docs

- [docs/project-graph.md](docs/project-graph.md) — full dependency graph of all phases, strategy bets, review findings
- [docs/research/](docs/research/) — the research base: web, OpenAlex, full texts (CORE), OSS landscape, source sweep
- [docs/audits/](docs/audits/) — phase audits (coverage matrix, novelty verdict, code quality)
- [docs/process/workforce.md](docs/process/workforce.md) — how this is built: an architect model (Claude) writes briefs and reviews line-by-line; cheap executor models (DeepSeek flash/pro) implement; every op lands with tests
- [docs/superpowers/specs & plans](docs/superpowers/) — design spec and per-phase implementation plans

## По-русски

ggrs — Rust-порт ggml с обучением на CPU как первоклассной возможностью.
Уже работает: полный autograd со сверкой градиентов конечными разностями,
AdamW, телеметрия; первая сеть обучена (loss 2.02 → 0.00). Дальше — токенизатор,
GPT и спидран TinyStories на ноутбучном процессоре. Вся исследовательская база,
граф проекта и аудиты — в `docs/` на русском.

## License

MIT
