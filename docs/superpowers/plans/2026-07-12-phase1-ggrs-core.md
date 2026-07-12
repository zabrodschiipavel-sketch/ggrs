# Фаза 1: ggrs-core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rust-crate `ggrs-core`: арена-контекст, тензоры со страйдами, статический граф вычислений и forward-операции для GPT (скалярные + AVX2/FMA ядра, потоки с барьерами как в ggml), проверенные тестами против numpy-эталонов.

**Architecture:** Порт архитектуры ggml: `Context` владеет bump-ареной и таблицей тензоров, снаружи тензор — `TensorId`. Операции строят узлы графа (ленивые), `build_forward` собирает топологический порядок, `compute` исполняет ядра; N потоков делят строки каждого узла и синхронизируются барьером. Ядра пишут в арену через сырые указатели (`UnsafeCell`), как ggml.

**Tech Stack:** Rust stable (toolchain `stable-x86_64-pc-windows-gnu`), zero runtime-зависимостей в Фазе 1. Python 3.12 + numpy — только dev-инструмент генерации фикстур.

## Global Constraints

- ОС: Windows 11, CPU Ryzen 7 7730U (AVX2+FMA есть; AVX-512/AMX нет).
- Rust: stable, edition 2021, toolchain `stable-x86_64-pc-windows-gnu` (без VS Build Tools).
- Зависимости ggrs-core: НОЛЬ (memmap2/half появятся в следующих фазах в других crates).
- Соглашение о layout как в ggml: `ne[0]` — самое быстрое измерение (длина строки), `nb[i]` — страйды в байтах; 2D-тензор «rows×cols» создаётся как `new_tensor_2d(F32, cols, rows)`.
- Каждое горячее ядро: скалярный эталон + AVX2/FMA через `std::arch`; выбор в рантайме (`is_x86_feature_detected!`). Ядра на exp/tanh (Silu, Gelu, SoftMax) в Фазе 1 — только скалярные (осознанное решение: полиномиальный exp в AVX2 — отдельная работа Фазы 6).
- Op::CrossEntropyLoss считается одним потоком (ith==0); остальные ждут на барьере — осознанное упрощение Фазы 1.
- Коммиты: `git -C C:\Users\pavel\ggrs`, идентичность уже настроена в репо.
- Все команды из корня репо `C:\Users\pavel\ggrs`, если не сказано иное.

---

### Task 0: Toolchain — установка Rust

**Files:** нет (системная установка).

**Interfaces:**
- Produces: рабочие `cargo`, `rustc` в PATH.

- [ ] **Step 1: Установить rustup с GNU-toolchain**

```powershell
winget install --id Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements
# новый PowerShell или обновить PATH:
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + $env:Path
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

- [ ] **Step 2: Проверить**

Run: `cargo --version; rustc --version`
Expected: версии печатаются, host `x86_64-pc-windows-gnu` в `rustc -vV`.

---

### Task 1: Workspace scaffold

**Files:**
- Create: `Cargo.toml`, `crates/ggrs-core/Cargo.toml`, `crates/ggrs-core/src/lib.rs`, `.gitignore`

**Interfaces:**
- Produces: пустой crate `ggrs-core`, `cargo test` зелёный.

- [ ] **Step 1: Создать файлы**

`Cargo.toml` (корень):
```toml
[workspace]
resolver = "2"
members = ["crates/ggrs-core"]
```

`crates/ggrs-core/Cargo.toml`:
```toml
[package]
name = "ggrs-core"
version = "0.1.0"
edition = "2021"

[dependencies]
```

`crates/ggrs-core/src/lib.rs`:
```rust
//! ggrs-core: порт ядра ggml на Rust — тензоры, граф, CPU-ядра.

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(2 + 2, 4);
    }
}
```

`.gitignore`:
```
/target
```

- [ ] **Step 2: Проверить сборку и тест**

Run: `cargo test -p ggrs-core`
Expected: `test tests::smoke ... ok`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: workspace scaffold, ggrs-core crate"
```

---

### Task 2: DType, Tensor, Context с ареной

**Files:**
- Create: `crates/ggrs-core/src/dtype.rs`, `crates/ggrs-core/src/tensor.rs`, `crates/ggrs-core/src/op.rs`, `crates/ggrs-core/src/context.rs`
- Modify: `crates/ggrs-core/src/lib.rs`
- Test: юнит-тесты внутри `context.rs`

**Interfaces:**
- Produces (всё публичное, используется всеми последующими задачами):
  - `DType { F32, I32 }`, `DType::size(self) -> usize`
  - `TensorId(pub usize)`; `MAX_DIMS: usize = 4`; `MAX_SRC: usize = 4`
  - `Tensor { dtype, ne: [usize;4], nb: [usize;4], op: Op, src: [Option<TensorId>;4], offset: usize, op_params: [u32;8], is_param: bool }` + методы `nelements()`, `nrows()`, `is_contiguous()`
  - `Op` — enum со ВСЕМИ вариантами фазы: `None, Add, Mul, Scale, Silu, Gelu, MulMat, SoftMax, RmsNorm, GetRows, Rope, Cont, Reshape, Permute, CrossEntropyLoss`
  - `Context::new(mem_size: usize)`, `new_tensor_1d/2d/3d/4d(&mut self, DType, ...) -> TensorId`, `t(&self, TensorId) -> &Tensor`, `n_tensors(&self) -> usize`
  - `data_f32(&self, id) -> &[f32]`, `data_f32_mut(&mut self, id) -> &mut [f32]`, `data_i32`, `data_i32_mut`, `set_f32(&mut self, id, &[f32])`, `set_i32`, `get_f32(&self, id, [usize;4]) -> f32` (строковое чтение через nb — работает для views)
  - `pub(crate) fn base(&self) -> *mut u8` — для ядер

- [ ] **Step 1: Написать падающий тест** (в конец `context.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn tensor_creation_and_strides() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_2d(DType::F32, 3, 2); // 2 строки по 3
        let t = ctx.t(a);
        assert_eq!(t.ne, [3, 2, 1, 1]);
        assert_eq!(t.nb, [4, 12, 24, 24]);
        assert_eq!(t.nelements(), 6);
        assert_eq!(t.nrows(), 2);
        assert!(t.is_contiguous());
    }

    #[test]
    fn data_roundtrip() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_1d(DType::F32, 4);
        ctx.set_f32(a, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ctx.data_f32(a), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ctx.get_f32(a, [2, 0, 0, 0]), 3.0);
        let b = ctx.new_tensor_1d(DType::I32, 2);
        ctx.set_i32(b, &[7, -1]);
        assert_eq!(ctx.data_i32(b), &[7, -1]);
    }

    #[test]
    #[should_panic(expected = "arena out of memory")]
    fn arena_overflow_panics() {
        let mut ctx = Context::new(16);
        let _ = ctx.new_tensor_1d(DType::F32, 1024);
    }
}
```

- [ ] **Step 2: Убедиться, что не компилируется**

Run: `cargo test -p ggrs-core`
Expected: ошибки компиляции (типов нет).

- [ ] **Step 3: Реализация**

`crates/ggrs-core/src/dtype.rs`:
```rust
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    I32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
        }
    }
}
```

`crates/ggrs-core/src/op.rs`:
```rust
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    None,
    Add,
    Mul,
    Scale,
    Silu,
    Gelu,
    MulMat,
    SoftMax,
    RmsNorm,
    GetRows,
    Rope,
    Cont,
    Reshape,
    Permute,
    CrossEntropyLoss,
}
```

`crates/ggrs-core/src/tensor.rs`:
```rust
use crate::dtype::DType;
use crate::op::Op;

pub const MAX_DIMS: usize = 4;
pub const MAX_SRC: usize = 4;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TensorId(pub usize);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub ne: [usize; MAX_DIMS], // число элементов по измерениям, ne[0] — самое быстрое
    pub nb: [usize; MAX_DIMS], // страйды в байтах
    pub op: Op,
    pub src: [Option<TensorId>; MAX_SRC],
    pub offset: usize, // байтовое смещение в арене
    pub op_params: [u32; 8],
    pub is_param: bool,
}

impl Tensor {
    pub fn nelements(&self) -> usize {
        self.ne.iter().product()
    }
    pub fn nrows(&self) -> usize {
        self.ne[1] * self.ne[2] * self.ne[3]
    }
    pub fn is_contiguous(&self) -> bool {
        let mut expected = self.dtype.size();
        for i in 0..MAX_DIMS {
            if self.ne[i] != 1 && self.nb[i] != expected {
                return false;
            }
            expected *= self.ne[i];
        }
        true
    }
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.ne == other.ne
    }
}
```

`crates/ggrs-core/src/context.rs`:
```rust
use std::cell::UnsafeCell;

use crate::dtype::DType;
use crate::op::Op;
use crate::tensor::{Tensor, TensorId, MAX_DIMS, MAX_SRC};

struct Arena {
    buf: UnsafeCell<Box<[u8]>>,
}
// Ядра пишут в арену через сырые указатели из нескольких потоков,
// каждый поток — в свои строки. Дисциплина не-алиасинга — на ядрах (как в ggml).
unsafe impl Sync for Arena {}

pub struct Context {
    tensors: Vec<Tensor>,
    arena: Arena,
    arena_used: usize,
}

impl Context {
    pub fn new(mem_size: usize) -> Context {
        Context {
            tensors: Vec::new(),
            arena: Arena { buf: UnsafeCell::new(vec![0u8; mem_size].into_boxed_slice()) },
            arena_used: 0,
        }
    }

    fn alloc(&mut self, nbytes: usize) -> usize {
        let offset = (self.arena_used + 31) & !31; // 32-байтное выравнивание
        let len = unsafe { (*self.arena.buf.get()).len() };
        assert!(offset + nbytes <= len, "ggrs: arena out of memory");
        self.arena_used = offset + nbytes;
        offset
    }

    pub fn new_tensor(&mut self, dtype: DType, ne: [usize; MAX_DIMS]) -> TensorId {
        let ts = dtype.size();
        let nb = [ts, ts * ne[0], ts * ne[0] * ne[1], ts * ne[0] * ne[1] * ne[2]];
        let nbytes = ts * ne.iter().product::<usize>();
        let offset = self.alloc(nbytes);
        self.push_tensor(Tensor {
            dtype, ne, nb, op: Op::None,
            src: [None; MAX_SRC], offset, op_params: [0; 8], is_param: false,
        })
    }

    pub(crate) fn push_tensor(&mut self, t: Tensor) -> TensorId {
        self.tensors.push(t);
        TensorId(self.tensors.len() - 1)
    }

    pub fn new_tensor_1d(&mut self, dtype: DType, ne0: usize) -> TensorId {
        self.new_tensor(dtype, [ne0, 1, 1, 1])
    }
    pub fn new_tensor_2d(&mut self, dtype: DType, ne0: usize, ne1: usize) -> TensorId {
        self.new_tensor(dtype, [ne0, ne1, 1, 1])
    }
    pub fn new_tensor_3d(&mut self, dtype: DType, ne0: usize, ne1: usize, ne2: usize) -> TensorId {
        self.new_tensor(dtype, [ne0, ne1, ne2, 1])
    }
    pub fn new_tensor_4d(&mut self, dtype: DType, ne0: usize, ne1: usize, ne2: usize, ne3: usize) -> TensorId {
        self.new_tensor(dtype, [ne0, ne1, ne2, ne3])
    }

    pub fn t(&self, id: TensorId) -> &Tensor {
        &self.tensors[id.0]
    }
    pub(crate) fn t_mut(&mut self, id: TensorId) -> &mut Tensor {
        &mut self.tensors[id.0]
    }
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn base(&self) -> *mut u8 {
        unsafe { (*self.arena.buf.get()).as_mut_ptr() }
    }

    pub fn data_f32(&self, id: TensorId) -> &[f32] {
        let t = self.t(id);
        assert_eq!(t.dtype, DType::F32);
        assert!(t.is_contiguous(), "data_f32: тензор не непрерывный, используй get_f32");
        unsafe {
            std::slice::from_raw_parts(self.base().add(t.offset) as *const f32, t.nelements())
        }
    }
    pub fn data_f32_mut(&mut self, id: TensorId) -> &mut [f32] {
        let t = self.t(id).clone();
        assert_eq!(t.dtype, DType::F32);
        assert!(t.is_contiguous());
        unsafe {
            std::slice::from_raw_parts_mut(self.base().add(t.offset) as *mut f32, t.nelements())
        }
    }
    pub fn data_i32(&self, id: TensorId) -> &[i32] {
        let t = self.t(id);
        assert_eq!(t.dtype, DType::I32);
        assert!(t.is_contiguous());
        unsafe {
            std::slice::from_raw_parts(self.base().add(t.offset) as *const i32, t.nelements())
        }
    }
    pub fn data_i32_mut(&mut self, id: TensorId) -> &mut [i32] {
        let t = self.t(id).clone();
        assert_eq!(t.dtype, DType::I32);
        assert!(t.is_contiguous());
        unsafe {
            std::slice::from_raw_parts_mut(self.base().add(t.offset) as *mut i32, t.nelements())
        }
    }
    pub fn set_f32(&mut self, id: TensorId, vals: &[f32]) {
        self.data_f32_mut(id).copy_from_slice(vals);
    }
    pub fn set_i32(&mut self, id: TensorId, vals: &[i32]) {
        self.data_i32_mut(id).copy_from_slice(vals);
    }
    /// Строковое чтение через страйды — работает и для views/permute.
    pub fn get_f32(&self, id: TensorId, idx: [usize; MAX_DIMS]) -> f32 {
        let t = self.t(id);
        assert_eq!(t.dtype, DType::F32);
        let off = t.offset + idx[0] * t.nb[0] + idx[1] * t.nb[1] + idx[2] * t.nb[2] + idx[3] * t.nb[3];
        unsafe { *(self.base().add(off) as *const f32) }
    }
}
```

`crates/ggrs-core/src/lib.rs` (замена целиком):
```rust
//! ggrs-core: порт ядра ggml на Rust — тензоры, граф, CPU-ядра.

pub mod context;
pub mod dtype;
pub mod op;
pub mod tensor;

pub use context::Context;
pub use dtype::DType;
pub use op::Op;
pub use tensor::{Tensor, TensorId, MAX_DIMS, MAX_SRC};
```

- [ ] **Step 4: Тесты зелёные**

Run: `cargo test -p ggrs-core`
Expected: 3 теста PASS (+smoke).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): DType, Tensor, Context с bump-ареной и доступом к данным"
```

---

### Task 3: Views — reshape, permute, transpose

**Files:**
- Modify: `crates/ggrs-core/src/context.rs` (методы + тесты)

**Interfaces:**
- Produces: `Context::reshape_2d(a, ne0, ne1)`, `reshape_3d(a, ne0, ne1, ne2)`, `permute(a, [usize;4])` (axes[i] = куда уходит измерение i, как в ggml_permute), `transpose(a)`. Все возвращают `TensorId` view-тензора (общие данные, `op` = Reshape/Permute, `src[0]=a`).

- [ ] **Step 1: Падающий тест** (в `context.rs::tests`)

```rust
#[test]
fn reshape_and_permute() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 4, 2); // 2 строки по 4
    ctx.set_f32(a, &[0., 1., 2., 3., 4., 5., 6., 7.]);

    let r = ctx.reshape_2d(a, 2, 4); // 4 строки по 2, данные общие
    assert_eq!(ctx.t(r).ne, [2, 4, 1, 1]);
    assert_eq!(ctx.get_f32(r, [1, 2, 0, 0]), 5.0);

    let p = ctx.transpose(a); // [2,4]: p[i,j] == a[j,i]
    assert_eq!(ctx.t(p).ne, [2, 4, 1, 1]);
    assert!(!ctx.t(p).is_contiguous());
    assert_eq!(ctx.get_f32(p, [1, 3, 0, 0]), 7.0); // a[3,1] = 7
    assert_eq!(ctx.get_f32(p, [0, 2, 0, 0]), 2.0); // a[2,0] = 2
}
```

- [ ] **Step 2: Проверить падение** — `cargo test -p ggrs-core` → ошибки компиляции.

- [ ] **Step 3: Реализация** (добавить в `impl Context`)

```rust
fn new_view(&mut self, src_id: TensorId, ne: [usize; MAX_DIMS], nb: [usize; MAX_DIMS], op: Op) -> TensorId {
    let src = self.t(src_id);
    let t = Tensor {
        dtype: src.dtype,
        ne, nb, op,
        src: [Some(src_id), None, None, None],
        offset: src.offset,
        op_params: [0; 8],
        is_param: false,
    };
    self.push_tensor(t)
}

pub fn reshape_2d(&mut self, a: TensorId, ne0: usize, ne1: usize) -> TensorId {
    self.reshape(a, [ne0, ne1, 1, 1])
}
pub fn reshape_3d(&mut self, a: TensorId, ne0: usize, ne1: usize, ne2: usize) -> TensorId {
    self.reshape(a, [ne0, ne1, ne2, 1])
}
fn reshape(&mut self, a: TensorId, ne: [usize; MAX_DIMS]) -> TensorId {
    let t = self.t(a);
    assert!(t.is_contiguous(), "reshape: источник должен быть непрерывным");
    assert_eq!(t.nelements(), ne.iter().product::<usize>(), "reshape: число элементов не совпадает");
    let ts = t.dtype.size();
    let nb = [ts, ts * ne[0], ts * ne[0] * ne[1], ts * ne[0] * ne[1] * ne[2]];
    self.new_view(a, ne, nb, Op::Reshape)
}

/// axes[i] — новая позиция измерения i (семантика ggml_permute).
pub fn permute(&mut self, a: TensorId, axes: [usize; MAX_DIMS]) -> TensorId {
    let t = self.t(a);
    let mut ne = [0usize; MAX_DIMS];
    let mut nb = [0usize; MAX_DIMS];
    for i in 0..MAX_DIMS {
        ne[axes[i]] = t.ne[i];
        nb[axes[i]] = t.nb[i];
    }
    self.new_view(a, ne, nb, Op::Permute)
}

pub fn transpose(&mut self, a: TensorId) -> TensorId {
    self.permute(a, [1, 0, 2, 3])
}
```

- [ ] **Step 4: Тесты зелёные** — `cargo test -p ggrs-core`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): views — reshape, permute, transpose"`

---

### Task 4: Граф — построение и топологический порядок

**Files:**
- Create: `crates/ggrs-core/src/graph.rs`
- Modify: `crates/ggrs-core/src/lib.rs` (`pub mod graph; pub use graph::{Graph, build_forward};`)

**Interfaces:**
- Produces: `Graph { pub nodes: Vec<TensorId> }`; `build_forward(ctx: &Context, result: TensorId) -> Graph` — DFS от результата, источники раньше потребителей.

- [ ] **Step 1: Падающий тест** (в `graph.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, DType, Op};
    use crate::tensor::TensorId;

    #[test]
    fn topo_order_diamond() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_1d(DType::F32, 4);
        // ромб: d = (a+a) * (a+a)  — общий подграф b
        let b = ctx.add(a, a);
        let d = ctx.mul(b, b);
        let g = build_forward(&ctx, d);
        let pos = |id: TensorId| g.nodes.iter().position(|&n| n == id).unwrap();
        assert!(pos(a) < pos(b) && pos(b) < pos(d));
        // без дублей
        let mut sorted = g.nodes.clone();
        sorted.sort_by_key(|t| t.0);
        sorted.dedup();
        assert_eq!(sorted.len(), g.nodes.len());
        assert_eq!(ctx.t(d).op, Op::Mul);
    }
}
```

Примечание: тест использует `ctx.add`/`ctx.mul` — билдеры из Task 5. Задачи 4 и 5 компилируются вместе; тест графа временно помечать `#[ignore]` не нужно — реализуй Task 4 и Task 5 билдеры (только билдеры, без ядер) до зелёного состояния этого теста.

- [ ] **Step 2: Реализация**

`crates/ggrs-core/src/graph.rs`:
```rust
use crate::context::Context;
use crate::tensor::TensorId;

pub struct Graph {
    pub nodes: Vec<TensorId>,
}

pub fn build_forward(ctx: &Context, result: TensorId) -> Graph {
    let mut visited = vec![false; ctx.n_tensors()];
    let mut nodes = Vec::new();
    visit(ctx, result, &mut visited, &mut nodes);
    Graph { nodes }
}

fn visit(ctx: &Context, id: TensorId, visited: &mut Vec<bool>, nodes: &mut Vec<TensorId>) {
    if visited[id.0] {
        return;
    }
    visited[id.0] = true;
    for s in ctx.t(id).src.iter().flatten() {
        visit(ctx, *s, visited, nodes);
    }
    nodes.push(id);
}
```

И минимальные билдеры в `context.rs` (полная версия с ядрами — Task 5):
```rust
pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
    self.binary_op(Op::Add, a, b)
}
pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
    self.binary_op(Op::Mul, a, b)
}
fn binary_op(&mut self, op: Op, a: TensorId, b: TensorId) -> TensorId {
    let ta = self.t(a);
    let tb = self.t(b);
    // broadcast src1: по каждому измерению ne равны или у b единица
    for i in 0..MAX_DIMS {
        assert!(tb.ne[i] == ta.ne[i] || tb.ne[i] == 1, "binary_op: несовместимые формы");
    }
    let ne = ta.ne;
    let dst = self.new_tensor(ta.dtype, ne);
    let d = self.t_mut(dst);
    d.op = op;
    d.src = [Some(a), Some(b), None, None];
    dst
}
```

- [ ] **Step 3: Тесты зелёные** — `cargo test -p ggrs-core`.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat(core): граф вычислений, топологический порядок, билдеры add/mul"`

---

### Task 5: Compute-драйвер + поэлементные ядра (скаляр)

**Files:**
- Create: `crates/ggrs-core/src/compute.rs`, `crates/ggrs-core/src/kernels/mod.rs`, `crates/ggrs-core/src/kernels/elementwise.rs`
- Modify: `crates/ggrs-core/src/context.rs` (билдеры `scale`, `silu`, `gelu`), `lib.rs` (`pub mod compute; mod kernels; pub use compute::compute;`)

**Interfaces:**
- Produces:
  - `compute(ctx: &Context, graph: &Graph, n_threads: usize)` — в этой задаче реально исполняет только n_threads=1 (аргумент уже в сигнатуре, потоки в Task 11)
  - Билдеры: `scale(a, s: f32)`, `silu(a)`, `gelu(a)` (tanh-аппроксимация, как ggml)
  - `kernels::split(n, ith, nth) -> (usize, usize)` — деление строк между потоками
  - Ядра получают `(ctx, dst_id, ith, nth)`; Op::None/Reshape/Permute — no-op
- Consumes: Task 2–4.

- [ ] **Step 1: Падающий тест** (`crates/ggrs-core/tests/ops_elementwise.rs`)

```rust
use ggrs_core::*;

fn ctx1m() -> Context { Context::new(1 << 20) }

#[test]
fn add_mul_scale() {
    let mut ctx = ctx1m();
    let a = ctx.new_tensor_1d(DType::F32, 5);
    let b = ctx.new_tensor_1d(DType::F32, 5);
    ctx.set_f32(a, &[1., 2., 3., 4., 5.]);
    ctx.set_f32(b, &[10., 20., 30., 40., 50.]);
    let s = ctx.add(a, b);
    let m = ctx.mul(s, a);
    let r = ctx.scale(m, 0.5);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    assert_eq!(ctx.data_f32(r), &[5.5, 22.0, 49.5, 88.0, 137.5]);
}

#[test]
fn add_broadcast_rows() {
    // маска [4,1] прибавляется к [4,3] построчно
    let mut ctx = ctx1m();
    let a = ctx.new_tensor_2d(DType::F32, 4, 3);
    let m = ctx.new_tensor_2d(DType::F32, 4, 1);
    ctx.set_f32(a, &[0.; 12]);
    ctx.set_f32(m, &[0., -1., -2., -3.]);
    let r = ctx.add(a, m);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    assert_eq!(ctx.data_f32(r), &[0., -1., -2., -3., 0., -1., -2., -3., 0., -1., -2., -3.]);
}

#[test]
fn silu_gelu_values() {
    let mut ctx = ctx1m();
    let a = ctx.new_tensor_1d(DType::F32, 3);
    ctx.set_f32(a, &[-1.0, 0.0, 2.0]);
    let s = ctx.silu(a);
    let g_ = ctx.gelu(a);
    let graph = build_forward(&ctx, s);
    compute(&ctx, &graph, 1);
    let graph2 = build_forward(&ctx, g_);
    compute(&ctx, &graph2, 1);
    let sv = ctx.data_f32(s);
    // silu(x) = x*sigmoid(x): silu(-1) ≈ -0.26894, silu(0)=0, silu(2) ≈ 1.76159
    assert!((sv[0] + 0.26894143).abs() < 1e-5);
    assert!(sv[1].abs() < 1e-9);
    assert!((sv[2] - 1.7615942).abs() < 1e-5);
    let gv = ctx.data_f32(g_);
    // gelu tanh-аппрокс: gelu(-1) ≈ -0.15881, gelu(0)=0, gelu(2) ≈ 1.95460
    assert!((gv[0] + 0.15880796).abs() < 1e-4);
    assert!((gv[2] - 1.9545977).abs() < 1e-4);
}
```

- [ ] **Step 2: Проверить падение** — `cargo test -p ggrs-core` → ошибки компиляции.

- [ ] **Step 3: Реализация**

Билдеры в `context.rs`:
```rust
pub fn scale(&mut self, a: TensorId, s: f32) -> TensorId {
    let dst = self.unary_op(Op::Scale, a);
    self.t_mut(dst).op_params[0] = s.to_bits();
    dst
}
pub fn silu(&mut self, a: TensorId) -> TensorId {
    self.unary_op(Op::Silu, a)
}
pub fn gelu(&mut self, a: TensorId) -> TensorId {
    self.unary_op(Op::Gelu, a)
}
fn unary_op(&mut self, op: Op, a: TensorId) -> TensorId {
    let ne = self.t(a).ne;
    let dtype = self.t(a).dtype;
    let dst = self.new_tensor(dtype, ne);
    let d = self.t_mut(dst);
    d.op = op;
    d.src = [Some(a), None, None, None];
    dst
}
```

`crates/ggrs-core/src/compute.rs`:
```rust
use crate::context::Context;
use crate::graph::Graph;
use crate::kernels;

pub fn compute(ctx: &Context, graph: &Graph, n_threads: usize) {
    assert!(n_threads >= 1);
    // Многопоточная ветка — Task 11. Пока исполняем одним потоком.
    let _ = n_threads;
    for &id in &graph.nodes {
        kernels::dispatch(ctx, id, 0, 1);
    }
}
```

`crates/ggrs-core/src/kernels/mod.rs`:
```rust
pub mod elementwise;

use crate::context::Context;
use crate::op::Op;
use crate::tensor::{Tensor, TensorId};

pub(crate) fn dispatch(ctx: &Context, id: TensorId, ith: usize, nth: usize) {
    match ctx.t(id).op {
        Op::None | Op::Reshape | Op::Permute => {}
        Op::Add => elementwise::add(ctx, id, ith, nth),
        Op::Mul => elementwise::mul(ctx, id, ith, nth),
        Op::Scale => elementwise::scale(ctx, id, ith, nth),
        Op::Silu => elementwise::silu(ctx, id, ith, nth),
        Op::Gelu => elementwise::gelu(ctx, id, ith, nth),
        op => unimplemented!("ядро для {:?} ещё не реализовано", op),
    }
}

/// Диапазон строк потока ith из nth.
pub(crate) fn split(n: usize, ith: usize, nth: usize) -> (usize, usize) {
    let per = n.div_ceil(nth);
    ((ith * per).min(n), ((ith + 1) * per).min(n))
}

/// Указатель на строку (i1,i2,i3) тензора.
pub(crate) unsafe fn row_ptr(ctx: &Context, t: &Tensor, i1: usize, i2: usize, i3: usize) -> *mut u8 {
    ctx.base().add(t.offset + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3])
}

/// Разложить плоский индекс строки в (i1,i2,i3) по ne тензора.
pub(crate) fn unravel_row(t: &Tensor, ir: usize) -> (usize, usize, usize) {
    let i3 = ir / (t.ne[1] * t.ne[2]);
    let i2 = (ir / t.ne[1]) % t.ne[2];
    let i1 = ir % t.ne[1];
    (i1, i2, i3)
}
```

`crates/ggrs-core/src/kernels/elementwise.rs`:
```rust
use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

/// Общий цикл бинарной операции с broadcast src1 (ne==ne или 1 по каждому измерению).
fn binary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, f: impl Fn(f32, f32) -> f32) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let b = ctx.t(dst.src[1].unwrap());
    assert_eq!(dst.nb[0], 4, "binary: dst строки должны быть плотными");
    assert_eq!(a.nb[0], 4, "binary: src0 строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pb = row_ptr(ctx, b, i1 % b.ne[1], i2 % b.ne[2], i3 % b.ne[3]) as *const f32;
            if b.ne[0] == ne0 {
                assert_eq!(b.nb[0], 4);
                for i in 0..ne0 {
                    *pd.add(i) = f(*pa.add(i), *pb.add(i));
                }
            } else {
                assert_eq!(b.ne[0], 1, "binary: broadcast только ne0==1");
                let s = *pb;
                for i in 0..ne0 {
                    *pd.add(i) = f(*pa.add(i), s);
                }
            }
        }
    }
}

fn unary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, f: impl Fn(f32) -> f32) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(dst.nb[0], 4);
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            for i in 0..ne0 {
                *pd.add(i) = f(*pa.add(i));
            }
        }
    }
}

pub fn add(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, |x, y| x + y);
}
pub fn mul(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, |x, y| x * y);
}
pub fn scale(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    let s = f32::from_bits(ctx.t(dst).op_params[0]);
    unary(ctx, dst, ith, nth, |x| x * s);
}
/// silu(x) = x·σ(x) = x/(1+e^(−x))
pub fn silu(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    unary(ctx, dst, ith, nth, |x| x / (1.0 + (-x).exp()));
}
pub fn gelu(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    unary(ctx, dst, ith, nth, |x| {
        0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
    });
}
```

- [ ] **Step 4: Тесты зелёные** — `cargo test -p ggrs-core` → все PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): compute-драйвер, поэлементные ядра add/mul/scale/silu/gelu"`

---

### Task 6: SIMD-модуль — AVX2/FMA векторные примитивы

**Files:**
- Create: `crates/ggrs-core/src/simd.rs`
- Modify: `crates/ggrs-core/src/kernels/elementwise.rs` (add/mul/scale через simd), `lib.rs` (`pub mod simd;`)

**Interfaces:**
- Produces:
  - `simd::have_avx2() -> bool` (кэш через `OnceLock`)
  - Диспетчеры: `simd::vec_add(a: &[f32], b: &[f32], d: &mut [f32])`, `vec_mul(a, b, d)`, `vec_scale(d: &mut [f32], s: f32)`, `vec_dot(a: &[f32], b: &[f32]) -> f32`
  - Публичные обе реализации для тестов: `simd::scalar::{vec_add, vec_mul, vec_scale, vec_dot}`, `simd::avx2::{...}` (под `#[cfg(target_arch = "x86_64")]`, unsafe с `#[target_feature(enable = "avx2,fma")]`)

- [ ] **Step 1: Падающий тест** (`crates/ggrs-core/tests/simd_parity.rs`)

```rust
use ggrs_core::simd;

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    // детерминированный LCG, без зависимостей
    let mut s = seed as u64;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }).collect()
}

#[test]
fn scalar_vs_avx2_parity() {
    if !simd::have_avx2() {
        eprintln!("AVX2 недоступен — тест пропущен");
        return;
    }
    for &n in &[1usize, 7, 8, 15, 16, 33, 1024, 1000] {
        let a = pseudo(n, 1);
        let b = pseudo(n, 2);

        let ds = {
            let mut d = vec![0.0; n];
            simd::scalar::vec_add(&a, &b, &mut d);
            d
        };
        let dv = {
            let mut d = vec![0.0; n];
            unsafe { simd::avx2::vec_add(&a, &b, &mut d) };
            d
        };
        assert_eq!(ds, dv, "vec_add n={n}"); // сложение поэлементное — бит-в-бит

        let dot_s = simd::scalar::vec_dot(&a, &b);
        let dot_v = unsafe { simd::avx2::vec_dot(&a, &b) };
        let tol = 1e-5 * (n as f32).sqrt().max(1.0);
        assert!((dot_s - dot_v).abs() <= tol * dot_s.abs().max(1.0), "vec_dot n={n}: {dot_s} vs {dot_v}");
    }
}
```

- [ ] **Step 2: Проверить падение** — ошибки компиляции.

- [ ] **Step 3: Реализация**

`crates/ggrs-core/src/simd.rs`:
```rust
//! Векторные примитивы: скалярный эталон + AVX2/FMA. Выбор в рантайме.

use std::sync::OnceLock;

pub fn have_avx2() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

pub mod scalar {
    pub fn vec_add(a: &[f32], b: &[f32], d: &mut [f32]) {
        for i in 0..d.len() {
            d[i] = a[i] + b[i];
        }
    }
    pub fn vec_mul(a: &[f32], b: &[f32], d: &mut [f32]) {
        for i in 0..d.len() {
            d[i] = a[i] * b[i];
        }
    }
    pub fn vec_scale(d: &mut [f32], s: f32) {
        for x in d.iter_mut() {
            *x *= s;
        }
    }
    pub fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    /// # Safety: вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
            i += 16;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }

    /// # Safety: вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_add(a: &[f32], b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let (pa, pb, pd) = (a.as_ptr(), b.as_ptr(), d.as_mut_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_add_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))));
            i += 8;
        }
        while i < n {
            *pd.add(i) = *pa.add(i) + *pb.add(i);
            i += 1;
        }
    }

    /// # Safety: вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_mul(a: &[f32], b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let (pa, pb, pd) = (a.as_ptr(), b.as_ptr(), d.as_mut_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))));
            i += 8;
        }
        while i < n {
            *pd.add(i) = *pa.add(i) * *pb.add(i);
            i += 1;
        }
    }

    /// # Safety: вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_scale(d: &mut [f32], s: f32) {
        let n = d.len();
        let pd = d.as_mut_ptr();
        let vs = _mm256_set1_ps(s);
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_mul_ps(_mm256_loadu_ps(pd.add(i)), vs));
            i += 8;
        }
        while i < n {
            *pd.add(i) *= s;
            i += 1;
        }
    }
}

// ---- диспетчеры ----
pub fn vec_add(a: &[f32], b: &[f32], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        unsafe { avx2::vec_add(a, b, d) };
        return;
    }
    scalar::vec_add(a, b, d);
}
pub fn vec_mul(a: &[f32], b: &[f32], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        unsafe { avx2::vec_mul(a, b, d) };
        return;
    }
    scalar::vec_mul(a, b, d);
}
pub fn vec_scale(d: &mut [f32], s: f32) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        unsafe { avx2::vec_scale(d, s) };
        return;
    }
    scalar::vec_scale(d, s);
}
pub fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        return unsafe { avx2::vec_dot(a, b) };
    }
    scalar::vec_dot(a, b)
}
```

Переключить плотный путь `binary` в `elementwise.rs` на SIMD. Рефактор `binary` — вместо замыкания принимает вид операции:

```rust
#[derive(Copy, Clone)]
enum BinKind { Add, Mul }

fn binary(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize, kind: BinKind) {
    // ... как раньше до внутреннего цикла, затем:
    unsafe {
        let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
        let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
        let pb = row_ptr(ctx, b, i1 % b.ne[1], i2 % b.ne[2], i3 % b.ne[3]) as *const f32;
        if b.ne[0] == ne0 {
            assert_eq!(b.nb[0], 4);
            let sa = std::slice::from_raw_parts(pa, ne0);
            let sb = std::slice::from_raw_parts(pb, ne0);
            let sd = std::slice::from_raw_parts_mut(pd, ne0);
            match kind {
                BinKind::Add => crate::simd::vec_add(sa, sb, sd),
                BinKind::Mul => crate::simd::vec_mul(sa, sb, sd),
            }
        } else {
            assert_eq!(b.ne[0], 1, "binary: broadcast только ne0==1");
            let s = *pb;
            match kind {
                BinKind::Add => for i in 0..ne0 { *pd.add(i) = *pa.add(i) + s; },
                BinKind::Mul => for i in 0..ne0 { *pd.add(i) = *pa.add(i) * s; },
            }
        }
    }
}

pub fn add(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, BinKind::Add);
}
pub fn mul(ctx: &Context, dst: TensorId, ith: usize, nth: usize) {
    binary(ctx, dst, ith, nth, BinKind::Mul);
}
```

`scale` — через `vec_scale`: скопировать строку a в dst (`copy_nonoverlapping`), затем `simd::vec_scale(sd, s)`. Unary с exp/tanh (silu/gelu) остаются скалярными (Global Constraints).

- [ ] **Step 4: Тесты зелёные** — `cargo test -p ggrs-core` (в т.ч. старые elementwise-тесты — они теперь идут через AVX2-путь).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): SIMD-модуль AVX2/FMA, поэлементные ядра на векторных примитивах"`

---

### Task 7: MulMat

**Files:**
- Create: `crates/ggrs-core/src/kernels/mulmat.rs`
- Modify: `context.rs` (билдер), `kernels/mod.rs` (диспатч), тест `crates/ggrs-core/tests/ops_mulmat.rs`

**Interfaces:**
- Produces: `Context::mul_mat(a, b) -> TensorId`. Семантика ggml: `a.ne0 == b.ne0` (общая размерность k); `dst.ne = [a.ne1, b.ne1, b.ne2, b.ne3]`; `dst[i0, i1, i2, i3] = dot(строка i0 тензора a (при i2%a.ne2, i3%a.ne3), строка i1 тензора b)`. Требование: строки a и b плотные (`nb[0]==4`).

- [ ] **Step 1: Падающий тест**

```rust
use ggrs_core::*;

#[test]
fn mulmat_2x3_times_4x3() {
    // a: k=3, m=2 строки; b: k=3, n=4 строки; dst: [m=2, n=4]
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2);
    let b = ctx.new_tensor_2d(DType::F32, 3, 4);
    ctx.set_f32(a, &[1., 2., 3., 4., 5., 6.]);
    ctx.set_f32(b, &[1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.]);
    let d = ctx.mul_mat(a, b);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);
    // dst[i0=строка a, i1=строка b]: dot(a_i0, b_i1)
    assert_eq!(ctx.data_f32(d), &[1., 4., 2., 5., 3., 6., 6., 15.]);
    assert_eq!(ctx.t(d).ne, [2, 4, 1, 1]);
}

#[test]
fn mulmat_matches_naive_random() {
    let mut ctx = Context::new(1 << 24);
    let (k, m, n) = (64usize, 17, 23);
    let a = ctx.new_tensor_2d(DType::F32, k, m);
    let b = ctx.new_tensor_2d(DType::F32, k, n);
    let av: Vec<f32> = (0..k * m).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| ((i * 40503usize + 7) % 1000) as f32 / 500.0 - 1.0).collect();
    ctx.set_f32(a, &av);
    ctx.set_f32(b, &bv);
    let d = ctx.mul_mat(a, b);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);
    let out = ctx.data_f32(d);
    for i1 in 0..n {
        for i0 in 0..m {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += av[i0 * k + kk] as f64 * bv[i1 * k + kk] as f64;
            }
            let got = out[i1 * m + i0];
            assert!((got as f64 - acc).abs() < 1e-3, "({i0},{i1}): {got} vs {acc}");
        }
    }
}
```

- [ ] **Step 2: Проверить падение.**

- [ ] **Step 3: Реализация**

Билдер в `context.rs`:
```rust
pub fn mul_mat(&mut self, a: TensorId, b: TensorId) -> TensorId {
    let ta = self.t(a);
    let tb = self.t(b);
    assert_eq!(ta.ne[0], tb.ne[0], "mul_mat: несовпадение k");
    assert!(tb.ne[2] % ta.ne[2] == 0 && tb.ne[3] % ta.ne[3] == 0, "mul_mat: broadcast a по dims 2,3");
    let ne = [ta.ne[1], tb.ne[1], tb.ne[2], tb.ne[3]];
    let dst = self.new_tensor(DType::F32, ne);
    let d = self.t_mut(dst);
    d.op = Op::MulMat;
    d.src = [Some(a), Some(b), None, None];
    dst
}
```

`crates/ggrs-core/src/kernels/mulmat.rs`:
```rust
use crate::context::Context;
use crate::simd;
use crate::tensor::TensorId;
use super::{row_ptr, split};

pub fn mul_mat(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let b = ctx.t(dst.src[1].unwrap());
    assert_eq!(a.nb[0], 4, "mul_mat: строки a должны быть плотными (сделай cont)");
    assert_eq!(b.nb[0], 4, "mul_mat: строки b должны быть плотными (сделай cont)");
    let k = a.ne[0];
    let m = a.ne[1];
    // работа: все строки dst = b.ne1 * b.ne2 * b.ne3; делим их между потоками
    let nr = dst.ne[1] * dst.ne[2] * dst.ne[3];
    let (ir0, ir1) = split(nr, ith, nth);
    for ir in ir0..ir1 {
        let i3 = ir / (dst.ne[1] * dst.ne[2]);
        let i2 = (ir / dst.ne[1]) % dst.ne[2];
        let i1 = ir % dst.ne[1];
        let a2 = i2 % a.ne[2];
        let a3 = i3 % a.ne[3];
        unsafe {
            let pb = row_ptr(ctx, b, i1, i2, i3) as *const f32;
            let brow = std::slice::from_raw_parts(pb, k);
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            for i0 in 0..m {
                let pa = (ctx.base().add(a.offset + i0 * a.nb[1] + a2 * a.nb[2] + a3 * a.nb[3])) as *const f32;
                let arow = std::slice::from_raw_parts(pa, k);
                *pd.add(i0) = simd::vec_dot(arow, brow);
            }
        }
    }
}
```

В `kernels/mod.rs`: `pub mod mulmat;` и `Op::MulMat => mulmat::mul_mat(ctx, id, ith, nth),`.

- [ ] **Step 4: Тесты зелёные.**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): mul_mat с семантикой ggml на vec_dot (AVX2)"`

---

### Task 8: SoftMax и RmsNorm

**Files:**
- Create: `crates/ggrs-core/src/kernels/rows.rs`
- Modify: `context.rs`, `kernels/mod.rs`, тест `crates/ggrs-core/tests/ops_rows.rs`

**Interfaces:**
- Produces: `Context::soft_max(a)` (построчный, численно стабильный), `Context::rms_norm(a, eps: f32)` (eps в `op_params[0]`, без умножения на веса — как `ggml_rms_norm`).

- [ ] **Step 1: Падающий тест**

```rust
use ggrs_core::*;

#[test]
fn softmax_rows() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2);
    ctx.set_f32(a, &[1., 2., 3., 1000., 1000., 1000.]);
    let s = ctx.soft_max(a);
    let g = build_forward(&ctx, s);
    compute(&ctx, &g, 1);
    let v = ctx.data_f32(s);
    let e = |x: f32| x.exp();
    let z = e(1.) + e(2.) + e(3.);
    assert!((v[0] - e(1.) / z).abs() < 1e-6);
    assert!((v[1] - e(2.) / z).abs() < 1e-6);
    assert!((v[2] - e(3.) / z).abs() < 1e-6);
    // большие значения не дают NaN (стабильность через вычитание max)
    for i in 3..6 {
        assert!((v[i] - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn rms_norm_row() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 4, 1);
    ctx.set_f32(a, &[1., 2., 3., 4.]);
    let r = ctx.rms_norm(a, 1e-5);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    let ms = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let v = ctx.data_f32(r);
    for (i, &x) in [1.0f32, 2., 3., 4.].iter().enumerate() {
        assert!((v[i] - x * inv).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Проверить падение.**

- [ ] **Step 3: Реализация**

Билдеры: `soft_max` через `unary_op(Op::SoftMax, a)`; `rms_norm`:
```rust
pub fn soft_max(&mut self, a: TensorId) -> TensorId {
    self.unary_op(Op::SoftMax, a)
}
pub fn rms_norm(&mut self, a: TensorId, eps: f32) -> TensorId {
    let dst = self.unary_op(Op::RmsNorm, a);
    self.t_mut(dst).op_params[0] = eps.to_bits();
    dst
}
```

`crates/ggrs-core/src/kernels/rows.rs`:
```rust
use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn soft_max(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let mut max = f32::NEG_INFINITY;
            for i in 0..ne0 {
                max = max.max(*pa.add(i));
            }
            let mut sum = 0.0f32;
            for i in 0..ne0 {
                let e = (*pa.add(i) - max).exp();
                *pd.add(i) = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for i in 0..ne0 {
                *pd.add(i) *= inv;
            }
        }
    }
}

pub fn rms_norm(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let eps = f32::from_bits(dst.op_params[0]);
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let mut ss = 0.0f32;
            for i in 0..ne0 {
                let x = *pa.add(i);
                ss += x * x;
            }
            let inv = 1.0 / (ss / ne0 as f32 + eps).sqrt();
            for i in 0..ne0 {
                *pd.add(i) = *pa.add(i) * inv;
            }
        }
    }
}
```

Диспатч: `Op::SoftMax => rows::soft_max(...)`, `Op::RmsNorm => rows::rms_norm(...)`.

- [ ] **Step 4: Тесты зелёные.**  
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): soft_max и rms_norm"`

---

### Task 9: GetRows и Cont

**Files:**
- Create: `crates/ggrs-core/src/kernels/copy.rs`
- Modify: `context.rs`, `kernels/mod.rs`, тест `crates/ggrs-core/tests/ops_copy.rs`

**Interfaces:**
- Produces:
  - `Context::get_rows(a, ids)` — a: F32 `[ne0, n]`, ids: I32 1D `[m]` → dst F32 `[ne0, m]`
  - `Context::cont(a)` — материализация любого view (permute/transpose) в новый непрерывный тензор той же формы

- [ ] **Step 1: Падающий тест**

```rust
use ggrs_core::*;

#[test]
fn get_rows_basic() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 2, 3); // 3 строки по 2
    ctx.set_f32(a, &[0., 1., 10., 11., 20., 21.]);
    let ids = ctx.new_tensor_1d(DType::I32, 2);
    ctx.set_i32(ids, &[2, 0]);
    let r = ctx.get_rows(a, ids);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    assert_eq!(ctx.data_f32(r), &[20., 21., 0., 1.]);
}

#[test]
fn cont_materializes_transpose() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F32, 3, 2); // [[0,1,2],[3,4,5]]
    ctx.set_f32(a, &[0., 1., 2., 3., 4., 5.]);
    let t = ctx.transpose(a); // логически [[0,3],[1,4],[2,5]]
    let c = ctx.cont(t);
    let g = build_forward(&ctx, c);
    compute(&ctx, &g, 1);
    assert!(ctx.t(c).is_contiguous());
    assert_eq!(ctx.data_f32(c), &[0., 3., 1., 4., 2., 5.]);
}
```

- [ ] **Step 2: Проверить падение.**

- [ ] **Step 3: Реализация**

Билдеры:
```rust
pub fn get_rows(&mut self, a: TensorId, ids: TensorId) -> TensorId {
    let ta = self.t(a);
    let tids = self.t(ids);
    assert_eq!(tids.dtype, DType::I32);
    let ne = [ta.ne[0], tids.ne[0], 1, 1];
    let dst = self.new_tensor(DType::F32, ne);
    let d = self.t_mut(dst);
    d.op = Op::GetRows;
    d.src = [Some(a), Some(ids), None, None];
    dst
}
pub fn cont(&mut self, a: TensorId) -> TensorId {
    let ne = self.t(a).ne;
    let dtype = self.t(a).dtype;
    let dst = self.new_tensor(dtype, ne);
    let d = self.t_mut(dst);
    d.op = Op::Cont;
    d.src = [Some(a), None, None, None];
    dst
}
```

`crates/ggrs-core/src/kernels/copy.rs`:
```rust
use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn get_rows(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let ids = ctx.data_i32(dst.src[1].unwrap());
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.ne[1], ith, nth);
    for ir in ir0..ir1 {
        let row = ids[ir] as usize;
        assert!(row < a.ne[1], "get_rows: индекс {row} вне диапазона");
        unsafe {
            let ps = row_ptr(ctx, a, row, 0, 0) as *const f32;
            let pd = row_ptr(ctx, dst, ir, 0, 0) as *mut f32;
            std::ptr::copy_nonoverlapping(ps, pd, ne0);
        }
    }
}

pub fn cont(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir);
        unsafe {
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            let base = ctx.base().add(a.offset + i1 * a.nb[1] + i2 * a.nb[2] + i3 * a.nb[3]);
            for i0 in 0..ne0 {
                *pd.add(i0) = *(base.add(i0 * a.nb[0]) as *const f32);
            }
        }
    }
}
```

Диспатч: `Op::GetRows => copy::get_rows(...)`, `Op::Cont => copy::cont(...)`.

- [ ] **Step 4: Тесты зелёные.**  
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): get_rows и cont (материализация views)"`

---

### Task 10: RoPE (режим NORM, как у LLaMA) и CrossEntropyLoss

**Files:**
- Create: `crates/ggrs-core/src/kernels/rope.rs`, `crates/ggrs-core/src/kernels/loss.rs`
- Modify: `context.rs`, `kernels/mod.rs`, тест `crates/ggrs-core/tests/ops_rope_loss.rs`

**Interfaces:**
- Produces:
  - `Context::rope(a, pos, n_dims: usize, base: f32)` — a: `[head_dim, n_head, T, B]`, pos: I32 `[T]`; NORM-режим: пары (x[2i], x[2i+1]), theta_i = pos·base^(−2i/n_dims); op_params[0]=n_dims, op_params[1]=base.to_bits()
  - `Context::cross_entropy_loss(logits, targets)` — logits/targets `[vocab, rows]` одной формы (targets — распределения), dst — скаляр `[1]`: `−(1/rows)·Σ targets·log_softmax(logits)`

- [ ] **Step 1: Падающий тест**

```rust
use ggrs_core::*;

#[test]
fn rope_rotates_pairs() {
    let mut ctx = Context::new(1 << 20);
    // head_dim=4, 1 голова, 2 позиции
    let a = ctx.new_tensor_3d(DType::F32, 4, 1, 2);
    ctx.set_f32(a, &[1., 0., 1., 0., 1., 0., 1., 0.]);
    let pos = ctx.new_tensor_1d(DType::I32, 2);
    ctx.set_i32(pos, &[0, 1]);
    let r = ctx.rope(a, pos, 4, 10000.0);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 1);
    let v = ctx.data_f32(r);
    // pos=0: без изменений
    assert!((v[0] - 1.0).abs() < 1e-6 && v[1].abs() < 1e-6);
    // pos=1, пара 0: theta = 1 * 10000^(0) = 1.0 → (cos1, sin1)
    assert!((v[4] - 1f32.cos()).abs() < 1e-6);
    assert!((v[5] - 1f32.sin()).abs() < 1e-6);
    // pos=1, пара 1: theta = 10000^(-2/4) = 0.01 → (cos0.01, sin0.01)
    assert!((v[6] - 0.01f32.cos()).abs() < 1e-6);
    assert!((v[7] - 0.01f32.sin()).abs() < 1e-6);
    // норма пары сохраняется
    assert!((v[4] * v[4] + v[5] * v[5] - 1.0).abs() < 1e-5);
}

#[test]
fn cross_entropy_uniform() {
    let mut ctx = Context::new(1 << 20);
    let logits = ctx.new_tensor_2d(DType::F32, 4, 2);
    ctx.set_f32(logits, &[0.; 8]); // равномерные логиты
    let targets = ctx.new_tensor_2d(DType::F32, 4, 2);
    ctx.set_f32(targets, &[1., 0., 0., 0., 0., 1., 0., 0.]); // one-hot
    let l = ctx.cross_entropy_loss(logits, targets);
    let g = build_forward(&ctx, l);
    compute(&ctx, &g, 1);
    // loss = -log(1/4) = ln4
    assert!((ctx.data_f32(l)[0] - 4.0f32.ln()).abs() < 1e-5);
}
```

- [ ] **Step 2: Проверить падение.**

- [ ] **Step 3: Реализация**

Билдеры:
```rust
pub fn rope(&mut self, a: TensorId, pos: TensorId, n_dims: usize, base: f32) -> TensorId {
    assert_eq!(self.t(pos).dtype, DType::I32);
    assert!(n_dims % 2 == 0 && n_dims <= self.t(a).ne[0]);
    let dst = self.unary_op(Op::Rope, a);
    let d = self.t_mut(dst);
    d.src[1] = Some(pos);
    d.op_params[0] = n_dims as u32;
    d.op_params[1] = base.to_bits();
    dst
}
pub fn cross_entropy_loss(&mut self, logits: TensorId, targets: TensorId) -> TensorId {
    assert!(self.t(logits).same_shape(self.t(targets)));
    let dst = self.new_tensor_1d(DType::F32, 1);
    let d = self.t_mut(dst);
    d.op = Op::CrossEntropyLoss;
    d.src = [Some(logits), Some(targets), None, None];
    dst
}
```

`crates/ggrs-core/src/kernels/rope.rs`:
```rust
use crate::context::Context;
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn rope(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let pos = ctx.data_i32(dst.src[1].unwrap());
    let n_dims = dst.op_params[0] as usize;
    let base = f32::from_bits(dst.op_params[1]);
    assert_eq!(a.nb[0], 4);
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    for ir in ir0..ir1 {
        let (i1, i2, i3) = unravel_row(dst, ir); // i1=голова, i2=позиция
        let p = pos[i2] as f32;
        unsafe {
            let pa = row_ptr(ctx, a, i1, i2, i3) as *const f32;
            let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
            for i in 0..n_dims / 2 {
                let theta = p * base.powf(-2.0 * i as f32 / n_dims as f32);
                let (sin_t, cos_t) = theta.sin_cos();
                let x0 = *pa.add(2 * i);
                let x1 = *pa.add(2 * i + 1);
                *pd.add(2 * i) = x0 * cos_t - x1 * sin_t;
                *pd.add(2 * i + 1) = x0 * sin_t + x1 * cos_t;
            }
            for i in n_dims..ne0 {
                *pd.add(i) = *pa.add(i);
            }
        }
    }
}
```

`crates/ggrs-core/src/kernels/loss.rs`:
```rust
use crate::context::Context;
use crate::tensor::TensorId;
use super::row_ptr;

/// Одним потоком (ith==0), Фаза 1. Численно стабильный log_softmax.
pub fn cross_entropy_loss(ctx: &Context, dst_id: TensorId, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let dst = ctx.t(dst_id);
    let logits = ctx.t(dst.src[0].unwrap());
    let targets = ctx.t(dst.src[1].unwrap());
    assert_eq!(logits.nb[0], 4);
    assert_eq!(targets.nb[0], 4);
    let ne0 = logits.ne[0];
    let nrows = logits.nrows();
    let mut total = 0.0f64;
    for ir in 0..nrows {
        let i3 = ir / (logits.ne[1] * logits.ne[2]);
        let i2 = (ir / logits.ne[1]) % logits.ne[2];
        let i1 = ir % logits.ne[1];
        unsafe {
            let pl = row_ptr(ctx, logits, i1, i2, i3) as *const f32;
            let pt = row_ptr(ctx, targets, i1, i2, i3) as *const f32;
            let mut max = f32::NEG_INFINITY;
            for i in 0..ne0 {
                max = max.max(*pl.add(i));
            }
            let mut sum = 0.0f32;
            for i in 0..ne0 {
                sum += (*pl.add(i) - max).exp();
            }
            let log_z = sum.ln() + max;
            for i in 0..ne0 {
                let t = *pt.add(i);
                if t != 0.0 {
                    total += (t * (*pl.add(i) - log_z)) as f64;
                }
            }
        }
    }
    unsafe {
        let pd = (ctx.base().add(dst.offset)) as *mut f32;
        *pd = -(total / nrows as f64) as f32;
    }
}
```

Диспатч: `Op::Rope => rope::rope(...)`, `Op::CrossEntropyLoss => loss::cross_entropy_loss(...)`.

- [ ] **Step 4: Тесты зелёные.**  
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): rope (NORM) и cross_entropy_loss"`

---

### Task 11: Многопоточность — барьеры как в ggml

**Files:**
- Modify: `crates/ggrs-core/src/compute.rs`
- Test: `crates/ggrs-core/tests/threading.rs`

**Interfaces:**
- Produces: `compute(ctx, graph, n_threads)` реально исполняет n_threads потоками: каждый поток идёт по узлам графа, вызывает ядро со своим `(ith, nth)`, после каждого узла — `Barrier::wait()`.

- [ ] **Step 1: Падающий тест**

```rust
use ggrs_core::*;

fn build_case(ctx: &mut Context) -> TensorId {
    let k = 96usize;
    let m = 33;
    let n = 47;
    let a = ctx.new_tensor_2d(DType::F32, k, m);
    let b = ctx.new_tensor_2d(DType::F32, k, n);
    let av: Vec<f32> = (0..k * m).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| (i % 23) as f32 * 0.07 - 0.7).collect();
    ctx.set_f32(a, &av);
    ctx.set_f32(b, &bv);
    let d = ctx.mul_mat(a, b);
    let s = ctx.soft_max(d);
    ctx.rms_norm(s, 1e-5)
}

#[test]
fn threads_produce_identical_results() {
    let mut c1 = Context::new(1 << 24);
    let r1 = build_case(&mut c1);
    let g1 = build_forward(&c1, r1);
    compute(&c1, &g1, 1);

    let mut c4 = Context::new(1 << 24);
    let r4 = build_case(&mut c4);
    let g4 = build_forward(&c4, r4);
    compute(&c4, &g4, 4);

    // деление по строкам не меняет порядок редукций → бит-в-бит
    assert_eq!(c1.data_f32(r1), c4.data_f32(r4));
}
```

- [ ] **Step 2: Проверить падение** — тест уже проходит? Нет: multi-thread ветки нет, `compute` игнорирует n_threads и тест ПРОЙДЁТ ложно. Поэтому сначала замени тело `compute` на `todo!()` для n_threads>1? Не нужно — правильный порядок: реализуй ветку, тест подтверждает эквивалентность. Этот тест — regression-тест корректности, а не TDD-провал; допустимое исключение.

- [ ] **Step 3: Реализация** (замена `compute.rs`)

```rust
use std::sync::Barrier;

use crate::context::Context;
use crate::graph::Graph;
use crate::kernels;

pub fn compute(ctx: &Context, graph: &Graph, n_threads: usize) {
    assert!(n_threads >= 1);
    if n_threads == 1 {
        for &id in &graph.nodes {
            kernels::dispatch(ctx, id, 0, 1);
        }
        return;
    }
    let barrier = Barrier::new(n_threads);
    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let barrier = &barrier;
            s.spawn(move || {
                for &id in &graph.nodes {
                    kernels::dispatch(ctx, id, ith, n_threads);
                    barrier.wait();
                }
            });
        }
    });
}
```

(`Context` уже `Sync` благодаря `unsafe impl Sync for Arena`; потоки пишут в непересекающиеся строки dst.)

- [ ] **Step 4: Тесты зелёные** — `cargo test -p ggrs-core` (все прежние + threading).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): многопоточное исполнение графа с барьерами"`

---

### Task 12: Numpy-эталоны

**Files:**
- Create: `tools/gen_fixtures.py`, `crates/ggrs-core/tests/fixtures.rs`
- Commit артефакта: `crates/ggrs-core/tests/fixtures/ops.bin`

**Interfaces:**
- Produces: бинарный формат фикстур: `[u32 n_tensors]`, затем для каждого: `[u32 name_len][name utf8][u32 dtype: 0=f32,1=i32][u32 ne0..ne3][raw data little-endian]`. Python пишет, Rust-тест читает и сравнивает с нашими ядрами (atol=1e-5, rtol=1e-5).

- [ ] **Step 1: Установить numpy и написать генератор**

Run: `pip install numpy`

`tools/gen_fixtures.py`:
```python
"""Генерирует эталонные фикстуры для ggrs-core из numpy. Запуск из корня репо."""
import struct
import numpy as np

rng = np.random.default_rng(42)
out = {}

# mulmat: a[m=5,k=7] строки, b[n=3,k=7]; ggml: dst[n,m], dst[i1,i0] = dot(a[i0], b[i1])
a = rng.standard_normal((5, 7)).astype(np.float32)
b = rng.standard_normal((3, 7)).astype(np.float32)
out["mulmat.a"] = a          # ne = [7, 5]
out["mulmat.b"] = b          # ne = [7, 3]
out["mulmat.out"] = (b @ a.T).astype(np.float32)  # shape [3, 5] → ne [5, 3]

# softmax по строкам
x = rng.standard_normal((4, 6)).astype(np.float32) * 3
e = np.exp(x - x.max(axis=1, keepdims=True))
out["softmax.x"] = x
out["softmax.out"] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

# rms_norm, eps=1e-5
x = rng.standard_normal((3, 8)).astype(np.float32)
out["rmsnorm.x"] = x
inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-5)
out["rmsnorm.out"] = (x * inv).astype(np.float32)

# rope NORM: head_dim=8, n_head=2, T=3, base=10000
hd, nh, T = 8, 2, 3
x = rng.standard_normal((T, nh, hd)).astype(np.float32)  # ne = [hd, nh, T]
pos = np.array([0, 1, 2], dtype=np.int32)
y = x.copy()
for t in range(T):
    for h in range(nh):
        for i in range(hd // 2):
            theta = pos[t] * (10000.0 ** (-2.0 * i / hd))
            c, s = np.cos(theta), np.sin(theta)
            x0, x1 = x[t, h, 2 * i], x[t, h, 2 * i + 1]
            y[t, h, 2 * i] = x0 * c - x1 * s
            y[t, h, 2 * i + 1] = x0 * s + x1 * c
out["rope.x"] = x
out["rope.pos"] = pos
out["rope.out"] = y.astype(np.float32)

# cross_entropy: logits [4 строки, vocab=10], one-hot targets
lg = rng.standard_normal((4, 10)).astype(np.float32)
tgt = np.zeros((4, 10), dtype=np.float32)
for r, c in enumerate([1, 0, 7, 3]):
    tgt[r, c] = 1.0
lz = lg - lg.max(axis=1, keepdims=True)
logsm = lz - np.log(np.exp(lz).sum(axis=1, keepdims=True))
loss = -(tgt * logsm).sum() / 4.0
out["xent.logits"] = lg
out["xent.targets"] = tgt
out["xent.out"] = np.array([loss], dtype=np.float32)

with open("crates/ggrs-core/tests/fixtures/ops.bin", "wb") as f:
    f.write(struct.pack("<I", len(out)))
    for name, arr in out.items():
        nb = name.encode()
        f.write(struct.pack("<I", len(nb)))
        f.write(nb)
        f.write(struct.pack("<I", 0 if arr.dtype == np.float32 else 1))
        # ne: numpy shape задом наперёд (ne0 — последняя ось numpy)
        ne = list(arr.shape[::-1]) + [1] * (4 - arr.ndim)
        f.write(struct.pack("<4I", *ne))
        f.write(arr.astype("<f4" if arr.dtype == np.float32 else "<i4").tobytes())
print(f"OK: {len(out)} тензоров")
```

Run: `mkdir crates\ggrs-core\tests\fixtures; python tools\gen_fixtures.py`
Expected: `OK: 13 тензоров`

- [ ] **Step 2: Падающий Rust-тест**

`crates/ggrs-core/tests/fixtures.rs`:
```rust
use std::collections::HashMap;

use ggrs_core::*;

struct Fx {
    dtype: u32,
    ne: [usize; 4],
    f32s: Vec<f32>,
    i32s: Vec<i32>,
}

fn load() -> HashMap<String, Fx> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ops.bin")).unwrap();
    let mut p = 0usize;
    let rd_u32 = |b: &[u8], p: &mut usize| {
        let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let n = rd_u32(&bytes, &mut p);
    let mut m = HashMap::new();
    for _ in 0..n {
        let nl = rd_u32(&bytes, &mut p) as usize;
        let name = String::from_utf8(bytes[p..p + nl].to_vec()).unwrap();
        p += nl;
        let dtype = rd_u32(&bytes, &mut p);
        let ne = [
            rd_u32(&bytes, &mut p) as usize,
            rd_u32(&bytes, &mut p) as usize,
            rd_u32(&bytes, &mut p) as usize,
            rd_u32(&bytes, &mut p) as usize,
        ];
        let count: usize = ne.iter().product();
        let mut f32s = Vec::new();
        let mut i32s = Vec::new();
        for i in 0..count {
            let raw = &bytes[p + i * 4..p + i * 4 + 4];
            if dtype == 0 {
                f32s.push(f32::from_le_bytes(raw.try_into().unwrap()));
            } else {
                i32s.push(i32::from_le_bytes(raw.try_into().unwrap()));
            }
        }
        p += count * 4;
        m.insert(name, Fx { dtype, ne, f32s, i32s });
    }
    m
}

fn assert_close(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: длина");
    for i in 0..got.len() {
        let (g, w) = (got[i], want[i]);
        assert!(
            (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
            "{label}[{i}]: {g} vs {w}"
        );
    }
}

#[test]
fn ops_match_numpy() {
    let fx = load();
    let tensor = |ctx: &mut Context, f: &Fx| -> TensorId {
        let id = ctx.new_tensor(if f.dtype == 0 { DType::F32 } else { DType::I32 }, f.ne);
        if f.dtype == 0 {
            ctx.set_f32(id, &f.f32s);
        } else {
            ctx.set_i32(id, &f.i32s);
        }
        id
    };

    // mulmat
    {
        let mut ctx = Context::new(1 << 22);
        let a = tensor(&mut ctx, &fx["mulmat.a"]);
        let b = tensor(&mut ctx, &fx["mulmat.b"]);
        let d = ctx.mul_mat(a, b);
        let g = build_forward(&ctx, d);
        compute(&ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["mulmat.out"].f32s, "mulmat");
    }
    // softmax
    {
        let mut ctx = Context::new(1 << 22);
        let x = tensor(&mut ctx, &fx["softmax.x"]);
        let d = ctx.soft_max(x);
        let g = build_forward(&ctx, d);
        compute(&ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["softmax.out"].f32s, "softmax");
    }
    // rms_norm
    {
        let mut ctx = Context::new(1 << 22);
        let x = tensor(&mut ctx, &fx["rmsnorm.x"]);
        let d = ctx.rms_norm(x, 1e-5);
        let g = build_forward(&ctx, d);
        compute(&ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["rmsnorm.out"].f32s, "rmsnorm");
    }
    // rope
    {
        let mut ctx = Context::new(1 << 22);
        let x = tensor(&mut ctx, &fx["rope.x"]);
        let pos = tensor(&mut ctx, &fx["rope.pos"]);
        let d = ctx.rope(x, pos, 8, 10000.0);
        let g = build_forward(&ctx, d);
        compute(&ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["rope.out"].f32s, "rope");
    }
    // cross_entropy
    {
        let mut ctx = Context::new(1 << 22);
        let lg = tensor(&mut ctx, &fx["xent.logits"]);
        let tg = tensor(&mut ctx, &fx["xent.targets"]);
        let d = ctx.cross_entropy_loss(lg, tg);
        let g = build_forward(&ctx, d);
        compute(&ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["xent.out"].f32s, "xent");
    }
}
```

- [ ] **Step 3: Тест зелёный** — `cargo test -p ggrs-core --test fixtures`.

- [ ] **Step 4: Commit (включая ops.bin)**

```bash
git add -A && git commit -m "test(core): numpy-эталоны для mulmat/softmax/rmsnorm/rope/xent"
```

---

### Task 13: Интеграция — forward крошечного трансформер-блока

**Files:**
- Test: `crates/ggrs-core/tests/transformer_forward.rs`

**Interfaces:**
- Consumes: всё выше. Никакого нового API.

- [ ] **Step 1: Написать интеграционный тест**

```rust
use ggrs_core::*;

/// Один трансформер-блок (1 голова, без масок упрощений не делаем — маска causal через add)
/// на псевдослучайных весах: проверяем конечность значений и эквивалентность 1 vs 8 потоков.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.2
}

fn fill(ctx: &mut Context, id: TensorId, seed: &mut u64) {
    let n = ctx.t(id).nelements();
    let v: Vec<f32> = (0..n).map(|_| lcg(seed)).collect();
    ctx.set_f32(id, &v);
}

fn build(ctx: &mut Context) -> TensorId {
    let (d, t, vocab) = (16usize, 6usize, 32usize);
    let mut seed = 7u64;

    let emb = ctx.new_tensor_2d(DType::F32, d, vocab);
    fill(ctx, emb, &mut seed);
    let wq = ctx.new_tensor_2d(DType::F32, d, d);
    let wk = ctx.new_tensor_2d(DType::F32, d, d);
    let wv = ctx.new_tensor_2d(DType::F32, d, d);
    let wo = ctx.new_tensor_2d(DType::F32, d, d);
    for w in [wq, wk, wv, wo] {
        fill(ctx, w, &mut seed);
    }
    let w_up = ctx.new_tensor_2d(DType::F32, d, 4 * d);
    let w_gate = ctx.new_tensor_2d(DType::F32, d, 4 * d);
    let w_down = ctx.new_tensor_2d(DType::F32, 4 * d, d);
    for w in [w_up, w_gate, w_down] {
        fill(ctx, w, &mut seed);
    }

    let ids = ctx.new_tensor_1d(DType::I32, t);
    ctx.set_i32(ids, &[3, 1, 4, 1, 5, 9]);
    let pos = ctx.new_tensor_1d(DType::I32, t);
    ctx.set_i32(pos, &[0, 1, 2, 3, 4, 5]);

    // causal маска [t, t]: 0 на/ниже диагонали, -1e9 выше
    let mask = ctx.new_tensor_2d(DType::F32, t, t);
    let mv: Vec<f32> = (0..t * t)
        .map(|i| if i % t > i / t { -1e9 } else { 0.0 })
        .collect();
    ctx.set_f32(mask, &mv);

    let x = ctx.get_rows(emb, ids); // [d, t]
    let xn = ctx.rms_norm(x, 1e-5);

    // 1 голова: q,k,v = [d, t]
    let q0 = ctx.mul_mat(wq, xn);
    let k0 = ctx.mul_mat(wk, xn);
    let v0 = ctx.mul_mat(wv, xn);
    let q = ctx.reshape_3d(q0, d, 1, t);
    let k = ctx.reshape_3d(k0, d, 1, t);
    let qr0 = ctx.rope(q, pos, d, 10000.0);
    let kr0 = ctx.rope(k, pos, d, 10000.0);
    let qr = ctx.reshape_2d(qr0, d, t);
    let kr = ctx.reshape_2d(kr0, d, t);

    let att0 = ctx.mul_mat(kr, qr); // [t(k-строки), t(q-строки)]
    let att1 = ctx.scale(att0, 1.0 / (d as f32).sqrt());
    let att2 = ctx.add(att1, mask);
    let att = ctx.soft_max(att2);

    // out[d, t]: для каждого q-токена — взвешенная сумма v; v^T [t, d] строками
    let vt0 = ctx.transpose(v0); // [t, d] view
    let vt = ctx.cont(vt0);
    let out0 = ctx.mul_mat(vt, att); // [d, t]
    let att_out = ctx.mul_mat(wo, out0);

    let h = ctx.add(x, att_out);
    let hn = ctx.rms_norm(h, 1e-5);
    let up = ctx.mul_mat(w_up, hn);
    let gate = ctx.mul_mat(w_gate, hn);
    let gate_s = ctx.silu(gate);
    let ff0 = ctx.mul(gate_s, up);
    let ff = ctx.mul_mat(w_down, ff0);
    let h2 = ctx.add(h, ff);

    let logits = ctx.mul_mat(emb, h2); // tied embeddings: [vocab, t]

    // one-hot цели: следующий токен
    let targets = ctx.new_tensor_2d(DType::F32, vocab, t);
    let mut tv = vec![0.0f32; vocab * t];
    for (r, &c) in [1i32, 4, 1, 5, 9, 2].iter().enumerate() {
        tv[r * vocab + c as usize] = 1.0;
    }
    ctx.set_f32(targets, &tv);
    ctx.cross_entropy_loss(logits, targets)
}

#[test]
fn transformer_block_forward() {
    let mut c1 = Context::new(1 << 24);
    let l1 = build(&mut c1);
    let g1 = build_forward(&c1, l1);
    compute(&c1, &g1, 1);
    let loss1 = c1.data_f32(l1)[0];
    assert!(loss1.is_finite(), "loss = {loss1}");
    // при случайных весах и vocab=32 loss ~ ln(32) ± немного
    assert!(loss1 > 1.0 && loss1 < 8.0, "loss = {loss1}");

    let mut c8 = Context::new(1 << 24);
    let l8 = build(&mut c8);
    let g8 = build_forward(&c8, l8);
    compute(&c8, &g8, 8);
    assert_eq!(loss1, c8.data_f32(l8)[0], "1 поток vs 8 потоков");
}
```

- [ ] **Step 2: Тест зелёный** — `cargo test -p ggrs-core --test transformer_forward`.
Если падает — это интеграционный смоук, отлаживать по частям (проверить формы каждого узла через `ctx.t(id).ne`).

- [ ] **Step 3: Прогнать всё**

Run: `cargo test -p ggrs-core`
Expected: все тесты PASS.

Run: `cargo clippy -p ggrs-core -- -D warnings` (установить: `rustup component add clippy`)
Expected: чисто. Починить, если нет.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "test(core): сквозной forward трансформер-блока, 1 vs 8 потоков"`

---

## Definition of Done (Фаза 1)

- `cargo test -p ggrs-core` зелёный: юнит-тесты операций, SIMD-паритет, потоки, numpy-эталоны, сквозной transformer-forward.
- `cargo clippy` без предупреждений.
- Все операции спеки Фазы 1 реализованы (Cpy/View сознательно исключены — Cont/Reshape покрывают потребности; отражено в спеке).
- Готова почва Фазы 2: `Tensor.is_param`, `op_params`, граф и ядра — backward добавляется поверх без перестройки.
