use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use crate::context::Context;
use crate::graph::Graph;
use crate::kernels;
use crate::op::Op;

/// Исполнить граф. Принимает `&mut Context`: исполнение пишет в тензоры арены,
/// и эксклюзивное заимствование на уровне типов запрещает два одновременных
/// compute на одном контексте (data race — аудит P0). Внутри один вызов
/// расшаривает контекст между worker-потоками (Arena: Sync); дисциплина
/// не-алиасинга по строкам — на ядрах, как в ggml.
pub fn compute(ctx: &mut Context, graph: &Graph, n_threads: usize) {
    assert!(n_threads >= 1);
    let ctx: &Context = &*ctx; // внутренняя деградация до разделяемой ссылки
    // Профилирование по env GGRS_PROFILE=1 (let-chain — edition 2024)
    if let Ok(val) = std::env::var("GGRS_PROFILE")
        && val == "1"
    {
        let timings = profiled_inner(ctx, graph);
        let total_ms: f64 = timings.iter().map(|(_, _, ms)| ms).sum();
        eprintln!("{:12} {:>5} {:>10} {:>8}", "Op", "calls", "ms", "%");
        for (op, calls, ms) in &timings {
            let pct = if total_ms > 0.0 { ms / total_ms * 100.0 } else { 0.0 };
            eprintln!("{:12} {:>5} {:>10.3} {:>7.2}%", format!("{:?}", op), calls, ms, pct);
        }
        return;
    }

    if n_threads == 1 {
        for &id in &graph.nodes {
            kernels::dispatch(ctx, id, 0, 1);
        }
        return;
    }
    let barrier = ComputeBarrier::new(n_threads);
    let failure = std::thread::scope(|s| {
        let mut workers = Vec::with_capacity(n_threads);
        let mut failure: Option<Box<dyn std::any::Any + Send>> = None;
        for ith in 0..n_threads {
            let barrier = &barrier;
            let worker = std::thread::Builder::new().spawn_scoped(s, move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    for &id in &graph.nodes {
                        kernels::dispatch(ctx, id, ith, n_threads);
                        if !barrier.wait() {
                            break;
                        }
                    }
                }));
                if result.is_err() {
                    barrier.cancel();
                }
                result
            });

            match worker {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    // Already started workers must not wait for a missing worker.
                    barrier.cancel();
                    failure = Some(Box::new(format!("failed to spawn compute worker: {error}")));
                    break;
                }
            }
        }
        for worker in workers {
            if let Err(payload) = worker.join().and_then(|result| result)
                && failure.is_none()
            {
                failure = Some(payload);
            }
        }
        failure
    });
    if let Some(payload) = failure {
        resume_unwind(payload);
    }
}

#[derive(Default)]
struct BarrierState {
    waiting: usize,
    generation: usize,
    cancelled: bool,
}

/// Unlike std::sync::Barrier, this barrier releases waiters when a kernel panics.
/// Cancellation persists across generations, so no dependent node can start
/// after a failed node and late workers never wait for an exited worker.
struct ComputeBarrier {
    n_threads: usize,
    state: Mutex<BarrierState>,
    changed: Condvar,
}

impl ComputeBarrier {
    fn new(n_threads: usize) -> Self {
        Self {
            n_threads,
            state: Mutex::new(BarrierState::default()),
            changed: Condvar::new(),
        }
    }

    fn wait(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.cancelled {
            return false;
        }
        let generation = state.generation;
        state.waiting += 1;
        if state.waiting == self.n_threads {
            state.waiting = 0;
            state.generation += 1;
            self.changed.notify_all();
        } else {
            while state.generation == generation && !state.cancelled {
                state = self.changed.wait(state).unwrap();
            }
        }
        !state.cancelled
    }

    fn cancel(&self) {
        let mut state = self.state.lock().unwrap();
        state.cancelled = true;
        self.changed.notify_all();
    }
}

/// Исполняет граф как compute, но замеряет per-op время.
/// Однопоточный путь (n_threads игнорируется в замере).
/// Возвращает Vec, отсортированный по total_ms убыванию.
/// `&mut` — по той же причине, что у compute (эксклюзивность исполнения).
pub fn compute_profiled(ctx: &mut Context, graph: &Graph, _n_threads: usize) -> Vec<(Op, u32, f64)> {
    profiled_inner(&*ctx, graph)
}

fn profiled_inner(ctx: &Context, graph: &Graph) -> Vec<(Op, u32, f64)> {
    let mut accum: HashMap<Op, (u32, f64)> = HashMap::new();

    for &id in &graph.nodes {
        match ctx.t(id).op {
            Op::None | Op::Reshape | Op::Permute => {
                kernels::dispatch(ctx, id, 0, 1);
            }
            op => {
                let start = Instant::now();
                kernels::dispatch(ctx, id, 0, 1);
                let elapsed = start.elapsed();
                let ms = elapsed.as_secs_f64() * 1000.0;
                let entry = accum.entry(op).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += ms;
            }
        }
    }

    let mut result: Vec<(Op, u32, f64)> = accum
        .into_iter()
        .map(|(op, (calls, ms))| (op, calls, ms))
        .collect();
    result.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    result
}
