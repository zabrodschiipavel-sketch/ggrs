use std::collections::HashMap;
use std::sync::Barrier;
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
