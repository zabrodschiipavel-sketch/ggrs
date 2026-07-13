use std::collections::HashMap;
use std::sync::Barrier;
use std::time::Instant;

use crate::context::Context;
use crate::graph::Graph;
use crate::kernels;
use crate::op::Op;

pub fn compute(ctx: &Context, graph: &Graph, n_threads: usize) {
    // Профилирование по env GGRS_PROFILE=1
    if let Ok(val) = std::env::var("GGRS_PROFILE") {
        if val == "1" {
            let timings = compute_profiled(ctx, graph, n_threads);
            let total_ms: f64 = timings.iter().map(|(_, _, ms)| ms).sum();
            eprintln!("{:12} {:>5} {:>10} {:>8}", "Op", "calls", "ms", "%");
            for (op, calls, ms) in &timings {
                let pct = if total_ms > 0.0 { ms / total_ms * 100.0 } else { 0.0 };
                eprintln!("{:12} {:>5} {:>10.3} {:>7.2}%", format!("{:?}", op), calls, ms, pct);
            }
            return;
        }
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
pub fn compute_profiled(ctx: &Context, graph: &Graph, _n_threads: usize) -> Vec<(Op, u32, f64)> {
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
