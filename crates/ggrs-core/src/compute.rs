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
