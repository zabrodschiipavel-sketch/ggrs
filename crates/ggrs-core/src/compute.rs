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
