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
