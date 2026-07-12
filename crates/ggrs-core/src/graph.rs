use crate::context::Context;
use crate::tensor::TensorId;

pub struct Graph {
    pub nodes: Vec<TensorId>,
}

pub fn build_forward(ctx: &Context, result: TensorId) -> Graph {
    let mut visited = vec![false; ctx.n_tensors()];
    let mut nodes = Vec::new();
    
    // Итеративный DFS с явным стеком (post-order: источники раньше потребителей).
    let mut stack = vec![(result, false)];

    while let Some((id, expanded)) = stack.pop() {
        if expanded {
            // Второй проход — узел попадает в пост-порядок ровно один раз.
            nodes.push(id);
            continue;
        }
        if visited[id.0] {
            continue;
        }
        visited[id.0] = true;
        stack.push((id, true));
        for &src in ctx.t(id).src.iter().flatten() {
            if !visited[src.0] {
                stack.push((src, false));
            }
        }
    }
    
    Graph { nodes }
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
