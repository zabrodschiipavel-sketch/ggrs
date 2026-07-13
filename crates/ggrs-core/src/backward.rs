use std::collections::HashMap;

use crate::context::Context;
use crate::graph::Graph;
use crate::op::Op;
use crate::tensor::TensorId;

/// Результат построения графа обратного распространения.
pub struct Backward {
    /// Для каждого узла исходного графа — соответствующий тензор градиента.
    pub grads: HashMap<TensorId, TensorId>,
    /// Корень backward-графа: collect всех градиентов is_param-узлов.
    pub root: TensorId,
}

/// Построить граф обратного распространения для заданной функции потерь.
///
/// * `ctx` — контекст (мутабельный, т.к. добавляет новые тензоры для градиентов).
/// * `gf` — forward-граф (нужен для топологического обхода).
/// * `loss` — тензор-скаляр (выход функции потерь).
///
/// Возвращает `Backward`, где `grads` отображает каждый узел исходного графа в его градиент.
/// Контракт: `grads[loss]` — новый F32-тензор формы loss, залитый единицами.
pub fn build_backward(ctx: &mut Context, gf: &Graph, loss: TensorId) -> Backward {
    let mut grads: HashMap<TensorId, TensorId> = HashMap::new();

    // Инициализация градиента потерь: единичный тензор той же формы, что loss.
    {
        let ne = ctx.t(loss).ne;
        let g_loss = ctx.new_tensor(ctx.t(loss).dtype, ne);
        // Заливаем единицами
        let data = ctx.data_f32_mut(g_loss);
        data.fill(1.0);
        grads.insert(loss, g_loss);
    }

    // Обратный обход узлов gf.nodes (они уже в топологическом порядке: источники раньше потребителей).
    // Нам нужен обратный порядок: от loss к источникам.
    for &node_id in gf.nodes.iter().rev() {
        let op = ctx.t(node_id).op;
        let Some(&g_dst) = grads.get(&node_id) else {
            // Если для узла нет градиента — не продолжаем (входы, Op::None).
            continue;
        };

        match op {
            Op::None => {
                // Остановка: ничего не делаем, входные узлы не получают градиент.
            }
            Op::Add => {
                let src0 = ctx.t(node_id).src[0].unwrap();
                let src1 = ctx.t(node_id).src[1].unwrap();
                // assert same_shape
                assert!(
                    ctx.t(src0).same_shape(ctx.t(src1)),
                    "Add backward: несовпадение форм {:?} и {:?}",
                    ctx.t(src0).ne,
                    ctx.t(src1).ne
                );
                // ∂a += g; ∂b += g
                accumulate(ctx, &mut grads, src0, g_dst);
                accumulate(ctx, &mut grads, src1, g_dst);
            }
            Op::Mul => {
                let src0 = ctx.t(node_id).src[0].unwrap();
                let src1 = ctx.t(node_id).src[1].unwrap();
                // ∂a += mul(g, b)
                let b = ctx.t(node_id).src[1].unwrap();
                let g_mul_b = ctx.mul(g_dst, b);
                accumulate(ctx, &mut grads, src0, g_mul_b);
                // ∂b += mul(g, a)
                let a = ctx.t(node_id).src[0].unwrap();
                let g_mul_a = ctx.mul(g_dst, a);
                accumulate(ctx, &mut grads, src1, g_mul_a);
            }
            Op::Scale => {
                let src = ctx.t(node_id).src[0].unwrap();
                let s = f32::from_bits(ctx.t(node_id).op_params[0]);
                // ∂a += scale(g, s)
                let g_scaled = ctx.scale(g_dst, s);
                accumulate(ctx, &mut grads, src, g_scaled);
            }
            Op::SumAll => {
                let src = ctx.t(node_id).src[0].unwrap();
                // ∂src += sum_all_back(g, src)
                let g_back = ctx.sum_all_back(g_dst, src);
                accumulate(ctx, &mut grads, src, g_back);
            }
            _ => {
                // Любой другой op с ненулевым grads[dst] — паника (задача T3+)
                if grads.contains_key(&node_id) {
                    panic!("backward для {:?} не реализован (задача T3+)", op);
                }
            }
        }
    }

    // Собираем градиенты всех is_param узлов.
    let param_grads: Vec<TensorId> = grads
        .iter()
        .filter(|(&id, _)| ctx.t(id).is_param)
        .map(|(_, &g)| g)
        .collect();

    let root = if param_grads.is_empty() {
        // Нет is_param-узлов — корнем служит градиент лосса (он есть всегда).
        grads[&loss]
    } else {
        ctx.collect(&param_grads)
    };

    Backward { grads, root }
}

/// Добавить вклад градиента: если для src уже есть градиент — складываем (add), иначе сохраняем.
fn accumulate(ctx: &mut Context, grads: &mut HashMap<TensorId, TensorId>, src: TensorId, contrib: TensorId) {
    match grads.entry(src) {
        std::collections::hash_map::Entry::Occupied(mut e) => {
            let old = *e.get();
            let acc = ctx.add(old, contrib);
            e.insert(acc);
        }
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(contrib);
        }
    }
}
