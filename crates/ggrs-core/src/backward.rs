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
/// ВАЖНО: вызывать не более ОДНОГО раза на контекст — повторный вызов продублирует
/// узлы градиентов и молча испортит граф (арена и список тензоров только растут).
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
            Op::Silu => {
                // silu(x): ∂x += g * silu'(x)
                let x = ctx.t(node_id).src[0].unwrap();
                let gb = ctx.silu_back(g_dst, x);
                accumulate(ctx, &mut grads, x, gb);
            }
            Op::Gelu => {
                // gelu(x): ∂x += g * gelu'(x)
                let x = ctx.t(node_id).src[0].unwrap();
                let gb = ctx.gelu_back(g_dst, x);
                accumulate(ctx, &mut grads, x, gb);
            }
            Op::SumAll => {
                let src = ctx.t(node_id).src[0].unwrap();
                // ∂src += sum_all_back(g, src)
                let g_back = ctx.sum_all_back(g_dst, src);
                accumulate(ctx, &mut grads, src, g_back);
            }
            Op::MulMat => {
                let a = ctx.t(node_id).src[0].unwrap();
                let b = ctx.t(node_id).src[1].unwrap();
                // Проверка: только 2D (Фаза 2)
                assert!(
                    ctx.t(a).ne[2] == 1 && ctx.t(a).ne[3] == 1
                        && ctx.t(b).ne[2] == 1 && ctx.t(b).ne[3] == 1,
                    "mulmat backward: 3D — Фаза 3"
                );
                // ∂a = out_prod(b, g_dst)  — b[K,N] × g_dst[M,N] → [K,M]
                let ga = ctx.out_prod(b, g_dst);
                accumulate(ctx, &mut grads, a, ga);
                // ∂b = mul_mat(cont(transpose(a)), g_dst)
                // transpose(a): [K,M] → [M,K]
                let at = ctx.transpose(a);
                let atc = ctx.cont(at);
                let gb = ctx.mul_mat(atc, g_dst);
                accumulate(ctx, &mut grads, b, gb);
            }
            Op::CrossEntropyLoss => {
                let logits = ctx.t(node_id).src[0].unwrap();
                let targets = ctx.t(node_id).src[1].unwrap();
                let gl = ctx.cross_entropy_loss_back(g_dst, logits, targets);
                accumulate(ctx, &mut grads, logits, gl);
                // targets не получают градиент
            }
            Op::GetRows => {
                let table = ctx.t(node_id).src[0].unwrap();
                let ids = ctx.t(node_id).src[1].unwrap();
                // ∂table += get_rows_back(g_dst, ids, table)
                let gt = ctx.get_rows_back(g_dst, ids, table);
                accumulate(ctx, &mut grads, table, gt);
                // ids градиента не имеют
            }
            Op::SoftMax => {
                // softmax(x): ∂x += soft_max_back(g, y), где y = ВЫХОД softmax (= сам узел)
                let y = node_id; // выход softmax — это сам узел
                let x = ctx.t(node_id).src[0].unwrap();
                let gx = ctx.soft_max_back(g_dst, y);
                accumulate(ctx, &mut grads, x, gx);
            }
            Op::RmsNorm => {
                // rms_norm(x): ∂x += rms_norm_back(g, x, eps)
                let x = ctx.t(node_id).src[0].unwrap();
                let eps = f32::from_bits(ctx.t(node_id).op_params[0]);
                let gx = ctx.rms_norm_back(g_dst, x, eps);
                accumulate(ctx, &mut grads, x, gx);
            }
            Op::Rope => {
                // rope(x, pos): ∂x += rope_back(g, pos, n_dims, base)
                let x = ctx.t(node_id).src[0].unwrap();
                let pos = ctx.t(node_id).src[1].unwrap();
                let n_dims = ctx.t(node_id).op_params[0] as usize;
                let base = f32::from_bits(ctx.t(node_id).op_params[1]);
                assert!(ctx.t(node_id).op_params[2] == 0,
                    "backward для rope_back не поддержан");
                let gx = ctx.rope_back(g_dst, pos, n_dims, base);
                accumulate(ctx, &mut grads, x, gx);
            }
            Op::Reshape => {
                // reshape(a, ne): g_a = reshape_like(g_dst, a)
                let x = ctx.t(node_id).src[0].unwrap();
                let gx = ctx.reshape_like(g_dst, x);
                accumulate(ctx, &mut grads, x, gx);
            }
            Op::Permute => {
                // permute(a, axes): g_a = cont(permute(g_dst, inv_axes))
                let x = ctx.t(node_id).src[0].unwrap();
                let axes = [
                    ctx.t(node_id).op_params[0] as usize,
                    ctx.t(node_id).op_params[1] as usize,
                    ctx.t(node_id).op_params[2] as usize,
                    ctx.t(node_id).op_params[3] as usize,
                ];
                // inverse: inv[axes[i]] = i
                let mut inv = [0usize; 4];
                for i in 0..4 {
                    inv[axes[i]] = i;
                }
                let gp = ctx.permute(g_dst, inv);
                let gx = ctx.cont(gp);
                accumulate(ctx, &mut grads, x, gx);
            }
            Op::Cont => {
                // cont(a): g_a = g_dst (форма совпадает).
                // Если a не-contiguous, g_dst всё равно имеет ту же ne — pass-through.
                let x = ctx.t(node_id).src[0].unwrap();
                accumulate(ctx, &mut grads, x, g_dst);
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
        .filter(|&(&id, _)| ctx.t(id).is_param)
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
