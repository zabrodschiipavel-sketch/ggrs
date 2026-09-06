use std::cell::UnsafeCell;

use crate::dtype::DType;
use crate::op::Op;
use crate::tensor::{checked_nelements, Tensor, TensorId, MAX_DIMS, MAX_SRC};
use crate::util::Lcg;

struct Arena {
    // u64-хранилище гарантирует выравнивание 8 байт для f32/i32-доступа
    // (vec<u8> давал бы выравнивание 1 — чтение *const f32 было бы UB).
    buf: Box<[UnsafeCell<u64>]>,
}
// Public writes require &mut Context, including compute. Its scoped workers
// write disjoint output rows; shared public access only reads tensor storage.
unsafe impl Sync for Arena {}

pub struct Context {
    tensors: Vec<Tensor>,
    arena: Arena,
    arena_used: usize,
}

impl Context {
    pub fn new(mem_size: usize) -> Context {
        let buf = vec![0u64; mem_size.div_ceil(8)].into_boxed_slice();
        // UnsafeCell<T> has exactly T's layout. Put interior mutability on the
        // allocation itself, so base() never reborrows all tensor data as &mut.
        let buf = unsafe { Box::from_raw(Box::into_raw(buf) as *mut [UnsafeCell<u64>]) };
        Context {
            tensors: Vec::new(),
            arena: Arena { buf },
            arena_used: 0,
        }
    }

    fn alloc(&mut self, nbytes: usize) -> usize {
        let offset = self.arena_used.checked_add(31)
            .expect("ggrs: allocation alignment overflow") & !31;
        let len = self.arena.buf.len() * 8;
        let end = offset.checked_add(nbytes).expect("ggrs: переполнение размера аллокации");
        assert!(end <= len, "ggrs: arena out of memory");
        self.arena_used = end;
        offset
    }

    /// Allocate a tensor. Every dimension must be nonzero.
    pub fn new_tensor(&mut self, dtype: DType, ne: [usize; MAX_DIMS]) -> TensorId {
        assert!(ne.iter().all(|&dim| dim > 0), "tensor dimensions must be nonzero");
        let nb0 = dtype.type_size();
        let nb1 = dtype.row_size(ne[0]);
        // Проверяемая арифметика layout: в release обычное умножение молча
        // заворачивается, создавая тензор меньше заявленного (аудит P0).
        // Валидация здесь делает инварианты (offset+idx*nb ≤ nbytes ≤ арена)
        // безопасными для всех последующих не-checked вычислений.
        let nb2 = nb1.checked_mul(ne[1]).expect("ggrs: переполнение layout (nb2)");
        let nb3 = nb2.checked_mul(ne[2]).expect("ggrs: переполнение layout (nb3)");
        let nbytes = nb3.checked_mul(ne[3]).expect("ggrs: переполнение layout (nbytes)");
        let nb = [nb0, nb1, nb2, nb3];
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

    /// Занято байт в арене (с учётом выравнивания аллокаций).
    pub fn mem_used(&self) -> usize {
        self.arena_used
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.arena.buf.as_ptr() as *mut u8
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
    pub fn data_u16(&self, id: TensorId) -> &[u16] {
        let t = self.t(id);
        assert_eq!(t.dtype, DType::F16);
        assert!(t.is_contiguous(), "data_u16: тензор не непрерывный");
        unsafe {
            std::slice::from_raw_parts(self.base().add(t.offset) as *const u16, t.nelements())
        }
    }
    fn data_u16_mut(&mut self, id: TensorId) -> &mut [u16] {
        let t = self.t(id).clone();
        assert_eq!(t.dtype, DType::F16);
        assert!(t.is_contiguous());
        unsafe {
            std::slice::from_raw_parts_mut(self.base().add(t.offset) as *mut u16, t.nelements())
        }
    }
    pub fn set_f32(&mut self, id: TensorId, vals: &[f32]) {
        self.data_f32_mut(id).copy_from_slice(vals);
    }
    pub fn set_i32(&mut self, id: TensorId, vals: &[i32]) {
        self.data_i32_mut(id).copy_from_slice(vals);
    }
    /// Запись f16 через конверсию f32→f16, поэлементно.
    pub fn set_f16(&mut self, id: TensorId, vals: &[f32]) {
        let buf = self.data_u16_mut(id);
        assert_eq!(buf.len(), vals.len(), "set_f16: длина не совпадает");
        for (dst, &src) in buf.iter_mut().zip(vals.iter()) {
            *dst = crate::dtype::f32_to_f16(src);
        }
    }

    /// Залить F32-тензор значениями N(mean, std) из rng. Тензор должен быть F32 и contiguous.
    pub fn fill_normal(&mut self, id: TensorId, mean: f32, std: f32, rng: &mut Lcg) {
        for v in self.data_f32_mut(id) {
            *v = rng.next_normal(mean, std);
        }
    }

    /// Залить F32-тензор равномерными значениями [lo, hi) из rng.
    pub fn fill_uniform(&mut self, id: TensorId, lo: f32, hi: f32, rng: &mut Lcg) {
        for v in self.data_f32_mut(id) {
            *v = lo + (rng.next_f32() + 0.5) * (hi - lo);
        }
    }

    /// Строковое чтение через страйды — работает и для views/permute.
    pub fn get_f32(&self, id: TensorId, idx: [usize; MAX_DIMS]) -> f32 {
        let t = self.t(id);
        assert_eq!(t.dtype, DType::F32);
        // Границы обязательны: смещение вне тензора читало бы чужую память арены (аудит P0).
        for d in 0..MAX_DIMS {
            assert!(idx[d] < t.ne[d], "get_f32: индекс {:?} вне формы {:?}", idx, t.ne);
        }
        let off = t.offset + idx[0] * t.nb[0] + idx[1] * t.nb[1] + idx[2] * t.nb[2] + idx[3] * t.nb[3];
        unsafe { *(self.base().add(off) as *const f32) }
    }

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
        assert_eq!(t.nelements(), checked_nelements(ne), "reshape: число элементов не совпадает");
        assert!(ne[0].is_multiple_of(t.dtype.blck_size()), "reshape: ne0 не кратен блоку");
        let ts = t.dtype.type_size();
        let rs = t.dtype.row_size(ne[0]);
        let nb2 = rs.checked_mul(ne[1]).expect("reshape: stride overflow");
        let nb3 = nb2.checked_mul(ne[2]).expect("reshape: stride overflow");
        let nb = [ts, rs, nb2, nb3];
        self.new_view(a, ne, nb, Op::Reshape)
    }

    /// Изменить форму `a` в форму `like` (Reshape-like view).
    /// `like` должен быть contiguous (иметь стандартные страйды).
    pub fn reshape_like(&mut self, a: TensorId, like: TensorId) -> TensorId {
        let ne = self.t(like).ne;
        self.reshape(a, ne)
    }

    /// axes[i] — новая позиция измерения i (семантика ggml_permute).
    /// Сохраняет axes в op_params[0..4] для использования в backward.
    pub fn permute(&mut self, a: TensorId, axes: [usize; MAX_DIMS]) -> TensorId {
        let mut seen = [false; MAX_DIMS];
        for &ax in &axes {
            assert!(ax < MAX_DIMS && !seen[ax], "permute: axes должны быть перестановкой 0..4");
            seen[ax] = true;
        }
        let t = self.t(a);
        let mut ne = [0usize; MAX_DIMS];
        let mut nb = [0usize; MAX_DIMS];
        for i in 0..MAX_DIMS {
            ne[axes[i]] = t.ne[i];
            nb[axes[i]] = t.nb[i];
        }
        let dst = self.new_view(a, ne, nb, Op::Permute);
        let d = self.t_mut(dst);
        for (i, &ax) in axes.iter().enumerate() {
            d.op_params[i] = ax as u32;
        }
        dst
    }

    pub fn transpose(&mut self, a: TensorId) -> TensorId {
        self.permute(a, [1, 0, 2, 3])
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.binary_op(Op::Add, a, b)
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.binary_op(Op::Mul, a, b)
    }

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

    /// Обратное распространение Silu: dst[i] = g[i] * silu'(x[i]).
    /// g и x должны быть F32, одинаковой формы.
    /// dst — форма x, F32.
    pub fn silu_back(&mut self, g: TensorId, x: TensorId) -> TensorId {
        let tg = self.t(g);
        let tx = self.t(x);
        assert_eq!(tg.dtype, DType::F32, "silu_back: g должен быть F32");
        assert_eq!(tx.dtype, DType::F32, "silu_back: x должен быть F32");
        assert!(tg.same_shape(tx), "silu_back: несовпадение форм g и x");
        let dst = self.new_tensor(DType::F32, tx.ne);
        let d = self.t_mut(dst);
        d.op = Op::SiluBack;
        d.src = [Some(g), Some(x), None, None];
        dst
    }

    /// Обратное распространение Gelu: dst[i] = g[i] * gelu'(x[i]).
    /// g и x должны быть F32, одинаковой формы.
    /// dst — форма x, F32.
    pub fn gelu_back(&mut self, g: TensorId, x: TensorId) -> TensorId {
        let tg = self.t(g);
        let tx = self.t(x);
        assert_eq!(tg.dtype, DType::F32, "gelu_back: g должен быть F32");
        assert_eq!(tx.dtype, DType::F32, "gelu_back: x должен быть F32");
        assert!(tg.same_shape(tx), "gelu_back: несовпадение форм g и x");
        let dst = self.new_tensor(DType::F32, tx.ne);
        let d = self.t_mut(dst);
        d.op = Op::GeluBack;
        d.src = [Some(g), Some(x), None, None];
        dst
    }

    pub fn rope(&mut self, a: TensorId, pos: TensorId, n_dims: usize, base: f32) -> TensorId {
        assert_eq!(self.t(pos).dtype, DType::I32);
        assert!(n_dims.is_multiple_of(2) && n_dims <= self.t(a).ne[0]);
        let dst = self.unary_op(Op::Rope, a);
        let d = self.t_mut(dst);
        d.src[1] = Some(pos);
        d.op_params[0] = n_dims as u32;
        d.op_params[1] = base.to_bits();
        // op_params[2] = 0: forward
        dst
    }

    /// Обратное распространение Rope: транспонированное вращение.
    /// Как rope, но op_params[2]=1 (backward).
    pub fn rope_back(&mut self, g: TensorId, pos: TensorId, n_dims: usize, base: f32) -> TensorId {
        assert_eq!(self.t(pos).dtype, DType::I32);
        assert!(n_dims.is_multiple_of(2) && n_dims <= self.t(g).ne[0]);
        let dst = self.unary_op(Op::Rope, g);
        let d = self.t_mut(dst);
        d.src[1] = Some(pos);
        d.op_params[0] = n_dims as u32;
        d.op_params[1] = base.to_bits();
        d.op_params[2] = 1; // backward
        dst
    }

    /// Обратное распространение SoftMax.
    /// y = ВЫХОД softmax forward; dst формы y; src[0]=g, src[1]=y.
    pub fn soft_max_back(&mut self, g: TensorId, y: TensorId) -> TensorId {
        let tg = self.t(g);
        let ty = self.t(y);
        assert_eq!(tg.dtype, DType::F32, "soft_max_back: g должен быть F32");
        assert_eq!(ty.dtype, DType::F32, "soft_max_back: y должен быть F32");
        assert!(tg.same_shape(ty), "soft_max_back: несовпадение форм g и y");
        let dst = self.new_tensor(DType::F32, ty.ne);
        let d = self.t_mut(dst);
        d.op = Op::SoftMaxBack;
        d.src = [Some(g), Some(y), None, None];
        dst
    }

    /// Обратное распространение RmsNorm.
    /// x = ВХОД rms_norm; op_params[0]=eps.to_bits(); src[0]=g, src[1]=x.
    pub fn rms_norm_back(&mut self, g: TensorId, x: TensorId, eps: f32) -> TensorId {
        let tg = self.t(g);
        let tx = self.t(x);
        assert_eq!(tg.dtype, DType::F32, "rms_norm_back: g должен быть F32");
        assert_eq!(tx.dtype, DType::F32, "rms_norm_back: x должен быть F32");
        assert!(tg.same_shape(tx), "rms_norm_back: несовпадение форм g и x");
        let dst = self.new_tensor(DType::F32, tx.ne);
        let d = self.t_mut(dst);
        d.op = Op::RmsNormBack;
        d.src = [Some(g), Some(x), None, None];
        d.op_params[0] = eps.to_bits();
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

    /// Обратное распространение CrossEntropyLoss.
    /// g — градиент лосса (форма [1], F32).
    /// logits, targets — F32, одинаковой формы.
    /// dst — форма logits, F32: dst = (softmax(logits) - targets) * g0 / nrows.
    pub fn cross_entropy_loss_back(&mut self, g: TensorId, logits: TensorId, targets: TensorId) -> TensorId {
        let tg = self.t(g);
        let tlogits = self.t(logits);
        let ttargets = self.t(targets);
        assert_eq!(tg.dtype, DType::F32, "cross_entropy_loss_back: g должен быть F32");
        assert_eq!(tlogits.dtype, DType::F32, "cross_entropy_loss_back: logits должен быть F32");
        assert_eq!(ttargets.dtype, DType::F32, "cross_entropy_loss_back: targets должен быть F32");
        assert!(tlogits.same_shape(ttargets), "cross_entropy_loss_back: несовпадение форм logits и targets");
        assert_eq!(tg.nelements(), 1, "cross_entropy_loss_back: g должен быть скаляром [1]");
        let dst = self.new_tensor(DType::F32, tlogits.ne);
        let d = self.t_mut(dst);
        d.op = Op::CrossEntropyLossBack;
        d.src = [Some(g), Some(logits), Some(targets), None];
        dst
    }

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

    /// Обратное распространение GetRows: аккумуляция градиентов в embedding-таблицу.
    /// g — градиент выхода get_rows (форма [E, T], F32).
    /// ids — индексы строк (форма [T], I32).
    /// table — исходная таблица эмбеддингов (форма [E, V], нужна только для формы; F32).
    /// dst — градиент таблицы (форма [E, V], F32).
    pub fn get_rows_back(&mut self, g: TensorId, ids: TensorId, table: TensorId) -> TensorId {
        let tg = self.t(g);
        let tids = self.t(ids);
        let ttable = self.t(table);
        assert_eq!(tg.dtype, DType::F32, "get_rows_back: g должен быть F32");
        assert_eq!(tids.dtype, DType::I32, "get_rows_back: ids должен быть I32");
        assert_eq!(ttable.dtype, DType::F32,
            "get_rows backward: F16-таблица — обучаемые embeddings только F32");
        assert_eq!(tg.ne[0], ttable.ne[0],
            "get_rows_back: несовпадение embedding-размера g.ne[0] и table.ne[0]");
        assert_eq!(tg.ne[1], tids.ne[0],
            "get_rows_back: несовпадение T: g.ne[1] != ids.ne[0]");
        let dst = self.new_tensor(DType::F32, ttable.ne);
        let d = self.t_mut(dst);
        d.op = Op::GetRowsBack;
        d.src = [Some(g), Some(ids), Some(table), None];
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

    /// cont_f32: сделать непрерывную f32-копию тензора (работает и для F16-источников).
    /// dst всегда DType::F32.
    pub fn cont_f32(&mut self, a: TensorId) -> TensorId {
        let ne = self.t(a).ne;
        let dst = self.new_tensor(DType::F32, ne);
        let d = self.t_mut(dst);
        d.op = Op::Cont;
        d.src = [Some(a), None, None, None];
        dst
    }

    pub fn soft_max(&mut self, a: TensorId) -> TensorId {
        self.unary_op(Op::SoftMax, a)
    }

    pub fn rms_norm(&mut self, a: TensorId, eps: f32) -> TensorId {
        let dst = self.unary_op(Op::RmsNorm, a);
        self.t_mut(dst).op_params[0] = eps.to_bits();
        dst
    }

    pub fn mul_mat(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let ta = self.t(a);
        let tb = self.t(b);
        assert_eq!(ta.ne[0], tb.ne[0], "mul_mat: несовпадение k");
        assert!(
            tb.ne[2].is_multiple_of(ta.ne[2]) && tb.ne[3].is_multiple_of(ta.ne[3]),
            "mul_mat: broadcast a по dims 2,3"
        );
        let ne = [ta.ne[1], tb.ne[1], tb.ne[2], tb.ne[3]];
        let dst = self.new_tensor(DType::F32, ne);
        let d = self.t_mut(dst);
        d.op = Op::MulMat;
        d.src = [Some(a), Some(b), None, None];
        dst
    }

    /// Outer product с батчем: dst[ix, iy, b] = Σ_{r} x[ix, r, b] · y[iy, r, b].
    /// x: [Dx, R, B], y: [Dy, R, B] — 2D (B=1) или 3D F32; dst: [Dx, Dy, B].
    pub fn out_prod(&mut self, x: TensorId, y: TensorId) -> TensorId {
        let tx = self.t(x);
        let ty = self.t(y);
        assert_eq!(tx.dtype, DType::F32, "out_prod: x должен быть F32");
        assert_eq!(ty.dtype, DType::F32, "out_prod: y должен быть F32");
        assert_eq!(tx.ne[1], ty.ne[1], "out_prod: несовпадение R");
        assert_eq!(tx.ne[2], ty.ne[2], "out_prod: несовпадение батча");
        assert_eq!(tx.ne[3], 1, "out_prod: 4D не поддержан");
        assert_eq!(ty.ne[3], 1, "out_prod: 4D не поддержан");
        let ne = [tx.ne[0], ty.ne[0], tx.ne[2], 1];
        let dst = self.new_tensor(DType::F32, ne);
        let d = self.t_mut(dst);
        d.op = Op::OutProd;
        d.src = [Some(x), Some(y), None, None];
        dst
    }

    /// Пометить тензор как обучаемый параметр.
    pub fn set_param(&mut self, id: TensorId) {
        self.t_mut(id).is_param = true;
    }

    /// Собрать до 4 тензоров в один (Op::Collect). Если > 4 — строит цепочку collect'ов.
    /// dst — F32 [1,1,1,1], данные не используются.
    pub fn collect(&mut self, srcs: &[TensorId]) -> TensorId {
        assert!(!srcs.is_empty(), "collect: пустой список");
        if srcs.len() <= MAX_SRC {
            let dst = self.new_tensor_1d(DType::F32, 1);
            let d = self.t_mut(dst);
            d.op = Op::Collect;
            for (i, &s) in srcs.iter().enumerate() {
                d.src[i] = Some(s);
            }
            dst
        } else {
            // Цепочка: первые MAX_SRC в первый collect, остальные рекурсивно.
            let mut remaining: Vec<TensorId> = srcs.to_vec();
            loop {
                if remaining.len() <= MAX_SRC {
                    return self.collect(&remaining);
                }
                let batch: Vec<TensorId> = remaining.drain(..MAX_SRC).collect();
                let c = self.collect(&batch);
                remaining.push(c);
            }
        }
    }

    /// Сумма всех элементов тензора a в скаляр [1] (Op::SumAll).
    pub fn sum_all(&mut self, a: TensorId) -> TensorId {
        let dst = self.new_tensor_1d(DType::F32, 1);
        let d = self.t_mut(dst);
        d.op = Op::SumAll;
        d.src = [Some(a), None, None, None];
        dst
    }

    /// Обратное распространение SumAll: dst формы like.ne, заполняется g[0].
    /// g — тензор [1] со значением градиента.
    pub fn sum_all_back(&mut self, g: TensorId, like: TensorId) -> TensorId {
        assert_eq!(self.t(g).dtype, DType::F32, "sum_all_back: gradient must be F32");
        assert_eq!(self.t(g).nelements(), 1, "sum_all_back: gradient must be scalar");
        let ne = self.t(like).ne;
        let dst = self.new_tensor(DType::F32, ne);
        let d = self.t_mut(dst);
        d.op = Op::SumAllBack;
        d.src = [Some(g), None, None, None];
        dst
    }

    /// Каузальная маска: по каждой строке (i1, i2) элементы i0 > i1 → -inf.
    /// a — F32 [tk, tq, h, 1]; применяется к матрице attention перед soft_max.
    pub fn diag_mask_inf(&mut self, a: TensorId) -> TensorId {
        assert_eq!(self.t(a).dtype, DType::F32, "diag_mask_inf: только F32");
        self.unary_op(Op::DiagMaskInf, a)
        // op_params[0] остаётся 0 → режим -inf
    }

    /// Grad-режим той же маски: маскированные позиции → 0.0. Только для build_backward.
    pub(crate) fn diag_mask_zero(&mut self, a: TensorId) -> TensorId {
        assert_eq!(self.t(a).dtype, DType::F32, "diag_mask_zero: только F32");
        let dst = self.unary_op(Op::DiagMaskInf, a);
        self.t_mut(dst).op_params[0] = 1;
        dst
    }

    fn binary_op(&mut self, op: Op, a: TensorId, b: TensorId) -> TensorId {
        let ta = self.t(a);
        let tb = self.t(b);
        assert_eq!(ta.dtype, DType::F32, "binary_op: only F32");
        assert_eq!(tb.dtype, DType::F32, "binary_op: only F32");
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

    fn unary_op(&mut self, op: Op, a: TensorId) -> TensorId {
        assert_eq!(self.t(a).dtype, DType::F32, "unary_op: only F32");
        let ne = self.t(a).ne;
        let dtype = self.t(a).dtype;
        let dst = self.new_tensor(dtype, ne);
        let d = self.t_mut(dst);
        d.op = op;
        d.src = [Some(a), None, None, None];
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{DType, f16_to_f32};

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

    #[test]
    fn f16_tensor_strides() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_2d(DType::F16, 4, 2);
        let t = ctx.t(a);
        assert_eq!(t.nb, [2, 8, 16, 16]);
        assert_eq!(t.nbytes(), 16);
        assert!(t.is_contiguous());
    }

    #[test]
    fn f16_set_and_read() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_2d(DType::F16, 4, 2);
        ctx.set_f16(a, &[1.0, -2.0, 0.5, 65504.0, 0.0, 1.5, -0.25, 3.0]);
        let data = ctx.data_u16(a);
        assert_eq!(data.len(), 8);
        assert!((f16_to_f32(data[0]) - 1.0).abs() < 1e-6);
        assert!((f16_to_f32(data[3]) - 65504.0).abs() < 1e-6);
    }

    #[test]
    fn i32_strides_unchanged() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_2d(DType::I32, 3, 2);
        let t = ctx.t(a);
        assert_eq!(t.nb, [4, 12, 24, 24]);
    }

    #[test]
    fn permute_saves_op_params() {
        let mut ctx = Context::new(1 << 20);
        let a = ctx.new_tensor_2d(DType::F32, 4, 3);
        let p = ctx.permute(a, [1, 0, 2, 3]);
        let params = ctx.t(p).op_params;
        assert_eq!(params[0], 1);
        assert_eq!(params[1], 0);
        assert_eq!(params[2], 2);
        assert_eq!(params[3], 3);
    }
}
