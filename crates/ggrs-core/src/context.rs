use std::cell::UnsafeCell;

use crate::dtype::DType;
use crate::op::Op;
use crate::tensor::{Tensor, TensorId, MAX_DIMS, MAX_SRC};

struct Arena {
    buf: UnsafeCell<Box<[u8]>>,
}
// Ядра пишут в арену через сырые указатели из нескольких потоков,
// каждый поток — в свои строки. Дисциплина не-алиасинга — на ядрах (как в ggml).
unsafe impl Sync for Arena {}

pub struct Context {
    tensors: Vec<Tensor>,
    arena: Arena,
    arena_used: usize,
}

impl Context {
    pub fn new(mem_size: usize) -> Context {
        Context {
            tensors: Vec::new(),
            arena: Arena { buf: UnsafeCell::new(vec![0u8; mem_size].into_boxed_slice()) },
            arena_used: 0,
        }
    }

    fn alloc(&mut self, nbytes: usize) -> usize {
        let offset = (self.arena_used + 31) & !31; // 32-байтное выравнивание
        let len = unsafe { (*(*self.arena.buf.get())).len() };
        assert!(offset + nbytes <= len, "ggrs: arena out of memory");
        self.arena_used = offset + nbytes;
        offset
    }

    pub fn new_tensor(&mut self, dtype: DType, ne: [usize; MAX_DIMS]) -> TensorId {
        let ts = dtype.size();
        let nb = [ts, ts * ne[0], ts * ne[0] * ne[1], ts * ne[0] * ne[1] * ne[2]];
        let nbytes = ts * ne.iter().product::<usize>();
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

    pub(crate) fn base(&self) -> *mut u8 {
        unsafe { (*self.arena.buf.get()).as_mut_ptr() }
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
    pub fn set_f32(&mut self, id: TensorId, vals: &[f32]) {
        self.data_f32_mut(id).copy_from_slice(vals);
    }
    pub fn set_i32(&mut self, id: TensorId, vals: &[i32]) {
        self.data_i32_mut(id).copy_from_slice(vals);
    }
    /// Строковое чтение через страйды — работает и для views/permute.
    pub fn get_f32(&self, id: TensorId, idx: [usize; MAX_DIMS]) -> f32 {
        let t = self.t(id);
        assert_eq!(t.dtype, DType::F32);
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
        assert_eq!(t.nelements(), ne.iter().product::<usize>(), "reshape: число элементов не совпадает");
        let ts = t.dtype.size();
        let nb = [ts, ts * ne[0], ts * ne[0] * ne[1], ts * ne[0] * ne[1] * ne[2]];
        self.new_view(a, ne, nb, Op::Reshape)
    }

    /// axes[i] — новая позиция измерения i (семантика ggml_permute).
    pub fn permute(&mut self, a: TensorId, axes: [usize; MAX_DIMS]) -> TensorId {
        let t = self.t(a);
        let mut ne = [0usize; MAX_DIMS];
        let mut nb = [0usize; MAX_DIMS];
        for i in 0..MAX_DIMS {
            ne[axes[i]] = t.ne[i];
            nb[axes[i]] = t.nb[i];
        }
        self.new_view(a, ne, nb, Op::Permute)
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

    fn binary_op(&mut self, op: Op, a: TensorId, b: TensorId) -> TensorId {
        let ta = self.t(a);
        let tb = self.t(b);
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
    use crate::dtype::DType;

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
}
