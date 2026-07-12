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
}
