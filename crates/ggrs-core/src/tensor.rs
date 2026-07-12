use crate::dtype::DType;
use crate::op::Op;

pub const MAX_DIMS: usize = 4;
pub const MAX_SRC: usize = 4;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TensorId(pub usize);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub ne: [usize; MAX_DIMS], // число элементов по измерениям, ne[0] — самое быстрое
    pub nb: [usize; MAX_DIMS], // страйды в байтах
    pub op: Op,
    pub src: [Option<TensorId>; MAX_SRC],
    pub offset: usize, // байтовое смещение в арене
    pub op_params: [u32; 8],
    pub is_param: bool,
}

impl Tensor {
    pub fn nelements(&self) -> usize {
        self.ne.iter().product()
    }
    pub fn nrows(&self) -> usize {
        self.ne[1] * self.ne[2] * self.ne[3]
    }
    pub fn is_contiguous(&self) -> bool {
        let mut expected = self.dtype.size();
        for i in 0..MAX_DIMS {
            if self.ne[i] != 1 && self.nb[i] != expected {
                return false;
            }
            expected *= self.ne[i];
        }
        true
    }
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.ne == other.ne
    }
}
