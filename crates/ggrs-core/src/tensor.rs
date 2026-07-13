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
    /// Число байт, занимаемых тензором в памяти (только для contiguous).
    pub fn nbytes(&self) -> usize {
        self.dtype.row_size(self.ne[0]) * self.ne[1] * self.ne[2] * self.ne[3]
    }
    pub fn is_contiguous(&self) -> bool {
        let ts = self.dtype.type_size();
        let rs = self.dtype.row_size(self.ne[0]);
        // nb[0] == type_size (если ne[0] != 1)
        if self.ne[0] != 1 && self.nb[0] != ts {
            return false;
        }
        // nb[1] == row_size(ne[0]) (если ne[1] != 1)
        if self.ne[1] != 1 && self.nb[1] != rs {
            return false;
        }
        // nb[2] == nb[1] * ne[1] (если ne[2] != 1)
        if self.ne[2] != 1 && self.nb[2] != rs * self.ne[1] {
            return false;
        }
        // nb[3] == nb[2] * ne[2] (если ne[3] != 1)
        if self.ne[3] != 1 && self.nb[3] != rs * self.ne[1] * self.ne[2] {
            return false;
        }
        true
    }
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.ne == other.ne
    }
}
