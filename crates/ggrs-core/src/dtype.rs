#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    I32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
        }
    }
}
