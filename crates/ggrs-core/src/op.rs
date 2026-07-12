#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    None,
    Add,
    Mul,
    Scale,
    Silu,
    Gelu,
    MulMat,
    SoftMax,
    RmsNorm,
    GetRows,
    Rope,
    Cont,
    Reshape,
    Permute,
    CrossEntropyLoss,
}
