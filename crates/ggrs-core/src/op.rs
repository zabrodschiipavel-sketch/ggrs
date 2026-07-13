#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
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
    /// Сборка нескольких тензоров в один (no-op, данные не используются).
    Collect,
    /// Сумма всех элементов тензора в скаляр [1].
    SumAll,
    /// Обратное распространение SumAll: заполняет тензор формы like значением g[0].
    SumAllBack,
    /// Outer product: dst[ix, iy] = Σ_r x[ix, r] * y[iy, r]
    OutProd,
}
