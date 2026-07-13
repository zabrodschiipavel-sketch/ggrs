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
    /// Обратное распространение Silu: dst[i] = g[i] * silu'(x[i])
    SiluBack,
    /// Обратное распространение Gelu: dst[i] = g[i] * gelu'(x[i])
    GeluBack,
    /// Обратное распространение CrossEntropyLoss: dst = (softmax(logits) - targets) * g0 / nrows
    CrossEntropyLossBack,
    /// Обратное распространение GetRows: аккумуляция градиентов в embedding-таблицу.
    /// dst (форма table) — градиент таблицы; src[0]=g, src[1]=ids, src[2]=table (для формы).
    GetRowsBack,
    /// Обратное распространение SoftMax: dst[i] = y[i] * (g[i] − Σ_i g[i]*y[i]).
    /// src[0]=g (градиент выхода), src[1]=y (выход softmax forward).
    SoftMaxBack,
    /// Обратное распространение RmsNorm: dst = r*g − x * r³ * dot / ne0.
    /// src[0]=g, src[1]=x (вход rms_norm); op_params[0]=eps.to_bits().
    RmsNormBack,
}
