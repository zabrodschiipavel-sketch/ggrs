use ggrs_core::*;

/// get_rows: src F16 -> dst F32 совпадает с src F32 -> dst F32
#[test]
fn get_rows_f16_matches_f32() {
    let mut ctx = Context::new(1 << 20);

    // Эталон: F32->F32
    let emb_f32 = ctx.new_tensor_2d(DType::F32, 4, 2);
    ctx.set_f32(emb_f32, &[1.0, -2.0, 0.5, 3.25, -0.75, 4.0, 0.125, -1.5]);

    // F16 версия
    let emb_f16 = ctx.new_tensor_2d(DType::F16, 4, 2);
    ctx.set_f16(emb_f16, &[1.0, -2.0, 0.5, 3.25, -0.75, 4.0, 0.125, -1.5]);

    let ids = ctx.new_tensor_1d(DType::I32, 2);
    ctx.set_i32(ids, &[1, 0]);

    // get_rows от F32
    let r_f32 = ctx.get_rows(emb_f32, ids);
    let g = build_forward(&ctx, r_f32);
    compute(&ctx, &g, 1);
    let expected = ctx.data_f32(r_f32).to_vec();

    // get_rows от F16
    let r_f16 = ctx.get_rows(emb_f16, ids);
    let g2 = build_forward(&ctx, r_f16);
    compute(&ctx, &g2, 1);
    let got = ctx.data_f32(r_f16);

    assert_eq!(got.len(), expected.len(), "get_rows: разные длины");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff < 1e-3,
            "get_rows_f16_matches_f32[{}]: got={}, expected={}, diff={}",
            i, g, e, diff
        );
    }
}

/// cont_f32: страйдовое чтение F16 view -> F32 dst
#[test]
fn cont_f16_to_f32() {
    let mut ctx = Context::new(1 << 20);

    // F16 [3,2] с известными значениями
    let f16_t = ctx.new_tensor_2d(DType::F16, 3, 2);
    ctx.set_f16(f16_t, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Транспонирование: [2,3], логически строки = [1,4], [2,5], [3,6]
    let t = ctx.transpose(f16_t);
    // cont_f32 сделает F32-копию из страйдового F16
    let c = ctx.cont_f32(t);
    let g = build_forward(&ctx, c);
    compute(&ctx, &g, 1);

    // Проверка: все 6 значений через data_f32
    let data = ctx.data_f32(c);
    // Ожидаемая перестановка от транспонирования [3,2]:
    // исходник: row0=[1,2,3], row1=[4,5,6]
    // после permute(1,0,2,3): ne=[2,3,1,1], nb=[8,4,12,12]
    // строки: (i1=0,i2=0) -> смещение 0: [1,4] (nb0=8)
    //         (i1=1,i2=0) -> см. 4:  [2,5]
    //         (i1=2,i2=0) -> см. 8:  [3,6]
    assert_eq!(data.len(), 6, "cont_f16_to_f32: ожидается 6 элементов");
    let expected: [f32; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-3,
            "cont_f16_to_f32[{}]: got={}, expected={}, diff={}",
            i, got, exp, diff
        );
    }
}

/// cont: F16 -> F16 roundtrip (data_u16 совпадает)
#[test]
fn cont_f16_roundtrip() {
    let mut ctx = Context::new(1 << 20);

    // F16 [4,1]
    let a = ctx.new_tensor_2d(DType::F16, 4, 1);
    let vals_f32 = [3.25, -0.75, 0.0, 1.5];
    ctx.set_f16(a, &vals_f32);

    let original_u16: Vec<u16> = ctx.data_u16(a).to_vec();

    // cont (F16 -> F16)
    let c = ctx.cont(a);
    let g = build_forward(&ctx, c);
    compute(&ctx, &g, 1);

    let after_u16 = ctx.data_u16(c);
    assert_eq!(
        after_u16.len(),
        original_u16.len(),
        "cont_f16_roundtrip: длины не совпадают"
    );
    for (i, (&got, &exp)) in after_u16.iter().zip(original_u16.iter()).enumerate() {
        assert_eq!(
            got, exp,
            "cont_f16_roundtrip[{}]: got={:#06x}, expected={:#06x}",
            i, got, exp
        );
    }
}

/// mul_mat паникует при F16-входе a
#[test]
#[should_panic(expected = "mul_mat: a только F32")]
fn mulmat_rejects_f16_a() {
    let mut ctx = Context::new(1 << 20);
    let a = ctx.new_tensor_2d(DType::F16, 4, 2);
    let b = ctx.new_tensor_2d(DType::F32, 4, 3);
    ctx.set_f16(a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    ctx.set_f32(b, &[1.0; 12]);
    let d = ctx.mul_mat(a, b);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);
}

/// Край subnormal-диапазона f32→f16: RNE на [2^-25, 2^-24) (аудит P2).
#[test]
fn f16_subnormal_edge_rne() {
    use ggrs_core::dtype::{f16_to_f32, f32_to_f16};
    // 3·2^-26 = 1.5·2^-25 ∈ (2^-25, 2^-24): ближайший — минимальный денормал 0x0001
    let v = 3.0f32 * (2.0f32).powi(-26);
    assert_eq!(f32_to_f16(v), 0x0001, "1.5·2^-25 должен округлиться к 0x0001");
    // ровно 2^-25 — середина: tie-to-even → 0
    assert_eq!(f32_to_f16((2.0f32).powi(-25)), 0x0000, "2^-25 — tie к чётному нулю");
    // чуть ниже 2^-25 → 0
    assert_eq!(f32_to_f16(0.99f32 * (2.0f32).powi(-25)), 0x0000);
    // чуть выше 2^-25 → 0x0001
    assert_eq!(f32_to_f16(1.01f32 * (2.0f32).powi(-25)), 0x0001);
    // 2^-24 — сам минимальный денормал
    assert_eq!(f32_to_f16((2.0f32).powi(-24)), 0x0001);
    assert_eq!(f16_to_f32(0x0001), (2.0f32).powi(-24));
    // знак сохраняется
    assert_eq!(f32_to_f16(-v), 0x8001);
}
