use crate::context::Context;
use crate::dtype::{DType, f16_to_f32};
use crate::tensor::TensorId;
use super::{row_ptr, split, unravel_row};

pub fn get_rows(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    let ids = ctx.data_i32(dst.src[1].unwrap());
    assert_eq!(dst.dtype, DType::F32, "get_rows: dst только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "get_rows: dst строки должны быть плотными");
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.ne[1], ith, nth);
    match a.dtype {
        DType::F32 => {
            assert_eq!(a.nb[0], 4, "get_rows: src строки должны быть плотными (F32)");
            for (ir, &id) in ids.iter().enumerate().take(ir1).skip(ir0) {
                let row = id as usize;
                assert!(row < a.ne[1], "get_rows: индекс {row} вне диапазона");
                unsafe {
                    let ps = row_ptr(ctx, a, row, 0, 0) as *const f32;
                    let pd = row_ptr(ctx, dst, ir, 0, 0) as *mut f32;
                    std::ptr::copy_nonoverlapping(ps, pd, ne0);
                }
            }
        }
        DType::F16 => {
            assert_eq!(a.nb[0], 2, "get_rows: src строки должны быть плотными (F16)");
            for (ir, &id) in ids.iter().enumerate().take(ir1).skip(ir0) {
                let row = id as usize;
                assert!(row < a.ne[1], "get_rows: индекс {row} вне диапазона");
                unsafe {
                    let ps = row_ptr(ctx, a, row, 0, 0) as *const u16;
                    let pd = row_ptr(ctx, dst, ir, 0, 0) as *mut f32;
                    for i in 0..ne0 {
                        *pd.add(i) = f16_to_f32(*ps.add(i));
                    }
                }
            }
        }
        _ => panic!("get_rows: неподдерживаемый dtype src {:?}", a.dtype),
    }
}

pub fn cont(ctx: &Context, dst_id: TensorId, ith: usize, nth: usize) {
    let dst = ctx.t(dst_id);
    let a = ctx.t(dst.src[0].unwrap());
    assert_eq!(
        dst.nb[0], dst.dtype.type_size(),
        "cont: dst строки должны быть плотными"
    );
    let ne0 = dst.ne[0];
    let (ir0, ir1) = split(dst.nrows(), ith, nth);
    match (a.dtype, dst.dtype) {
        (DType::F32, DType::F32) => {
            for ir in ir0..ir1 {
                let (i1, i2, i3) = unravel_row(dst, ir);
                unsafe {
                    let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
                    let base = ctx.base().add(a.offset + i1 * a.nb[1] + i2 * a.nb[2] + i3 * a.nb[3]);
                    for i0 in 0..ne0 {
                        *pd.add(i0) = *(base.add(i0 * a.nb[0]) as *const f32);
                    }
                }
            }
        }
        (DType::F16, DType::F16) => {
            for ir in ir0..ir1 {
                let (i1, i2, i3) = unravel_row(dst, ir);
                unsafe {
                    let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut u16;
                    let base = ctx.base().add(a.offset + i1 * a.nb[1] + i2 * a.nb[2] + i3 * a.nb[3]);
                    for i0 in 0..ne0 {
                        *pd.add(i0) = *(base.add(i0 * a.nb[0]) as *const u16);
                    }
                }
            }
        }
        (DType::F16, DType::F32) => {
            for ir in ir0..ir1 {
                let (i1, i2, i3) = unravel_row(dst, ir);
                unsafe {
                    let pd = row_ptr(ctx, dst, i1, i2, i3) as *mut f32;
                    let base = ctx.base().add(a.offset + i1 * a.nb[1] + i2 * a.nb[2] + i3 * a.nb[3]);
                    for i0 in 0..ne0 {
                        *pd.add(i0) = f16_to_f32(*(base.add(i0 * a.nb[0]) as *const u16));
                    }
                }
            }
        }
        _ => panic!(
            "cont: неподдерживаемая комбинация dtype src={:?} dst={:?}",
            a.dtype, dst.dtype
        ),
    }
}

/// Обратное распространение GetRows: аккумуляция градиентов в embedding-таблицу.
/// Только ith==0 (ids могут повторяться — параллельная запись недопустима).
/// dst (форма [E, V]) — градиент таблицы.
/// src[0]=g (форма [E, T]), src[1]=ids (форма [T]), src[2]=table (форма [E, V], только для asserts).
pub fn get_rows_back(ctx: &Context, dst_id: TensorId, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let dst = ctx.t(dst_id);
    let g = ctx.t(dst.src[0].unwrap());
    let ids = ctx.data_i32(dst.src[1].unwrap());

    assert_eq!(dst.dtype, DType::F32, "get_rows_back: dst только F32");
    assert_eq!(g.dtype, DType::F32, "get_rows_back: g только F32");
    assert_eq!(dst.nb[0], dst.dtype.type_size(), "get_rows_back: dst строки должны быть плотными");
    assert_eq!(g.nb[0], g.dtype.type_size(), "get_rows_back: g строки должны быть плотными");

    let e = dst.ne[0];
    let v = dst.ne[1];

    // Зануляем весь dst (градиент таблицы)
    unsafe {
        let dst_slice = std::slice::from_raw_parts_mut(
            ctx.base().add(dst.offset) as *mut f32,
            dst.nelements(),
        );
        dst_slice.fill(0.0);
    }

    // Аккумулируем: для каждого t: dst[row-строка][*] += g[t-строка][*]
    #[allow(clippy::needless_range_loop)]
    for ti in 0..g.ne[1] {
        let row = ids[ti] as usize;
        assert!(row < v, "get_rows_back: индекс строки {row} вне диапазона [0, {v})");
        unsafe {
            let pg = row_ptr(ctx, g, ti, 0, 0) as *const f32;
            let pd = row_ptr(ctx, dst, row, 0, 0) as *mut f32;
            for i in 0..e {
                *pd.add(i) += *pg.add(i);
            }
        }
    }
}
