use ggrs_core::*;

/// Простой LCG для детерминированных случайных данных.
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as u32) as f32 / 4294967296.0 * 2.0 - 1.0
    }
}

fn fill_lcg(ctx: &mut Context, id: TensorId, seed: u64) {
    let n = ctx.t(id).nelements();
    let mut rng = Lcg::new(seed);
    let vals: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    ctx.set_f32(id, &vals);
}

#[test]
fn outprod_matches_naive() {
    let mut ctx = Context::new(1 << 24);
    let dx = 5usize;
    let dy = 4usize;
    let r = 7usize;

    let x = ctx.new_tensor_2d(DType::F32, dx, r);
    let y = ctx.new_tensor_2d(DType::F32, dy, r);
    fill_lcg(&mut ctx, x, 21);
    fill_lcg(&mut ctx, y, 22);

    let xv: Vec<f32> = ctx.data_f32(x).to_vec();
    let yv: Vec<f32> = ctx.data_f32(y).to_vec();

    let d = ctx.out_prod(x, y);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 2);

    let out = ctx.data_f32(d);
    // out — [Dx, Dy] в row-major (строка iy)
    for iy in 0..dy {
        for ix in 0..dx {
            let mut acc = 0.0f64;
            for rr in 0..r {
                // x[ix, rr] = xv[rr * dx + ix]
                // y[iy, rr] = yv[rr * dy + iy]
                acc += xv[rr * dx + ix] as f64 * yv[rr * dy + iy] as f64;
            }
            let got = out[iy * dx + ix];
            assert!(
                (got as f64 - acc).abs() < 1e-5,
                "outprod_matches_naive: ({ix},{iy}) got {got}, expected {acc}",
            );
        }
    }
}

#[test]
fn outprod_symmetric() {
    let mut ctx = Context::new(1 << 20);
    let n = 6usize;
    let r = 3usize;

    let x = ctx.new_tensor_2d(DType::F32, n, r);
    fill_lcg(&mut ctx, x, 42);

    let d = ctx.out_prod(x, x);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);

    let out = ctx.data_f32(d);
    // out — [n, n] в row-major; dst[i,j] == dst[j,i]
    for i in 0..n {
        for j in 0..n {
            let val_ij = out[j * n + i];
            let val_ji = out[i * n + j];
            assert!(
                (val_ij - val_ji).abs() < 1e-6,
                "outprod_symmetric: ({i},{j}) {val_ij} vs ({j},{i}) {val_ji}",
            );
        }
    }
}

#[test]
fn out_prod_3d_vs_naive() {
    let mut ctx = Context::new(1 << 24);
    let dx = 3usize;
    let dy = 5usize;
    let r = 4usize;
    let b = 2usize;

    // x: [3, 4, 2], y: [5, 4, 2]
    let x = ctx.new_tensor_3d(DType::F32, dx, r, b);
    let y = ctx.new_tensor_3d(DType::F32, dy, r, b);
    fill_lcg(&mut ctx, x, 51);
    fill_lcg(&mut ctx, y, 52);

    // Сохраняем копии входов через get_f32 (читает по страйдам)
    let xv: Vec<f32> = (0..(dx * r * b))
        .map(|i| {
            let ib = i / (dx * r);
            let ir = (i / dx) % r;
            let ix = i % dx;
            ctx.get_f32(x, [ix, ir, ib, 0])
        })
        .collect();
    let yv: Vec<f32> = (0..(dy * r * b))
        .map(|i| {
            let ib = i / (dy * r);
            let ir = (i / dy) % r;
            let iy = i % dy;
            ctx.get_f32(y, [iy, ir, ib, 0])
        })
        .collect();

    let d = ctx.out_prod(x, y);
    assert_eq!(ctx.t(d).ne, [dx, dy, b, 1]);
    let g = build_forward(&ctx, d);
    compute(&ctx, &g, 1);

    // dst: [dx, dy, b] — row-major: самый быстрый ix, затем iy, затем батч
    for ib in 0..b {
        for iy in 0..dy {
            for ix in 0..dx {
                let mut acc = 0.0f64;
                for rr in 0..r {
                    let x_idx = ib * dx * r + rr * dx + ix;
                    let y_idx = ib * dy * r + rr * dy + iy;
                    acc += xv[x_idx] as f64 * yv[y_idx] as f64;
                }
                let got = ctx.get_f32(d, [ix, iy, ib, 0]);
                assert!(
                    (got as f64 - acc).abs() < 1e-4,
                    "out_prod_3d_vs_naive: ({ix},{iy},{ib}) got {got}, expected {acc}",
                );
            }
        }
    }
}

#[test]
fn out_prod_3d_threads_parity() {
    let dx = 3usize;
    let dy = 5usize;
    let r = 4usize;
    let b = 2usize;

    // Два контекста с одинаковыми данными
    let mut ctx1 = Context::new(1 << 24);
    let x1 = ctx1.new_tensor_3d(DType::F32, dx, r, b);
    let y1 = ctx1.new_tensor_3d(DType::F32, dy, r, b);
    fill_lcg(&mut ctx1, x1, 61);
    fill_lcg(&mut ctx1, y1, 62);

    let mut ctx2 = Context::new(1 << 24);
    let x2 = ctx2.new_tensor_3d(DType::F32, dx, r, b);
    let y2 = ctx2.new_tensor_3d(DType::F32, dy, r, b);
    ctx2.set_f32(x2, ctx1.data_f32(x1));
    ctx2.set_f32(y2, ctx1.data_f32(y1));

    let d1 = ctx1.out_prod(x1, y1);
    let d2 = ctx2.out_prod(x2, y2);

    let g1 = build_forward(&ctx1, d1);
    let g2 = build_forward(&ctx2, d2);

    compute(&ctx1, &g1, 1);
    compute(&ctx2, &g2, 4);

    let out1 = ctx1.data_f32(d1);
    let out2 = ctx2.data_f32(d2);
    assert_eq!(out1.len(), out2.len());
    for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(a, b, "out_prod_3d_threads_parity: mismatch at element {i}: {a} vs {b}");
    }
}
