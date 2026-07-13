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
#[should_panic(expected = "out_prod: только 2D")]
fn outprod_rejects_3d() {
    let mut ctx = Context::new(1 << 20);
    let x = ctx.new_tensor_3d(DType::F32, 4, 3, 2); // ne[2]=2
    let y = ctx.new_tensor_2d(DType::F32, 5, 3);
    let _ = ctx.out_prod(x, y);
}
