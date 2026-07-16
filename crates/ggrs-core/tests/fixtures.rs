use std::collections::HashMap;

use ggrs_core::*;

struct Fx {
    dtype: u32,
    ne: [usize; 4],
    f32s: Vec<f32>,
    i32s: Vec<i32>,
}

fn load() -> HashMap<String, Fx> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ops.bin")).unwrap();
    let mut p = 0usize;
    let rd_u32 = |b: &[u8], p: &mut usize| {
        let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let n = rd_u32(&bytes, &mut p);
    let mut m = HashMap::new();
    for _ in 0..n {
        let nl = rd_u32(&bytes, &mut p) as usize;
        let name = String::from_utf8(bytes[p..p + nl].to_vec()).unwrap();
        p += nl;
        let dtype = rd_u32(&bytes, &mut p);
        let ne = [
            rd_u32(&bytes, &mut p) as usize,
            rd_u32(&bytes, &mut p) as usize,
            rd_u32(&bytes, &mut p) as usize,
            rd_u32(&bytes, &mut p) as usize,
        ];
        let count: usize = ne.iter().product();
        let mut f32s = Vec::new();
        let mut i32s = Vec::new();
        for i in 0..count {
            let raw = &bytes[p + i * 4..p + i * 4 + 4];
            if dtype == 0 {
                f32s.push(f32::from_le_bytes(raw.try_into().unwrap()));
            } else {
                i32s.push(i32::from_le_bytes(raw.try_into().unwrap()));
            }
        }
        p += count * 4;
        m.insert(name, Fx { dtype, ne, f32s, i32s });
    }
    m
}

fn assert_close(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: длина");
    for i in 0..got.len() {
        let (g, w) = (got[i], want[i]);
        assert!(
            (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
            "{label}[{i}]: {g} vs {w}"
        );
    }
}

#[test]
fn ops_match_numpy() {
    let fx = load();
    let tensor = |ctx: &mut Context, f: &Fx| -> TensorId {
        let id = ctx.new_tensor(if f.dtype == 0 { DType::F32 } else { DType::I32 }, f.ne);
        if f.dtype == 0 {
            ctx.set_f32(id, &f.f32s);
        } else {
            ctx.set_i32(id, &f.i32s);
        }
        id
    };

    // mulmat
    {
        let mut ctx = Context::new(1 << 22);
        let a = tensor(&mut ctx, &fx["mulmat.a"]);
        let b = tensor(&mut ctx, &fx["mulmat.b"]);
        let d = ctx.mul_mat(a, b);
        let g = build_forward(&ctx, d);
        compute(&mut ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["mulmat.out"].f32s, "mulmat");
    }
    // softmax
    {
        let mut ctx = Context::new(1 << 22);
        let x = tensor(&mut ctx, &fx["softmax.x"]);
        let d = ctx.soft_max(x);
        let g = build_forward(&ctx, d);
        compute(&mut ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["softmax.out"].f32s, "softmax");
    }
    // rms_norm
    {
        let mut ctx = Context::new(1 << 22);
        let x = tensor(&mut ctx, &fx["rmsnorm.x"]);
        let d = ctx.rms_norm(x, 1e-5);
        let g = build_forward(&ctx, d);
        compute(&mut ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["rmsnorm.out"].f32s, "rmsnorm");
    }
    // rope
    {
        let mut ctx = Context::new(1 << 22);
        let x = tensor(&mut ctx, &fx["rope.x"]);
        let pos = tensor(&mut ctx, &fx["rope.pos"]);
        let d = ctx.rope(x, pos, 8, 10000.0);
        let g = build_forward(&ctx, d);
        compute(&mut ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["rope.out"].f32s, "rope");
    }
    // cross_entropy
    {
        let mut ctx = Context::new(1 << 22);
        let lg = tensor(&mut ctx, &fx["xent.logits"]);
        let tg = tensor(&mut ctx, &fx["xent.targets"]);
        let d = ctx.cross_entropy_loss(lg, tg);
        let g = build_forward(&ctx, d);
        compute(&mut ctx, &g, 2);
        assert_close(ctx.data_f32(d), &fx["xent.out"].f32s, "xent");
    }
}
