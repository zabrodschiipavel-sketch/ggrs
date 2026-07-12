use ggrs_core::{build_forward, compute, Context, DType};

fn new_tensor_3d(ctx: &mut Context, d0: usize, d1: usize, d2: usize) -> ggrs_core::TensorId {
    ctx.new_tensor_3d(DType::F32, d0, d1, d2)
}

#[test]
fn mulmat_3d_per_head() {
    let mut ctx = Context::new(1 << 24);
    
    // a: [k=4, m=3, h=2] -> ne=[4,3,2,1], layout: k fastest
    let a = new_tensor_3d(&mut ctx, 4, 3, 2);
    let a_data: Vec<f32> = (0..(4*3*2)).map(|i| (i % 7) as f32 * 0.5 - 1.5).collect();
    ctx.set_f32(a, &a_data);
    
    // b: [k=4, n=2, h=2] -> ne=[4,2,2,1], layout: k fastest
    let b = new_tensor_3d(&mut ctx, 4, 2, 2);
    let b_data: Vec<f32> = (0..(4*2*2)).map(|i| (i % 5) as f32 * 0.3 - 0.6).collect();
    ctx.set_f32(b, &b_data);
    
    let dst = ctx.mul_mat(a, b);
    assert_eq!(ctx.t(dst).ne, [3, 2, 2, 1]);
    let g = build_forward(&ctx, dst);
    compute(&ctx, &g, 2);
    
    // наивное вычисление: dst[i0,i1,i2,0] = sum_k a[k + i0*4 + i2*16] * b[k + i1*4 + i2*8]
    // layout: ne0 самое быстрое, т.е. k fastest, затем m/n, затем h
    // для a [4,3,2]: страйд головы = 4*3 = 12 → index = k + i0*4 + i2*12
    // для b [4,2,2]: страйд головы = 4*2 = 8  → index = k + i1*4 + i2*8
    
    let mut correct = true;
    for i2 in 0..2 { // h
        for i1 in 0..2 { // n
            for i0 in 0..3 { // m
                let mut expected = 0.0f32;
                for k in 0..4 {
                    let a_idx = k + i0 * 4 + i2 * 12;
                    let b_idx = k + i1 * 4 + i2 * 8;
                    expected += a_data[a_idx] * b_data[b_idx];
                }
                let got = ctx.get_f32(dst, [i0, i1, i2, 0]);
                if (got - expected).abs() > 1e-5 {
                    correct = false;
                    eprintln!("mulmat_3d_per_head: mismatch at [{},{},{},0]: got {}, expected {}", 
                              i0, i1, i2, got, expected);
                }
            }
        }
    }
    
    assert!(correct, "mulmat_3d_per_head: some elements mismatched");
}

#[test]
fn rope_multihead() {
    let mut ctx = Context::new(1 << 24);
    
    // a: head_dim=4, n_head=2, T=3 -> ne=[4,2,3,1]
    let a = new_tensor_3d(&mut ctx, 4, 2, 3);
    let ne0 = 4;
    let ne1 = 2;
    let ne2 = 3;
    let total = ne0 * ne1 * ne2;
    let a_data: Vec<f32> = (0..total).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
    ctx.set_f32(a, &a_data);
    
    let pos = ctx.new_tensor_1d(DType::I32, 3);
    ctx.set_i32(pos, &[0, 1, 2]);
    let r = ctx.rope(a, pos, 4, 10000.0f32);
    let g = build_forward(&ctx, r);
    compute(&ctx, &g, 2);

    // Проверка pos=0: обе головы неизменны
    let mut ok_pos0 = true;
    for i1 in 0..ne1 { // n_head
        for i0 in 0..ne0 { // head_dim
            let got = ctx.get_f32(r, [i0, i1, 0, 0]);
            let expected = a_data[i0 + i1 * ne0];
            if (got - expected).abs() > 1e-5 {
                ok_pos0 = false;
                eprintln!("rope_multihead: pos=0 mismatch at [{},{},0,0]: got {}, expected {}", 
                          i0, i1, got, expected);
            }
        }
    }
    assert!(ok_pos0, "rope_multihead: pos=0 should be unchanged");
    
    // Проверка pos=1: обе головы повёрнуты с ОДИНАКОВЫМИ углами
    // theta_i = 10000^(-2i/d) для i в [0, head_dim/2)
    // theta_0 = 10000^0 = 1.0, theta_1 = 10000^(-2/4) = 10000^(-0.5) = 0.01
    let theta_0: f32 = 1.0;
    let theta_1: f32 = 0.01;
    let cos_0 = theta_0.cos();
    let sin_0 = theta_0.sin();
    let cos_1 = theta_1.cos();
    let sin_1 = theta_1.sin();
    
    let mut ok_pos1 = true;
    for h in 0..ne1 { // n_head
        // i0=0 и i0=1 (первая пара)
        let x0 = a_data[h * ne0 + ne0 * ne1];
        let x1 = a_data[1 + h * ne0 + ne0 * ne1];
        let y0 = a_data[2 + h * ne0 + ne0 * ne1];
        let y1 = a_data[3 + h * ne0 + ne0 * ne1];
        
        let got_0 = ctx.get_f32(r, [0, h, 1, 0]);
        let got_1 = ctx.get_f32(r, [1, h, 1, 0]);
        let got_2 = ctx.get_f32(r, [2, h, 1, 0]);
        let got_3 = ctx.get_f32(r, [3, h, 1, 0]);
        
        let exp_0 = x0 * cos_0 - x1 * sin_0;
        let exp_1 = x0 * sin_0 + x1 * cos_0;
        let exp_2 = y0 * cos_1 - y1 * sin_1;
        let exp_3 = y0 * sin_1 + y1 * cos_1;
        
        if (got_0 - exp_0).abs() > 1e-5 || (got_1 - exp_1).abs() > 1e-5 ||
            (got_2 - exp_2).abs() > 1e-5 || (got_3 - exp_3).abs() > 1e-5 {
            ok_pos1 = false;
            eprintln!("rope_multihead: pos=1 mismatch at head {}: got [{},{},{},{}], exp [{},{},{},{}]", 
                      h, got_0, got_1, got_2, got_3, exp_0, exp_1, exp_2, exp_3);
        }
    }
    
    assert!(ok_pos1, "rope_multihead: pos=1 rotation should be head-independent");
}
