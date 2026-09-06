use ggrs_core::*;

fn build_case(ctx: &mut Context) -> TensorId {
    let k = 96usize;
    let m = 33;
    let n = 47;
    let a = ctx.new_tensor_2d(DType::F32, k, m);
    let b = ctx.new_tensor_2d(DType::F32, k, n);
    let av: Vec<f32> = (0..k * m).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| (i % 23) as f32 * 0.07 - 0.7).collect();
    ctx.set_f32(a, &av);
    ctx.set_f32(b, &bv);
    let d = ctx.mul_mat(a, b);
    let s = ctx.soft_max(d);
    ctx.rms_norm(s, 1e-5)
}

#[test]
fn threads_produce_identical_results() {
    let mut c1 = Context::new(1 << 24);
    let r1 = build_case(&mut c1);
    let g1 = build_forward(&c1, r1);
    compute(&mut c1, &g1, 1);

    let mut c4 = Context::new(1 << 24);
    let r4 = build_case(&mut c4);
    let g4 = build_forward(&c4, r4);
    compute(&mut c4, &g4, 4);

    // деление по строкам не меняет порядок редукций → бит-в-бит
    assert_eq!(c1.data_f32(r1), c4.data_f32(r4));
}

#[test]
fn worker_panic_returns_without_executing_dependents() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc;
    use std::time::Duration;

    // A timeout makes a broken barrier fail this test instead of hanging the suite.
    let (done, completion) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        for n_threads in [2, 4, 8] {
            for bad_index in 0..4 {
                let mut ctx = Context::new(4096);
                let table = ctx.new_tensor_2d(DType::F32, 2, 2);
                ctx.set_f32(table, &[1.0, 2.0, 3.0, 4.0]);
                let ids = ctx.new_tensor_1d(DType::I32, 4);
                let mut indices = [0, 1, 0, 1];
                indices[bad_index] = -1;
                ctx.set_i32(ids, &indices);
                let rows = ctx.get_rows(table, ids);
                let output = ctx.scale(rows, 2.0);
                ctx.set_f32(output, &[-99.0; 8]);
                let graph = build_forward(&ctx, output);

                let panic = catch_unwind(AssertUnwindSafe(|| {
                    compute(&mut ctx, &graph, n_threads);
                }))
                .expect_err("an invalid embedding index must panic");
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .expect("kernel panic should contain a message");
                assert!(message.contains("get_rows:"), "unexpected panic: {message}");
                assert_eq!(ctx.data_f32(output), &[-99.0; 8]);

                // All workers must have stopped before compute returns its panic.
                ctx.set_i32(ids, &[0, 1, 0, 1]);
                compute(&mut ctx, &graph, n_threads);
                assert_eq!(ctx.data_f32(output), &[2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0]);
            }
        }
        done.send(()).unwrap();
    });
    completion
        .recv_timeout(Duration::from_secs(10))
        .expect("compute panicked unexpectedly or deadlocked after a worker panic");
    worker.join().unwrap();
}
