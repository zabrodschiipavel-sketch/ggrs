"""Генерирует эталонные фикстуры для ggrs-core из numpy. Запуск из корня репо."""
import struct
import numpy as np

rng = np.random.default_rng(42)
out = {}

# mulmat: a[m=5,k=7] строки, b[n=3,k=7]; ggml: dst[n,m], dst[i1,i0] = dot(a[i0], b[i1])
a = rng.standard_normal((5, 7)).astype(np.float32)
b = rng.standard_normal((3, 7)).astype(np.float32)
out["mulmat.a"] = a          # ne = [7, 5]
out["mulmat.b"] = b          # ne = [7, 3]
out["mulmat.out"] = (b @ a.T).astype(np.float32)  # shape [3, 5] → ne [5, 3]

# softmax по строкам
x = rng.standard_normal((4, 6)).astype(np.float32) * 3
e = np.exp(x - x.max(axis=1, keepdims=True))
out["softmax.x"] = x
out["softmax.out"] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

# rms_norm, eps=1e-5
x = rng.standard_normal((3, 8)).astype(np.float32)
out["rmsnorm.x"] = x
inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-5)
out["rmsnorm.out"] = (x * inv).astype(np.float32)

# rope NORM: head_dim=8, n_head=2, T=3, base=10000
hd, nh, T = 8, 2, 3
x = rng.standard_normal((T, nh, hd)).astype(np.float32)  # ne = [hd, nh, T]
pos = np.array([0, 1, 2], dtype=np.int32)
y = x.copy()
for t in range(T):
    for h in range(nh):
        for i in range(hd // 2):
            theta = pos[t] * (10000.0 ** (-2.0 * i / hd))
            c, s = np.cos(theta), np.sin(theta)
            x0, x1 = x[t, h, 2 * i], x[t, h, 2 * i + 1]
            y[t, h, 2 * i] = x0 * c - x1 * s
            y[t, h, 2 * i + 1] = x0 * s + x1 * c
out["rope.x"] = x
out["rope.pos"] = pos
out["rope.out"] = y.astype(np.float32)

# cross_entropy: logits [4 строки, vocab=10], one-hot targets
lg = rng.standard_normal((4, 10)).astype(np.float32)
tgt = np.zeros((4, 10), dtype=np.float32)
for r, c in enumerate([1, 0, 7, 3]):
    tgt[r, c] = 1.0
lz = lg - lg.max(axis=1, keepdims=True)
logsm = lz - np.log(np.exp(lz).sum(axis=1, keepdims=True))
loss = -(tgt * logsm).sum() / 4.0
out["xent.logits"] = lg
out["xent.targets"] = tgt
out["xent.out"] = np.array([loss], dtype=np.float32)

with open("crates/ggrs-core/tests/fixtures/ops.bin", "wb") as f:
    f.write(struct.pack("<I", len(out)))
    for name, arr in out.items():
        nb = name.encode()
        f.write(struct.pack("<I", len(nb)))
        f.write(nb)
        f.write(struct.pack("<I", 0 if arr.dtype == np.float32 else 1))
        # ne: numpy shape задом наперёд (ne0 — последняя ось numpy)
        ne = list(arr.shape[::-1]) + [1] * (4 - arr.ndim)
        f.write(struct.pack("<4I", *ne))
        f.write(arr.astype("<f4" if arr.dtype == np.float32 else "<i4").tobytes())
print(f"OK: {len(out)} тензоров")
