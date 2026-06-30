# FlashAttention-CUDA

A from-scratch CUDA implementation of **fused multi-head attention** — the
FlashAttention idea built up from first principles. The fused kernels stream
K/V tiles through shared memory and keep a running **online-softmax**
accumulator, so they **never materialise the `[seq_len × seq_len]` score
matrix**. Memory drops from `O(N²)` to `O(N)` and, on an RTX 4070 Laptop, the
forward is **~8–10× faster** than the naive materialised baseline.

Everything is custom CUDA C++17 — no cuDNN, no cuBLAS, no Python. The point of
the project is the kernel, not a trained model.

| Binary            | What it does |
|-------------------|--------------|
| `flash_demo`      | One MHA forward with the fused kernel, checked against the CPU reference; prints the `O(N)` vs `O(N²)` memory picture. |
| `flash_benchmark` | Latency / GFLOP-s / memory: naive vs flash (thread-row, warp-row) vs FP16, across sequence lengths. |
| `test_flash`      | 12 numerical-correctness tests vs a CPU reference (forward, causal, FP16, log-sum-exp, backward). |

---

## Contents

1. [The online-softmax idea](#the-online-softmax-idea)
2. [Kernels](#kernels)
3. [Two forward designs (thread-row vs warp-row)](#two-forward-designs-thread-row-vs-warp-row)
4. [Backward pass](#backward-pass)
5. [Build & run](#build--run)
6. [Benchmarks](#benchmarks)
7. [File layout](#file-layout)
8. [Design notes & gotchas](#design-notes--gotchas)

---

## The online-softmax idea

Standard attention computes, per head:

```
S = Q · Kᵀ · scale         # [N, N]   ← the O(N²) matrix
P = softmax(S, axis=-1)    # [N, N]
O = P · V                  # [N, D]
```

Materialising `S` (and `P`) is what makes long sequences blow up memory and
saturate HBM bandwidth. FlashAttention rewrites the softmax as a **streaming
recurrence** so a query row only ever holds three running scalars/vectors —
`m` (running max), `l` (running denominator), and the output accumulator `acc`
— while K/V are consumed one tile at a time:

```
for each key tile:
    s        = scale · (q · kⱼ)
    m_new    = max(m, s)
    corr     = exp(m − m_new)          # rescale what we have so far
    p        = exp(s − m_new)
    l        = l · corr + p
    acc      = acc · corr + p · vⱼ
    m        = m_new
O = acc / l                            # one division at the very end
```

This is mathematically identical to the materialised version (the tests check
`max|flash − reference| ≈ 1e-7`) but uses `O(D)` state per query instead of
`O(N)`, and the score `s` lives in a register, never in DRAM.

`L = m + log(l)` (the log-sum-exp) is the one extra scalar saved per row — it is
all the backward pass needs to recompute `P` without storing it.

---

## Kernels

| Kernel | Role |
|--------|------|
| `naive_scores / naive_softmax / naive_output` | Materialised `[N×N]` attention — correctness anchor **and** the memory baseline the flash path is measured against. |
| `flash_fwd_thread_kernel<D>` | Fused forward, **one thread per query row** (fastest for `D ≤ 128`). |
| `flash_fwd_warp_kernel<D>`   | Fused forward, **one warp per query row**, head_dim spread across lanes. |
| `flash_fwd_fp16_kernel<D>`   | Thread-row forward with **FP16 I/O**, FP32 accumulate. |
| `bwd_preprocess_kernel`      | `Δᵢ = Σ_d dOᵢ·Oᵢ` row reduction. |
| `flash_bwd_dq_kernel<D>`     | `dQ`, one warp per query row, recompute `P`. |
| `flash_bwd_dkv_kernel<D>`    | `dK`, `dV`, one warp per key row, recompute `P`. |

`head_dim ∈ {32, 64, 128}` is templated (compile-time unroll); causal masking,
arbitrary `batch`/`num_heads`, and non-power-of-two `seq_len` are all supported.

---

## Two forward designs (thread-row vs warp-row)

The repo ships **two** forward strategies because the obvious "make it
warp-parallel" instinct turns out to be the *slower* one for these head dims —
a result worth seeing measured rather than assumed.

**`thread-row`** — one thread owns a query row, the whole `head_dim` lives in
registers, and a block cooperatively stages a K/V tile into shared memory. The
`q·k` dot product is a fully-unrolled register loop: high ILP, **no
warp-shuffle**.

**`warp-row`** — one warp owns a query row, `head_dim` is split across the 32
lanes, and each `q·k` dot is a 5-step `__shfl_down` reduction + broadcast. The
online-softmax scalars (`m`, `l`, `corr`, `p`) are then recomputed redundantly
by all 32 lanes.

For `head_dim ≤ 128` the per-key shuffle and redundant scalar work dominate, so
**thread-row wins by ~2–3.4×** (see table). warp-row earns its keep only when
`head_dim` grows large enough that thread-row's register file spills — and it is
the natural layout to later swap the shuffle dot-product for a Tensor-Core
`mma.sync` tile. `flash_attention_forward()` defaults to **thread-row**.

> Takeaway: spreading a small contraction dimension across a warp trades cheap
> register FMAs for expensive shuffles. Parallelise the *rows*, not the 64-wide
> dot, until the dot is wide enough (or a Tensor Core) to pay for itself.

---

## Backward pass

Recompute-based and **atomic-free**. Given `Q,K,V,O,dO` and the saved `L`:

```
Δᵢ      = Σ_d dOᵢ · Oᵢ
Pᵢⱼ     = exp(scale·qᵢ·kⱼ − Lᵢ)          # recomputed, never stored
dVⱼ    += Pᵢⱼ · dOᵢ
dPᵢⱼ    = dOᵢ · vⱼ
dSᵢⱼ    = Pᵢⱼ · (dPᵢⱼ − Δᵢ) · scale
dQᵢ    += dSᵢⱼ · kⱼ
dKⱼ    += dSᵢⱼ · qᵢ
```

Write conflicts are avoided by **splitting the loops by output ownership**: the
`dQ` kernel assigns one warp per query row `i` and sums over `j`; the `dK/dV`
kernel assigns one warp per key row `j` and sums over `i`. No two warps ever
write the same row, so there are zero atomics. Validated against the CPU
reference to `relerr ≈ 3e-7`.

---

## Build & run

```bash
cd 18.FlashAttention-CUDA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/test_flash                       # 12 correctness tests
./build/flash_demo --seq 1024 --heads 8  # demo + memory table  (add --noncausal)
./build/flash_benchmark                  # non-causal sweep      (add --causal)
./build/flash_benchmark --causal --dim 64 --heads 8
```

Requires CUDA 12.x and an SM 7.5+ GPU (`CMAKE_CUDA_ARCHITECTURES = 75 80 86 89`).

---

## Benchmarks

RTX 4070 Laptop (SM 8.9), `batch=1, heads=8, head_dim=64`, FP32 unless noted.
Latency is per forward, averaged after warmup.

**Non-causal**

| seq  | naive (ms) | flash thread (ms) | speedup | thread GFLOP/s | warp (ms) | fp16 (ms) | scores buffer |
|-----:|-----------:|------------------:|--------:|---------------:|----------:|----------:|--------------:|
| 256  |   0.642    | **0.119**         | 5.4×    | 1132           | 0.194     | 0.121     | 2 MB |
| 512  |   2.331    | **0.290**         | 8.0×    | 1854           | 0.696     | 0.324     | 8 MB |
| 1024 |   8.535    | **1.046**         | 8.2×    | 2053           | 3.067     | 1.056     | 34 MB |
| 2048 |  35.796    | **4.108**         | 8.7×    | 2091           | 13.404    | 3.939     | 134 MB |
| 4096 | 149.063    | **15.480**        | 9.6×    | 2220           | 51.355    | 14.342    | 537 MB |

**Causal**

| seq  | naive (ms) | flash thread (ms) | speedup | warp (ms) | fp16 (ms) |
|-----:|-----------:|------------------:|--------:|----------:|----------:|
| 1024 |   5.058    | **0.963**         | 5.3×    | 1.531     | 1.070     |
| 2048 |  21.761    | **3.397**         | 6.4×    | 7.457     | 3.244     |
| 4096 |  90.110    | **11.964**        | 7.5×    | 29.425    | 11.517    |

**Memory** — the `scores buffer` column is the `O(N²)` matrix flash never
allocates. Flash's footprint is just `O + L`:

| seq  | flash O+L | naive scores | ratio |
|-----:|----------:|-------------:|------:|
| 512  |  1.06 MB  |    8.39 MB   |   8× |
| 2048 |  4.26 MB  |  134.22 MB   |  32× |
| 4096 |  8.52 MB  |  536.87 MB   |  63× |
| 8192 | 17.04 MB  | 2147.48 MB   | 126× |

The naive path is automatically **skipped (OOM)** once its score buffer would
exceed half of free VRAM — exactly the wall flash is built to avoid.

---

## File layout

```
18.FlashAttention-CUDA/
├── include/
│   ├── cuda_utils.h          # CUDA_CHECK, CudaTimer (shared repo helper)
│   ├── flash_attention.cuh   # config struct + launcher API
│   └── attention_ref.h       # header-only CPU reference (fwd + bwd)
├── src/
│   ├── flash_kernels.cu       # all kernels + host launchers
│   ├── main.cu               # flash_demo
│   └── benchmark.cu          # flash_benchmark
├── tests/
│   └── test_flash.cu         # 12 tests vs CPU reference
└── CMakeLists.txt
```

Tensor layout everywhere: `Q,K,V,O = [batch, num_heads, seq_len, head_dim]`,
`L = [batch, num_heads, seq_len]`, row-major contiguous.

---

## Design notes & gotchas

- **Numerical stability is free.** The running-max subtraction means scores
  never exponentiate raw — the stability test feeds `N(0, 8²)` logits and the
  output stays finite with `1e-5` error. No `S` ever overflows because no `S`
  is ever stored.
- **Causal masking is a `break`, not a multiply.** Keys are visited in order,
  so once `kⱼ > qᵢ` the loop exits — causal does *half* the work, not the same
  work with masked-out terms. The early `break` is warp-uniform (every lane of a
  warp shares the same query row), so it costs no divergence.
- **`__syncthreads()` lives only in the tile loop.** A thread/warp that finished
  its causal keys still re-enters the cooperative K/V load for the rows its
  block-mates need, so the barrier is always reached by every thread — no
  deadlock from early exits.
- **head_dim register pressure.** thread-row keeps `2·head_dim` floats live
  (`q` + `acc`); at `head_dim=128` that brushes the 255-register cap and the
  launcher drops the block to 32 rows to keep shared memory ≤ 32 KB. This is the
  regime where warp-row's tiny per-lane footprint starts to matter.
- **FP16 is half-I/O, FP32-accumulate.** Q/K/V/O are stored as `__half`
  (halved bandwidth + shared memory) but the dot products and the softmax
  accumulate in FP32, so accuracy stays at `relerr ≈ 4e-4` — and at long
  sequences it matches or beats the FP32 thread-row kernel. A true Tensor-Core
  `mma.sync` path (warp-row layout) is the natural next step.
