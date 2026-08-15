# Mamba Selective Scan CUDA

A from-scratch CUDA C++17 implementation of the selective state-space scan at
the center of a Mamba inference block. The project compares a sequential GPU
baseline, a custom hierarchical associative scan, CUB BlockScan, and a fused
recurrent inference kernel. It also provides FP16 storage with FP32 recurrence.

The implementation follows the selective state-space idea introduced in
[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
It is an educational inference implementation with deterministic random
weights; it does not include model training or pretrained checkpoints.

| Binary | Purpose |
|---|---|
| `mamba_demo` | Runs a complete Mamba block and reports latency, throughput, and an output checksum. |
| `mamba_benchmark` | Compares all four FP32 scan strategies across sequence lengths. |
| `test_mamba` | Checks CUDA results against the CPU reference, including cross-tile and FP16 cases. |

## Mamba Block

The complete inference path is:

```text
Input [B, L, D]
   |
   +-- RMSNorm
   |
   +-- Input projection [D -> 2 * inner]
   |      |                         |
   |      x                         z
   |      |                         |
   |      +-- depthwise causal Conv1D
   |      +-- SiLU
   |      +-- x projection -> dt, B, C
   |      +-- dt projection
   |      +-- fused selective scan
   |                                |
   +-------------------------- SiLU gate
                              |
                       Output projection
                              |
                       Residual connection
                              |
                       Output [B, L, D]
```

The dense projections are intentionally simple custom CUDA kernels. The focus
of this project is the scan and its memory behavior, rather than GEMM tuning.

## Selective State-Space Recurrence

For each batch item, channel `d`, state `n`, and token `t`, the discretized
recurrence is:

```text
dt[t,d] = softplus(delta[t,d] + delta_bias)
a[t,d,n] = exp(dt[t,d] * A[d,n])
b[t,d,n] = dt[t,d] * B[t,n] * u[t,d]

h[t,d,n] = a[t,d,n] * h[t-1,d,n] + b[t,d,n]
y[t,d]   = sum_n(C[t,n] * h[t,d,n]) + D[d] * u[t,d]
```

`A` is negative in the generated weights, which keeps the transition stable.
`B`, `C`, and `dt` depend on the input produced by the block projections.

Each recurrence step is represented as an affine pair `(a, b)`. Two adjacent
pairs compose associatively:

```text
(a1, b1) followed by (a2, b2)
    = (a2 * a1, a2 * b1 + b2)
```

Associativity allows all prefixes in a sequence tile to be evaluated in
parallel without changing the recurrence result.

## CUDA Implementations

### Naive recurrent

One CUDA thread owns one `(batch, channel, state)` row and advances through the
sequence serially. It is the GPU correctness and performance baseline. The
complete state tensor is materialized for the output reduction.

### Custom hierarchical scan

One block owns one `(batch, channel, state)` row and processes 256-token tiles.
Each warp performs an inclusive affine scan with `__shfl_up_sync`. The eight
warp totals are then scanned by the first warp and distributed back to the
block. A carry pair connects consecutive tiles.

### CUB BlockScan

Uses `cub::BlockScan` with the same affine composition operator, tile size, and
state layout as the custom path. This makes it a direct comparison between the
hand-written hierarchy and NVIDIA's reusable block primitive. CUB is included
with the CUDA Toolkit; see the
[CUB documentation](https://docs.nvidia.com/cuda/cub/).

### Fused recurrent

One block owns a `(batch, channel)` row. Threads keep the small state dimension
in FP32 registers and reduce `C * h` in shared memory at every token. This path
does not materialize `[B, L, inner, state]`, so its additional state workspace
is `O(B * inner * state)` rather than `O(B * L * inner * state)`. It is used by
the complete Mamba block.

The FP16 variant stores inputs, parameters, and output as `half`, while the
state recurrence and reduction remain FP32.

## Build

Requirements:

- CUDA Toolkit 12.x
- CMake 3.18+
- GCC 9+ or Clang 10+
- NVIDIA GPU with compute capability 7.5+

```bash
cd 19.Mamba-Selective-Scan-CUDA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The default architecture list is `75;80;86;89`. Override it when needed:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

## Run

```bash
# Complete block with default dimensions
./build/mamba_demo

# Configure the demo
./build/mamba_demo --seq 1024 --dim 128 --inner 256 --state 16 --dt-rank 8

# Sweep sequence lengths
./build/mamba_benchmark

# Benchmark one length
./build/mamba_benchmark --seq 8192

# Numerical correctness
./build/test_mamba
```

The test suite covers:

1. Naive GPU recurrence against the CPU reference.
2. Custom warp/block scan against the CPU reference.
3. A 513-token custom scan crossing three tiles.
4. CUB BlockScan across multiple tiles.
5. Fused FP32 recurrence against the CPU reference.
6. Fused FP16 storage against the FP32 CPU reference.
7. Deterministic, finite output from the complete Mamba block.

## Tensor Layouts

All tensors are contiguous and row major:

| Tensor | Shape |
|---|---|
| `u`, `delta`, `y` | `[batch, seq_len, dim]` |
| `A` | `[dim, state_size]` |
| `B`, `C` | `[batch, seq_len, state_size]` |
| `D` | `[dim]` |
| materialized state | `[batch, seq_len, dim, state_size]` |

The public scan API supports state sizes from 1 through 128 and arbitrary
positive sequence lengths, including lengths that are not multiples of 256.

## File Layout

```text
19.Mamba-Selective-Scan-CUDA/
|-- CMakeLists.txt
|-- README.md
|-- include/
|   |-- cuda_utils.h
|   |-- mamba_block.h
|   |-- mamba_reference.h
|   `-- selective_scan.cuh
|-- src/
|   |-- benchmark.cu
|   |-- main.cu
|   |-- mamba_block.cu
|   `-- selective_scan.cu
`-- tests/
    `-- test_mamba.cu
```

## Profiling

```bash
nsys profile --stats=true ./build/mamba_demo --seq 2048
ncu --set full --kernel-name regex:parallel_state_kernel ./build/mamba_benchmark --seq 4096
compute-sanitizer --tool memcheck ./build/test_mamba
```

The benchmark prints measured results from the current GPU. No device-specific
numbers are embedded in this document.
