# 13. GPT-Style Transformer Decoder Inference Engine

**CUDA C++ GPT-style autoregressive inference with KV Cache, GQA, INT8 quantization, and Speculative Decoding**

High-performance transformer decoder inference engine built entirely in CUDA C++. Demonstrates production-quality techniques used in modern LLM inference: Grouped Query Attention, rotary embeddings (RoPE), SwiGLU FFN, KV caching, multi-precision quantization, and speculative decoding.

---

## Features

- **KV Cache** — O(1) memory per new token during autoregressive decode
- **Grouped Query Attention (GQA)** — n_kv_heads < n_heads reduces KV cache size by kv_groups×
- **RoPE** — Rotary Positional Embeddings (Llama-style, no learned position embeddings)
- **SwiGLU FFN** — Gated feed-forward network (gate_proj × SiLU + up_proj → down_proj)
- **RMSNorm** — Pre-normalization with warp-shuffle reduction (no LayerNorm bias)
- **FP16 cuBLAS GEMM** — Tensor Core accelerated matrix multiplications
- **INT8 per-channel quantization** — ~2× memory reduction with scale-based dequant
- **Speculative Decoding** — Draft model generates K tokens; target verifies for throughput gain
- **Char-level tokenizer** — Zero-dependency tokenizer (256-token vocab, 1 byte = 1 token)
- **Interactive CLI** — Prompt → streaming character output with live tokens/sec stats

---

## Architecture

```
Token Embedding  [vocab_size, d_model]
       │
  ┌────▼────────────────────────────────────────────────────┐
  │  Decoder Layer × n_layers                               │
  │                                                          │
  │  RMSNorm → Q, K, V projections (cuBLAS FP16)           │
  │          → RoPE(Q, pos), RoPE(K, pos)                  │
  │          → Write K, V to KV Cache                       │
  │          → GQA Attention (scores→softmax→weighted sum)  │
  │          → Output projection (cuBLAS) + Residual        │
  │                                                          │
  │  RMSNorm → Gate, Up projections (cuBLAS FP16)          │
  │          → SwiGLU(gate, up)                             │
  │          → Down projection (cuBLAS) + Residual          │
  └──────────────────────────────────────────────────────────┘
       │
  Final RMSNorm
       │
  LM Head [vocab_size, d_model] (cuBLAS)
       │
  Greedy / Top-K Sampling  →  Next Token
```

### Grouped Query Attention (GQA)

```
n_heads = 8  (query heads)
n_kv_heads = 2  (key-value heads)
kv_groups = 4  (each KV head is shared by 4 query heads)

Memory saved vs MHA: 4× smaller KV cache
```

### KV Cache Layout

```
d_k / d_v: [n_layers, n_kv_heads, max_seq_len, head_dim]

Layer 0, KV-head 0, position 5:
  d_k + 0 * (n_kv_heads * max_seq_len * head_dim)
      + 0 * (max_seq_len * head_dim)
      + 5 * head_dim
```

---

## Build

### Requirements

- CUDA 12.0+ (cuBLAS, cuRAND)
- CMake 3.18+
- GCC 9+ / Clang 10+
- GPU: sm_75+ (Turing / RTX 2000+)

### Compile

```bash
cd 13.GPT-Transformer-Inference
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Targets built:

| Binary | Description |
|--------|-------------|
| `inference` | Interactive text generation |
| `benchmark` | Tokens/sec benchmark across precision modes |
| `test_attention` | Kernel correctness tests (CPU reference vs GPU) |

---

## Usage

### Run kernel tests first

```bash
./test_attention
```

Expected output:
```
[Test] RMSNorm...          PASS  max_err=0.0035
[Test] SwiGLU...           PASS  max_err=0.0018
[Test] Attention scores... PASS  max_err=0.00042
[Test] Embedding lookup... PASS  max_err=0.0010
[Test] RoPE invertibility. PASS  max_err=0.0081
All 5 tests PASSED!
```

### Benchmark

```bash
# Default: FP16 + INT8 comparison, 6-layer 512-dim model
./benchmark

# Custom model size (GPT-2 Medium scale)
./benchmark --layers 24 --d-model 1024 --n-heads 16 --n-kv-heads 4 \
            --ff-dim 4096 --tokens 50

# Only FP16
./benchmark --mode fp16
```

Example output (RTX 4070, default config):
```
=== GPT Transformer Inference Benchmark ===
  Config: layers=6 d_model=512 n_heads=8 n_kv_heads=2 (GQA)
  ff_dim=2048 vocab=256 prompt_len=128 generate=100 tokens

Mode       Tokens/sec    ms/token   Speedup
------------------------------------------------------
FP16        3842.1 tok/s    0.26 ms    1.00x
INT8        4513.7 tok/s    0.22 ms    1.17x
```

### Interactive inference

```bash
# Random weights (for testing mechanics)
./inference --load-random

# Specific prompt
./inference --load-random --prompt "Hello, world!" --max-tokens 100

# Top-K sampling with temperature
./inference --load-random --sampling top-k --top-k 40 --temp 0.8 \
            --prompt "Once upon a time"

# Load saved weights
./inference --save data/weights.bin --load-random   # first: save random weights
./inference --model data/weights.bin --prompt "Test"  # then: load and run
```

### Speculative decoding

Speculative decoding is demonstrated inside `benchmark.cu` and `speculative.cu`.
The draft model uses 2 layers (vs target's 6) for faster candidate generation.

```
Expected speedup depends on acceptance rate:
  α = 0.9 (high similarity between draft and target): ~3.5x
  α = 0.5 (moderate): ~1.8x
  α = 0.1 (low): ~1.1x (barely faster)
```

---

## File Structure

```
13.GPT-Transformer-Inference/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── cuda_utils.h              CUDA_CHECK, CudaTimer, memory helpers
│   ├── transformer_kernels.cuh   All kernel declarations
│   ├── transformer.h             TransformerConfig, LayerWeights, TransformerModel
│   ├── kv_cache.h                KVCache struct
│   └── tokenizer.h               CharTokenizer (256-token char-level)
├── src/
│   ├── transformer_kernels.cu    RMSNorm, RoPE, Softmax, SwiGLU, Attention, Sampling
│   ├── transformer.cu            Forward pass, weight init/load/save
│   ├── kv_cache.cu               KV cache memory management
│   ├── quantization.cu           INT8 per-channel quantize / dequantize
│   ├── speculative.cu            SpeculativeDecoder class + benchmark
│   ├── inference.cu              Interactive generation executable
│   └── benchmark.cu              Precision comparison benchmark
└── tests/
    └── test_attention.cu         CPU vs GPU correctness tests
```

---

## Technical Details

### Custom CUDA Kernels

| Kernel | Description | Optimization |
|--------|-------------|--------------|
| `rmsnorm_kernel` | RMSNorm with warp shuffle | Single block, shared-mem reduction |
| `rope_kernel` | RoPE in-place | 1 thread per (cos,sin) pair |
| `softmax_kernel` | Numerically stable softmax | Online max + warp shuffle |
| `swiglu_kernel` | SwiGLU activation | Element-wise, high occupancy |
| `attention_scores_kernel` | GQA QK^T | Per-head-per-position thread |
| `attention_output_kernel` | Weighted sum V | Per-head-per-dim thread |
| `argmax_kernel` | Greedy sampling | Shared-mem tree reduction |
| `top_k_sampling_kernel` | Top-K with temperature | Single thread (K≤40) |
| `quantize_fp16_to_int8_kernel` | Per-row INT8 quant | Row-parallel with reduction |

### cuBLAS GEMM (FP16 Tensor Cores)

All large matrix multiplications (Q, K, V, O projections; FFN gate, up, down) use
`cublasHgemm` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to utilize Tensor Cores.
For single-token inference: all GEMMs are `[1, k] × [k, n] → [1, n]` (GEMV-like).

### Memory Usage

| Component | Formula | Example (6L-512D-GQA-2048) |
|-----------|---------|---------------------------|
| KV Cache | 2 × L × H_kv × S × D_h × 2B | 6×2×2048×64×2 = 6.0 MB |
| Model weights | ~12 × L × D² + 2 × V × D | ~28 MB |
| Activations | Tiny (single token) | < 1 MB |

---

## CUDA Optimization Highlights

- **Warp shuffle reductions** in RMSNorm and Softmax (no atomics needed)
- **Tensor Core FP16 GEMM** via cuBLAS for all projection layers
- **RoPE in-place** (no extra buffer, head_dim/2 threads per head)
- **KV cache** eliminates recomputation of past attention keys and values
- **GQA** reduces KV cache size by n_heads/n_kv_heads = 4× vs standard MHA
- **INT8 per-channel quantization** with scale factors for 2× memory reduction
- **Speculative decoding** leverages draft model acceptance to amortize target model cost

---

## Learning Path

This project builds on:
- **11. LSTM CUDA** — Recurrent state, gate computation patterns
- **12. DQN CUDA** — Neural network layer struct patterns, Adam optimizer

Recommended next:
- **Flash Attention** — IO-aware attention for long sequences
- **Multi-GPU pipeline** — Tensor parallelism across GPUs

---

## Performance Tips

```bash
# Profile with Nsight Systems
nsys profile --stats=true ./benchmark

# Kernel-level metrics with Nsight Compute
ncu --set full --target-processes all ./benchmark --tokens 10

# Check GPU utilization during inference
nvidia-smi dmon -s u -d 1
```
