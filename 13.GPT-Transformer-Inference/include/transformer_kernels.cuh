#ifndef TRANSFORMER_KERNELS_CUH
#define TRANSFORMER_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

// ============================================================================
// Constants
// ============================================================================

constexpr int   PT_WARP_SIZE       = 32;
constexpr int   PT_BLOCK_SIZE      = 256;
constexpr int   PT_MAX_SEQ_LEN     = 2048;
constexpr float PT_RMSNORM_EPS     = 1e-6f;
constexpr float PT_ROPE_THETA      = 10000.0f;

// ============================================================================
// Element-wise / reduction kernels
// ============================================================================

// RMSNorm: out[i] = in[i] / rms(in) * w[i]
// Grid: (1,)  Block: (min(d, 512),)
// d_out and d_in can be the same pointer (in-place) if d <= 512
__global__ void rmsnorm_kernel(half*       d_out,
                               const half* d_in,
                               const half* d_weight,
                               int         d,
                               float       eps);

// Add residual: inout[i] += src[i]
// Grid: (d / BLOCK,)  Block: (BLOCK,)
__global__ void add_residual_kernel(half*       d_inout,
                                    const half* d_src,
                                    int         d);

// RoPE: in-place rotary positional embedding applied to Q or K
// x layout: [n_heads, head_dim]
// Grid: (n_heads,)  Block: (head_dim / 2,)
__global__ void rope_kernel(half* d_x,
                             int   pos,
                             int   n_heads,
                             int   head_dim,
                             float theta);

// Softmax in-place over seq_len values, one per head
// scores layout: [n_heads, seq_len]  (float, not half)
// Grid: (n_heads,)  Block: (min(seq_len, 512),)
__global__ void softmax_kernel(float* d_scores,
                                int    n_heads,
                                int    seq_len);

// SwiGLU: hidden[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x * sigmoid(x)
// Grid: (ff_dim / BLOCK,)  Block: (BLOCK,)
__global__ void swiglu_kernel(half*       d_hidden,
                               const half* d_gate,
                               const half* d_up,
                               int         ff_dim);

// Embedding lookup: copy row token_id from embedding table
// Grid: (d_model / BLOCK,)  Block: (BLOCK,)
__global__ void embedding_lookup_kernel(half*       d_out,
                                        const half* d_table,
                                        int         token_id,
                                        int         d_model);

// Write Q/K into KV cache at position pos
// k layout: [n_kv_heads, head_dim] → kv_cache[layer_offset + kv_h * max_seq * hd + pos * hd]
// Grid: (n_kv_heads,)  Block: (head_dim,)
__global__ void write_kv_cache_kernel(half*       d_k_cache,
                                       half*       d_v_cache,
                                       const half* d_k,
                                       const half* d_v,
                                       int         pos,
                                       int         n_kv_heads,
                                       int         max_seq_len,
                                       int         head_dim);

// ============================================================================
// Attention kernels (custom — seq-len ≤ 2048, single-token decode)
// ============================================================================

// Compute attention scores: scores[h][t] = dot(Q[h], K_cache[kv_h][t]) * scale
// GQA: kv_h = h / kv_groups  (kv_groups = n_heads / n_kv_heads)
// Q layout      : [n_heads,    head_dim]
// K_cache layout: [n_kv_heads, max_seq_len, head_dim]
// scores layout : [n_heads,    seq_len]   (float, pre-softmax)
// Grid: (n_heads, seq_len / BLOCK_T + 1)  Block: (BLOCK_T,)
__global__ void attention_scores_kernel(float*      d_scores,
                                         const half* d_q,
                                         const half* d_k_cache,
                                         int         n_heads,
                                         int         n_kv_heads,
                                         int         head_dim,
                                         int         seq_len,
                                         int         max_seq_len,
                                         float       scale);

// Weighted sum of V: out[h][i] = sum_t probs[h][t] * V_cache[kv_h][t][i]
// probs layout  : [n_heads,    seq_len]   (after softmax)
// V_cache layout: [n_kv_heads, max_seq_len, head_dim]
// out layout    : [n_heads,    head_dim]
// Grid: (n_heads,)  Block: (head_dim,)
__global__ void attention_output_kernel(half*        d_out,
                                         const float* d_probs,
                                         const half*  d_v_cache,
                                         int          n_heads,
                                         int          n_kv_heads,
                                         int          seq_len,
                                         int          max_seq_len,
                                         int          head_dim);

// ============================================================================
// Quantization kernels
// ============================================================================

// Per-row quantize FP16 → INT8
// scale[row] = max(|w[row, :]|) / 127
// Grid: (rows,)  Block: (min(cols, BLOCK),)
__global__ void quantize_fp16_to_int8_kernel(const half* d_src,
                                              int8_t*     d_dst,
                                              float*      d_scales,
                                              int         rows,
                                              int         cols);

// Per-row dequantize INT8 → FP16
// Grid: (rows,)  Block: (min(cols, BLOCK),)
__global__ void dequantize_int8_to_fp16_kernel(const int8_t* d_src,
                                                half*         d_dst,
                                                const float*  d_scales,
                                                int           rows,
                                                int           cols);

// FP32 → FP16 cast (for initial quantization pass)
// Grid: ((n + BLOCK - 1) / BLOCK,)  Block: (BLOCK,)
__global__ void fp32_to_fp16_kernel(const float* d_src, half* d_dst, int n);

// FP16 logits → FP32 (for sampling)
// Grid: ((vocab / BLOCK + 1),)  Block: (BLOCK,)
__global__ void logits_fp16_to_fp32_kernel(const half* d_src, float* d_dst, int vocab_size);

// ============================================================================
// Sampling kernels
// ============================================================================

// Greedy decode: argmax over float logits
// Grid: (1,)  Block: (min(vocab_size, 512),)
__global__ void argmax_kernel(const float* d_logits,
                               int*         d_out_token,
                               float*       d_max_val,    // shared workspace
                               int          vocab_size);

// Top-K sampling with temperature
// Modifies logits in-place (partial sort), then samples
// Grid: (1,)  Block: (512,)
__global__ void top_k_sampling_kernel(float*       d_logits,
                                       int*         d_out_token,
                                       int          vocab_size,
                                       int          k,
                                       float        temperature,
                                       curandState* d_rng_state);

// Scale logits by 1/temperature in-place
// Grid: ((vocab_size + BLOCK - 1) / BLOCK,)  Block: (BLOCK,)
__global__ void scale_logits_kernel(float* d_logits, float inv_temp, int vocab_size);

// ============================================================================
// curand state initialization
// ============================================================================

// One thread per sampling position (1 for inference, 1 for speculative)
__global__ void init_curand_state_kernel(curandState* d_state,
                                          unsigned long long seed,
                                          int                n);

#endif // TRANSFORMER_KERNELS_CUH
