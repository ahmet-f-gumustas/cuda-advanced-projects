#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

// ============================================================================
// FlashAttention-CUDA — from-scratch fused multi-head attention
// ----------------------------------------------------------------------------
// Tensor layout (row-major, contiguous):
//
//     Q, K, V, O   :  [batch, num_heads, seq_len, head_dim]
//     L (logsumexp):  [batch, num_heads, seq_len]
//
// The "flash" forward never materialises the [seq_len x seq_len] score matrix:
// it streams K/V tiles through shared memory and keeps a running (max, sum,
// output) accumulator using the online-softmax recurrence. Memory is O(N) per
// query instead of O(N^2).
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------------
// Configuration for one attention call.
// ----------------------------------------------------------------------------
struct FlashAttnConfig {
    int   batch     = 1;
    int   num_heads = 1;
    int   seq_len   = 128;
    int   head_dim  = 64;     // must be one of {32, 64, 128} for the flash path
    bool  causal    = false;  // GPT-style lower-triangular masking
    float scale     = 0.0f;   // <= 0  ->  1 / sqrt(head_dim)

    long long num_elems() const {
        return (long long)batch * num_heads * seq_len * head_dim;
    }
    float eff_scale() const {
        return scale > 0.0f ? scale : 1.0f / sqrtf((float)head_dim);
    }
};

// Which forward kernel design to run. These are two *strategies*, not a
// v1->v2 improvement: for head_dim <= 128 THREAD_ROW is the faster one (no
// per-key warp-shuffle, more ILP); WARP_ROW spreads head_dim across a warp and
// is the substrate that scales to large head_dim / Tensor-Core MMA. See README.
enum FlashKernel {
    FLASH_THREAD_ROW = 1,   // one thread per query row, full head_dim in registers
    FLASH_WARP_ROW   = 2     // one warp per query row, lane-parallel head_dim + shuffle
};

// ----------------------------------------------------------------------------
// Naive (materialised) attention — reference + memory/perf baseline.
//   d_scores : workspace of [batch*num_heads*seq_len*seq_len] floats. This is
//              the O(N^2) buffer the flash path avoids; pass a real allocation.
// ----------------------------------------------------------------------------
void naive_attention_forward(const float* d_Q, const float* d_K, const float* d_V,
                             float* d_O, float* d_scores,
                             const FlashAttnConfig& cfg,
                             cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// FlashAttention forward (FP32).
//   d_L may be nullptr if the log-sum-exp (needed only for backward) is unused.
// ----------------------------------------------------------------------------
void flash_attention_forward(const float* d_Q, const float* d_K, const float* d_V,
                             float* d_O, float* d_L,
                             const FlashAttnConfig& cfg,
                             FlashKernel kernel = FLASH_THREAD_ROW,
                             cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// FlashAttention forward (FP16 storage, FP32 accumulate) — thread-per-row
// design with half-precision I/O. d_O is half; d_L stays float.
// ----------------------------------------------------------------------------
void flash_attention_forward_fp16(const __half* d_Q, const __half* d_K, const __half* d_V,
                                  __half* d_O, float* d_L,
                                  const FlashAttnConfig& cfg,
                                  cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// FlashAttention backward (FP32, recompute-based, atomic-free).
//   Inputs : Q,K,V, O (forward output), dO (grad wrt O), L (logsumexp).
//   Outputs: dQ, dK, dV.
//   d_workspace : [batch*num_heads*seq_len] floats for the D = rowsum(dO o O)
//                 reduction. Pass a real allocation.
// ----------------------------------------------------------------------------
void flash_attention_backward(const float* d_Q, const float* d_K, const float* d_V,
                              const float* d_O, const float* d_dO, const float* d_L,
                              float* d_dQ, float* d_dK, float* d_dV,
                              float* d_workspace,
                              const FlashAttnConfig& cfg,
                              cudaStream_t stream = 0);

#endif // FLASH_ATTENTION_CUH
