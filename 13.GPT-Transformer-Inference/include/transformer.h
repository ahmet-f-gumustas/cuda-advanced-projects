#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "kv_cache.h"
#include "transformer_kernels.cuh"

// ============================================================================
// Quantization mode
// ============================================================================

enum class QuantMode {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2
};

// ============================================================================
// Model configuration
// ============================================================================

struct TransformerConfig {
    int       vocab_size  = 256;     // Default: char-level (256 bytes)
    int       n_layers    = 6;       // Decoder layer count
    int       d_model     = 512;     // Model dimension
    int       n_heads     = 8;       // Query head count
    int       n_kv_heads  = 2;       // KV head count (GQA ratio = n_heads/n_kv_heads = 4)
    int       head_dim    = 64;      // = d_model / n_heads
    int       ff_dim      = 2048;    // SwiGLU intermediate dimension
    int       max_seq_len = 2048;    // Maximum sequence length
    float     rope_theta  = 10000.0f;
    QuantMode quant       = QuantMode::FP16;

    // Derived helpers
    int kv_dim()     const { return n_kv_heads * head_dim; }
    int q_dim()      const { return n_heads * head_dim; }
    int kv_groups()  const { return n_heads / n_kv_heads; }  // queries per KV head
};

// ============================================================================
// Per-layer device weights (FP16)
// ============================================================================

struct LayerWeights {
    // Attention norm (RMSNorm)
    half* d_attn_norm;          // [d_model]

    // Attention projections
    half* d_wq;                 // [n_heads * head_dim, d_model]
    half* d_wk;                 // [n_kv_heads * head_dim, d_model]
    half* d_wv;                 // [n_kv_heads * head_dim, d_model]
    half* d_wo;                 // [d_model, n_heads * head_dim]

    // FFN norm (RMSNorm)
    half* d_ffn_norm;           // [d_model]

    // FFN projections (SwiGLU: gate_proj + up_proj → down_proj)
    half* d_gate_proj;          // [ff_dim, d_model]
    half* d_up_proj;            // [ff_dim, d_model]
    half* d_down_proj;          // [d_model, ff_dim]

    // INT8 per-row scales (only allocated when quant == INT8)
    float* d_wq_scale;          // [n_heads * head_dim]
    float* d_wk_scale;          // [n_kv_heads * head_dim]
    float* d_wv_scale;          // [n_kv_heads * head_dim]
    float* d_wo_scale;          // [d_model]
    float* d_gate_scale;        // [ff_dim]
    float* d_up_scale;          // [ff_dim]
    float* d_down_scale;        // [d_model]

    // INT8 weight copies (only allocated when quant == INT8)
    int8_t* d_wq_int8;
    int8_t* d_wk_int8;
    int8_t* d_wv_int8;
    int8_t* d_wo_int8;
    int8_t* d_gate_int8;
    int8_t* d_up_int8;
    int8_t* d_down_int8;
};

// ============================================================================
// Full model device weights
// ============================================================================

struct ModelWeights {
    half* d_embed;              // [vocab_size, d_model] — token embedding table
    std::vector<LayerWeights> layers;   // [n_layers]
    half* d_final_norm;         // [d_model]
    half* d_lm_head;            // [vocab_size, d_model] — unembedding / logit projection
};

// ============================================================================
// TransformerModel
// ============================================================================

class TransformerModel {
public:
    explicit TransformerModel(const TransformerConfig& cfg);
    ~TransformerModel();

    // Disable copy
    TransformerModel(const TransformerModel&)            = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    // Initialize weights with random values (Glorot uniform)
    // Useful for benchmarking without real checkpoint
    void initRandom(unsigned long long seed = 42);

    // Load weights from a simple binary file:
    //   [header: magic(4B) + n_layers(4B) + d_model(4B) + n_heads(4B) +
    //            n_kv_heads(4B) + ff_dim(4B) + vocab_size(4B)]
    //   [weights: embed, layer0_attn_norm, layer0_wq, ..., final_norm, lm_head]
    //   All stored as float32, converted to FP16 on load.
    bool loadWeights(const std::string& path);

    // Save current (random) weights to binary file for reproducibility
    bool saveWeights(const std::string& path) const;

    // Single-token forward pass using KV cache
    //   token_id  — input token
    //   d_logits  — output float buffer [vocab_size] (device memory, FP32)
    //   kv_cache  — KV cache (updated in-place)
    //   pos       — current position in sequence (0-indexed)
    void forward(int token_id, float* d_logits, KVCache& kv_cache, int pos);

    // Prefill: process prompt tokens sequentially, populate KV cache
    // After prefill, kv_cache.current_pos == n_tokens
    void prefill(const std::vector<int>& token_ids, KVCache& kv_cache);

    const TransformerConfig& config() const { return cfg_; }

    // cuBLAS handle exposed for speculative decoder (draft model shares handle)
    cublasHandle_t cublas_handle() const { return cublas_handle_; }

private:
    // Allocate all weight tensors on device
    void allocWeights();

    // Free all weight tensors
    void freeWeights();

    // Quantize FP16 weights → INT8 (called after initRandom or loadWeights)
    void quantizeWeightsINT8();

    // Helper: GEMM  C = alpha * A * B^T + beta * C
    // A: [m, k]  B: [n, k]  C: [m, n]  (column-major ≡ transposed row-major)
    // Used for single-token forward: m=1 always
    void gemm_fp16(const half* d_A, const half* d_B, half* d_C,
                   int m, int n, int k,
                   float alpha = 1.0f, float beta = 0.0f);

    // Single-layer forward (called inside forward())
    void forward_layer(int layer_idx, KVCache& kv_cache, int pos);

    TransformerConfig  cfg_;
    ModelWeights       weights_;
    cublasHandle_t     cublas_handle_;

    // Activation buffers (reused every forward call — single-token, no batching)
    half*  d_residual_;      // [d_model]
    half*  d_normed_;        // [d_model]  — post-RMSNorm
    half*  d_q_;             // [n_heads * head_dim]
    half*  d_k_;             // [n_kv_heads * head_dim]
    half*  d_v_;             // [n_kv_heads * head_dim]
    half*  d_attn_out_;      // [n_heads * head_dim]  — weighted sum of V
    half*  d_proj_out_;      // [d_model]  — after Wo projection
    half*  d_ffn_gate_;      // [ff_dim]
    half*  d_ffn_up_;        // [ff_dim]
    half*  d_ffn_hidden_;    // [ff_dim]   — SwiGLU(gate, up)
    half*  d_ffn_out_;       // [d_model]  — after down projection
    half*  d_logits_fp16_;   // [vocab_size] — before FP32 cast

    float* d_attn_scores_;   // [n_heads, max_seq_len]  — QK^T scores (float)
    curandState* d_rng_;     // [1] — for top-k sampling
};

// ============================================================================
// Convenience: allocate FP32 logit buffer on device
// ============================================================================
inline float* allocLogitBuffer(int vocab_size) {
    return cudaMallocDevice<float>(vocab_size);
}

#endif // TRANSFORMER_H
