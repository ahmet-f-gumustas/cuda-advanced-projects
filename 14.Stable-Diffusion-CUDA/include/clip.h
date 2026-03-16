#ifndef CLIP_H
#define CLIP_H

#include "cuda_utils.h"

// ============================================================
// Simplified CLIP Text Encoder
// Transformer-based encoder: token embeddings -> context vectors
// ============================================================

struct CLIPConfig {
    int vocab_size = 256;       // Char-level (like project 13)
    int max_seq_len = 64;
    int d_model = 512;
    int n_layers = 4;
    int n_heads = 8;
    int ff_dim = 2048;          // 4x d_model
    float layer_norm_eps = 1e-5f;

    int head_dim() const { return d_model / n_heads; }
};

struct CLIPLayerWeights {
    // Self-attention
    half* d_ln1_gamma;      // [d_model]
    half* d_ln1_beta;       // [d_model]
    half* d_wq;             // [d_model, d_model]
    half* d_wk;             // [d_model, d_model]
    half* d_wv;             // [d_model, d_model]
    half* d_wo;             // [d_model, d_model]

    // FFN
    half* d_ln2_gamma;      // [d_model]
    half* d_ln2_beta;       // [d_model]
    half* d_ff1;            // [d_model, ff_dim]
    half* d_ff2;            // [ff_dim, d_model]
};

struct CLIPWeights {
    half* d_token_embed;        // [vocab_size, d_model]
    half* d_position_embed;     // [max_seq_len, d_model]
    std::vector<CLIPLayerWeights> layers;
    half* d_final_ln_gamma;     // [d_model]
    half* d_final_ln_beta;      // [d_model]
};

class CLIPEncoder {
public:
    CLIPEncoder(const CLIPConfig& cfg);
    ~CLIPEncoder();

    void initRandom(unsigned long long seed = 42);

    // Encode token IDs to context embeddings
    // token_ids: [seq_len] on host
    // d_output: [seq_len, d_model] on device (FP16)
    void encode(const std::vector<int>& token_ids, half* d_output);

    const CLIPConfig& config() const { return cfg_; }
    int contextDim() const { return cfg_.d_model; }

private:
    void allocWeights();
    void freeWeights();

    // cuBLAS GEMM: C[m,n] = A[m,k] * B[n,k]^T
    void gemm(const half* A, const half* B, half* C,
              int m, int n, int k, float alpha = 1.0f, float beta = 0.0f);

    void forwardLayer(int layer_idx, half* d_hidden, int seq_len);

    CLIPConfig cfg_;
    CLIPWeights weights_;
    cublasHandle_t cublas_handle_;

    // Activation buffers
    half* d_hidden_;        // [max_seq_len, d_model]
    half* d_normed_;        // [max_seq_len, d_model]
    half* d_q_;             // [max_seq_len, d_model]
    half* d_k_;             // [max_seq_len, d_model]
    half* d_v_;             // [max_seq_len, d_model]
    half* d_attn_out_;      // [max_seq_len, d_model]
    half* d_ff_hidden_;     // [max_seq_len, ff_dim]
    float* d_attn_scores_;  // [n_heads, max_seq_len, max_seq_len]
};

#endif // CLIP_H
