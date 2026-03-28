#ifndef DECODER_H
#define DECODER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

struct DecoderConfig {
    int d_model = 512;
    int n_heads = 8;
    int n_layers = 4;
    int ffn_dim = 2048;
    int vocab_size = 52;      // CharTokenizer vocab
    int max_seq_len = 448;    // Max text tokens
};

struct DecoderLayerWeights {
    // Masked self-attention
    float* d_sa_wq;  float* d_sa_wk;  float* d_sa_wv;  float* d_sa_wo;
    float* d_sa_bq;  float* d_sa_bk;  float* d_sa_bv;  float* d_sa_bo;
    float* d_ln1_gamma;  float* d_ln1_beta;

    // Cross-attention
    float* d_ca_wq;  float* d_ca_wk;  float* d_ca_wv;  float* d_ca_wo;
    float* d_ca_bq;  float* d_ca_bk;  float* d_ca_bv;  float* d_ca_bo;
    float* d_ln2_gamma;  float* d_ln2_beta;

    // FFN
    float* d_w1;  float* d_b1;
    float* d_w2;  float* d_b2;
    float* d_ln3_gamma;  float* d_ln3_beta;
};

class WhisperDecoder {
public:
    WhisperDecoder(const DecoderConfig& config);
    ~WhisperDecoder();

    // Forward pass: d_tokens[num_tokens] + d_encoder_out[enc_len, d_model]
    // -> d_logits[num_tokens, vocab_size]
    void forward(const int* d_tokens, int num_tokens,
                 const float* d_encoder_out, int enc_len,
                 float* d_logits);

    // Forward single step (for autoregressive decoding with KV cache)
    // d_token: single token, step: current decoding step
    // d_logits: [1, vocab_size]
    void forward_step(int token, int step,
                      const float* d_encoder_out, int enc_len,
                      float* d_logits);

    // Reset KV cache (call before new sequence)
    void reset_kv_cache();

    void init_random_weights();

    DecoderConfig config;

private:
    cublasHandle_t cublas_;

    // Token embedding
    float* d_token_embed;    // [vocab_size, d_model]
    float* d_pe;             // [max_seq_len, d_model]

    // Output projection
    float* d_out_proj;       // [d_model, vocab_size]
    float* d_out_bias;       // [vocab_size]

    // Final LayerNorm
    float* d_ln_final_gamma;
    float* d_ln_final_beta;

    // Layers
    DecoderLayerWeights* layers_;

    // KV Cache for self-attention
    float** d_sa_k_cache;    // [n_layers][max_seq_len * d_model]
    float** d_sa_v_cache;    // [n_layers][max_seq_len * d_model]

    // KV Cache for cross-attention (computed once per audio)
    float** d_ca_k_cache;
    float** d_ca_v_cache;
    bool cross_attn_cached_;
    int cached_enc_len_;

    // Workspace
    float* d_embed_out;
    float* d_ln_out;
    float* d_q;  float* d_k;  float* d_v;
    float* d_q_heads;  float* d_k_heads;  float* d_v_heads;
    float* d_attn_scores;
    float* d_attn_out;
    float* d_attn_proj;
    float* d_residual;
    float* d_ffn_mid;
    float* d_ffn_out;
    float* d_step_buf;   // Buffer for single-step decoding

    void allocate_layer_weights(DecoderLayerWeights& layer);
    void free_layer_weights(DecoderLayerWeights& layer);

    void masked_self_attention(const float* d_input, float* d_output,
                               DecoderLayerWeights& layer, int layer_idx,
                               int seq_len, bool use_cache, int step);

    void cross_attention(const float* d_input, float* d_output,
                         DecoderLayerWeights& layer, int layer_idx,
                         const float* d_encoder_out, int enc_len,
                         int seq_len);

    void ffn(const float* d_input, float* d_output,
             DecoderLayerWeights& layer, int seq_len);
};

#endif // DECODER_H
