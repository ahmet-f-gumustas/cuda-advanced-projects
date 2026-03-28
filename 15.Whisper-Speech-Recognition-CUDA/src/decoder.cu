#include "decoder.h"
#include "whisper_kernels.cuh"
#include "cuda_utils.h"
#include <cstdlib>
#include <cmath>
#include <vector>

WhisperDecoder::WhisperDecoder(const DecoderConfig& cfg) : config(cfg),
    cross_attn_cached_(false), cached_enc_len_(0) {
    CUBLAS_CHECK(cublasCreate(&cublas_));

    int d = cfg.d_model;
    int max_seq = cfg.max_seq_len;
    int heads = cfg.n_heads;

    // Embeddings
    CUDA_CHECK(cudaMalloc(&d_token_embed, cfg.vocab_size * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pe, max_seq * d * sizeof(float)));
    launch_sinusoidal_pe(d_pe, max_seq, d);

    // Output projection
    CUDA_CHECK(cudaMalloc(&d_out_proj, d * cfg.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_bias, cfg.vocab_size * sizeof(float)));

    // Final LN
    CUDA_CHECK(cudaMalloc(&d_ln_final_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_final_beta, d * sizeof(float)));

    // Layers
    layers_ = new DecoderLayerWeights[cfg.n_layers];
    for (int i = 0; i < cfg.n_layers; i++) {
        allocate_layer_weights(layers_[i]);
    }

    // KV caches
    d_sa_k_cache = new float*[cfg.n_layers];
    d_sa_v_cache = new float*[cfg.n_layers];
    d_ca_k_cache = new float*[cfg.n_layers];
    d_ca_v_cache = new float*[cfg.n_layers];
    // Max encoder length for cross-attention cache
    int max_enc = 1500; // ~30s audio
    for (int i = 0; i < cfg.n_layers; i++) {
        CUDA_CHECK(cudaMalloc(&d_sa_k_cache[i], max_seq * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sa_v_cache[i], max_seq * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ca_k_cache[i], max_enc * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ca_v_cache[i], max_enc * d * sizeof(float)));
    }

    // Workspace
    CUDA_CHECK(cudaMalloc(&d_embed_out, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_out, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_heads, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_heads, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_heads, max_seq * d * sizeof(float)));
    int max_attn = heads * max_seq * max_seq;
    int max_cross_attn = heads * max_seq * max_enc;
    int max_scores = (max_attn > max_cross_attn) ? max_attn : max_cross_attn;
    CUDA_CHECK(cudaMalloc(&d_attn_scores, max_scores * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_proj, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_mid, max_seq * cfg.ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_step_buf, d * sizeof(float)));
}

WhisperDecoder::~WhisperDecoder() {
    cudaFree(d_token_embed); cudaFree(d_pe);
    cudaFree(d_out_proj); cudaFree(d_out_bias);
    cudaFree(d_ln_final_gamma); cudaFree(d_ln_final_beta);

    for (int i = 0; i < config.n_layers; i++) {
        free_layer_weights(layers_[i]);
        cudaFree(d_sa_k_cache[i]); cudaFree(d_sa_v_cache[i]);
        cudaFree(d_ca_k_cache[i]); cudaFree(d_ca_v_cache[i]);
    }
    delete[] layers_;
    delete[] d_sa_k_cache; delete[] d_sa_v_cache;
    delete[] d_ca_k_cache; delete[] d_ca_v_cache;

    cudaFree(d_embed_out); cudaFree(d_ln_out);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_q_heads); cudaFree(d_k_heads); cudaFree(d_v_heads);
    cudaFree(d_attn_scores); cudaFree(d_attn_out); cudaFree(d_attn_proj);
    cudaFree(d_residual); cudaFree(d_ffn_mid); cudaFree(d_ffn_out);
    cudaFree(d_step_buf);

    cublasDestroy(cublas_);
}

void WhisperDecoder::allocate_layer_weights(DecoderLayerWeights& l) {
    int d = config.d_model;
    int ff = config.ffn_dim;
    // Self-attention
    CUDA_CHECK(cudaMalloc(&l.d_sa_wq, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_wk, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_wv, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_wo, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_bq, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_bk, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_bv, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_sa_bo, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln1_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln1_beta, d * sizeof(float)));
    // Cross-attention
    CUDA_CHECK(cudaMalloc(&l.d_ca_wq, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_wk, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_wv, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_wo, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_bq, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_bk, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_bv, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ca_bo, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln2_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln2_beta, d * sizeof(float)));
    // FFN
    CUDA_CHECK(cudaMalloc(&l.d_w1, d * ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_b1, ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_w2, ff * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_b2, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln3_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln3_beta, d * sizeof(float)));
}

void WhisperDecoder::free_layer_weights(DecoderLayerWeights& l) {
    cudaFree(l.d_sa_wq); cudaFree(l.d_sa_wk); cudaFree(l.d_sa_wv); cudaFree(l.d_sa_wo);
    cudaFree(l.d_sa_bq); cudaFree(l.d_sa_bk); cudaFree(l.d_sa_bv); cudaFree(l.d_sa_bo);
    cudaFree(l.d_ln1_gamma); cudaFree(l.d_ln1_beta);
    cudaFree(l.d_ca_wq); cudaFree(l.d_ca_wk); cudaFree(l.d_ca_wv); cudaFree(l.d_ca_wo);
    cudaFree(l.d_ca_bq); cudaFree(l.d_ca_bk); cudaFree(l.d_ca_bv); cudaFree(l.d_ca_bo);
    cudaFree(l.d_ln2_gamma); cudaFree(l.d_ln2_beta);
    cudaFree(l.d_w1); cudaFree(l.d_b1); cudaFree(l.d_w2); cudaFree(l.d_b2);
    cudaFree(l.d_ln3_gamma); cudaFree(l.d_ln3_beta);
}

void WhisperDecoder::init_random_weights() {
    srand(123);
    auto fill_random = [](float* d_ptr, int n, float scale) {
        std::vector<float> h(n);
        for (int i = 0; i < n; i++) {
            h[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };
    auto fill_ones = [](float* d_ptr, int n) {
        std::vector<float> h(n, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };
    auto fill_zeros = [](float* d_ptr, int n) {
        CUDA_CHECK(cudaMemset(d_ptr, 0, n * sizeof(float)));
    };

    int d = config.d_model;
    float embed_scale = sqrtf(2.0f / (config.vocab_size + d));
    float attn_scale = sqrtf(2.0f / (d + d));
    float ffn_scale = sqrtf(2.0f / (d + config.ffn_dim));
    float proj_scale = sqrtf(2.0f / (d + config.vocab_size));

    fill_random(d_token_embed, config.vocab_size * d, embed_scale);
    fill_random(d_out_proj, d * config.vocab_size, proj_scale);
    fill_zeros(d_out_bias, config.vocab_size);
    fill_ones(d_ln_final_gamma, d);
    fill_zeros(d_ln_final_beta, d);

    for (int i = 0; i < config.n_layers; i++) {
        auto& l = layers_[i];
        fill_random(l.d_sa_wq, d * d, attn_scale);
        fill_random(l.d_sa_wk, d * d, attn_scale);
        fill_random(l.d_sa_wv, d * d, attn_scale);
        fill_random(l.d_sa_wo, d * d, attn_scale);
        fill_zeros(l.d_sa_bq, d); fill_zeros(l.d_sa_bk, d);
        fill_zeros(l.d_sa_bv, d); fill_zeros(l.d_sa_bo, d);
        fill_ones(l.d_ln1_gamma, d); fill_zeros(l.d_ln1_beta, d);

        fill_random(l.d_ca_wq, d * d, attn_scale);
        fill_random(l.d_ca_wk, d * d, attn_scale);
        fill_random(l.d_ca_wv, d * d, attn_scale);
        fill_random(l.d_ca_wo, d * d, attn_scale);
        fill_zeros(l.d_ca_bq, d); fill_zeros(l.d_ca_bk, d);
        fill_zeros(l.d_ca_bv, d); fill_zeros(l.d_ca_bo, d);
        fill_ones(l.d_ln2_gamma, d); fill_zeros(l.d_ln2_beta, d);

        fill_random(l.d_w1, d * config.ffn_dim, ffn_scale);
        fill_zeros(l.d_b1, config.ffn_dim);
        fill_random(l.d_w2, config.ffn_dim * d, ffn_scale);
        fill_zeros(l.d_b2, d);
        fill_ones(l.d_ln3_gamma, d); fill_zeros(l.d_ln3_beta, d);
    }
}

void WhisperDecoder::reset_kv_cache() {
    cross_attn_cached_ = false;
    cached_enc_len_ = 0;
    for (int i = 0; i < config.n_layers; i++) {
        CUDA_CHECK(cudaMemset(d_sa_k_cache[i], 0,
                               config.max_seq_len * config.d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sa_v_cache[i], 0,
                               config.max_seq_len * config.d_model * sizeof(float)));
    }
}

void WhisperDecoder::masked_self_attention(const float* d_input, float* d_output,
                                            DecoderLayerWeights& layer, int layer_idx,
                                            int seq_len, bool use_cache, int step) {
    int d = config.d_model;
    int heads = config.n_heads;
    int head_dim = d / heads;
    float alpha = 1.0f, beta_val = 0.0f;

    // LayerNorm
    launch_layer_norm(d_input, layer.d_ln1_gamma, layer.d_ln1_beta,
                      d_ln_out, seq_len, d);

    // Q projection
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_sa_wq, d, d_ln_out, d,
                              &beta_val, d_q, d));
    launch_add_bias(d_q, layer.d_sa_bq, seq_len, d);

    int k_len;
    if (use_cache && step >= 0) {
        // Single-step: compute K, V for current token and append to cache
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  d, 1, d, &alpha,
                                  layer.d_sa_wk, d, d_ln_out, d,
                                  &beta_val, d_k, d));
        launch_add_bias(d_k, layer.d_sa_bk, 1, d);

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  d, 1, d, &alpha,
                                  layer.d_sa_wv, d, d_ln_out, d,
                                  &beta_val, d_v, d));
        launch_add_bias(d_v, layer.d_sa_bv, 1, d);

        // Copy to cache at position step
        CUDA_CHECK(cudaMemcpy(d_sa_k_cache[layer_idx] + step * d,
                               d_k, d * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_sa_v_cache[layer_idx] + step * d,
                               d_v, d * sizeof(float), cudaMemcpyDeviceToDevice));

        k_len = step + 1;
        // Use cached K, V
        launch_transpose_0213(d_q, d_q_heads, 1, heads, head_dim);
        launch_transpose_0213(d_sa_k_cache[layer_idx], d_k_heads, k_len, heads, head_dim);
        launch_transpose_0213(d_sa_v_cache[layer_idx], d_v_heads, k_len, heads, head_dim);
    } else {
        // Full sequence
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  d, seq_len, d, &alpha,
                                  layer.d_sa_wk, d, d_ln_out, d,
                                  &beta_val, d_k, d));
        launch_add_bias(d_k, layer.d_sa_bk, seq_len, d);

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  d, seq_len, d, &alpha,
                                  layer.d_sa_wv, d, d_ln_out, d,
                                  &beta_val, d_v, d));
        launch_add_bias(d_v, layer.d_sa_bv, seq_len, d);

        k_len = seq_len;
        launch_transpose_0213(d_q, d_q_heads, seq_len, heads, head_dim);
        launch_transpose_0213(d_k, d_k_heads, seq_len, heads, head_dim);
        launch_transpose_0213(d_v, d_v_heads, seq_len, heads, head_dim);
    }

    // Batched GEMM: scores = Q @ K^T
    int q_len = use_cache ? 1 : seq_len;
    float scale = 1.0f / sqrtf((float)head_dim);
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                            k_len, q_len, head_dim,
                                            &scale,
                                            d_k_heads, head_dim, (long long)k_len * head_dim,
                                            d_q_heads, head_dim, (long long)q_len * head_dim,
                                            &beta_val,
                                            d_attn_scores, k_len, (long long)q_len * k_len,
                                            heads));

    // Causal masking + softmax
    launch_masked_softmax(d_attn_scores, heads * q_len, k_len);

    // context = attn @ V
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                            head_dim, q_len, k_len,
                                            &alpha,
                                            d_v_heads, head_dim, (long long)k_len * head_dim,
                                            d_attn_scores, k_len, (long long)q_len * k_len,
                                            &beta_val,
                                            d_attn_out, head_dim, (long long)q_len * head_dim,
                                            heads));

    // Transpose back
    launch_transpose_1023(d_attn_out, d_attn_proj, heads, q_len, head_dim);

    // Output projection
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, q_len, d, &alpha,
                              layer.d_sa_wo, d, d_attn_proj, d,
                              &beta_val, d_output, d));
    launch_add_bias(d_output, layer.d_sa_bo, q_len, d);

    // Residual
    launch_add_residual(d_output, d_input, d_output, q_len * d);
}

void WhisperDecoder::cross_attention(const float* d_input, float* d_output,
                                      DecoderLayerWeights& layer, int layer_idx,
                                      const float* d_encoder_out, int enc_len,
                                      int seq_len) {
    int d = config.d_model;
    int heads = config.n_heads;
    int head_dim = d / heads;
    float alpha = 1.0f, beta_val = 0.0f;

    // LayerNorm
    launch_layer_norm(d_input, layer.d_ln2_gamma, layer.d_ln2_beta,
                      d_ln_out, seq_len, d);

    // Q from decoder
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_ca_wq, d, d_ln_out, d,
                              &beta_val, d_q, d));
    launch_add_bias(d_q, layer.d_ca_bq, seq_len, d);

    // K, V from encoder (cache if not already done)
    if (!cross_attn_cached_ || cached_enc_len_ != enc_len) {
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  d, enc_len, d, &alpha,
                                  layer.d_ca_wk, d, d_encoder_out, d,
                                  &beta_val, d_ca_k_cache[layer_idx], d));
        launch_add_bias(d_ca_k_cache[layer_idx], layer.d_ca_bk, enc_len, d);

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  d, enc_len, d, &alpha,
                                  layer.d_ca_wv, d, d_encoder_out, d,
                                  &beta_val, d_ca_v_cache[layer_idx], d));
        launch_add_bias(d_ca_v_cache[layer_idx], layer.d_ca_bv, enc_len, d);
    }

    // Multi-head attention: Q from decoder, K/V from encoder
    launch_transpose_0213(d_q, d_q_heads, seq_len, heads, head_dim);
    launch_transpose_0213(d_ca_k_cache[layer_idx], d_k_heads, enc_len, heads, head_dim);
    launch_transpose_0213(d_ca_v_cache[layer_idx], d_v_heads, enc_len, heads, head_dim);

    // scores = Q @ K^T (no causal mask for cross-attention)
    float scale = 1.0f / sqrtf((float)head_dim);
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                            enc_len, seq_len, head_dim,
                                            &scale,
                                            d_k_heads, head_dim, (long long)enc_len * head_dim,
                                            d_q_heads, head_dim, (long long)seq_len * head_dim,
                                            &beta_val,
                                            d_attn_scores, enc_len, (long long)seq_len * enc_len,
                                            heads));

    // Normal softmax (no causal mask)
    launch_softmax(d_attn_scores, heads * seq_len, enc_len);

    // context = attn @ V
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                            head_dim, seq_len, enc_len,
                                            &alpha,
                                            d_v_heads, head_dim, (long long)enc_len * head_dim,
                                            d_attn_scores, enc_len, (long long)seq_len * enc_len,
                                            &beta_val,
                                            d_attn_out, head_dim, (long long)seq_len * head_dim,
                                            heads));

    launch_transpose_1023(d_attn_out, d_attn_proj, heads, seq_len, head_dim);

    // Output projection
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_ca_wo, d, d_attn_proj, d,
                              &beta_val, d_output, d));
    launch_add_bias(d_output, layer.d_ca_bo, seq_len, d);

    // Residual
    launch_add_residual(d_output, d_input, d_output, seq_len * d);
}

void WhisperDecoder::ffn(const float* d_input, float* d_output,
                          DecoderLayerWeights& layer, int seq_len) {
    int d = config.d_model;
    int ff = config.ffn_dim;
    float alpha = 1.0f, beta_val = 0.0f;

    launch_layer_norm(d_input, layer.d_ln3_gamma, layer.d_ln3_beta,
                      d_ln_out, seq_len, d);

    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              ff, seq_len, d, &alpha,
                              layer.d_w1, ff, d_ln_out, d,
                              &beta_val, d_ffn_mid, ff));
    launch_add_bias(d_ffn_mid, layer.d_b1, seq_len, ff);
    launch_gelu(d_ffn_mid, seq_len * ff);

    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, ff, &alpha,
                              layer.d_w2, d, d_ffn_mid, ff,
                              &beta_val, d_output, d));
    launch_add_bias(d_output, layer.d_b2, seq_len, d);

    launch_add_residual(d_output, d_input, d_output, seq_len * d);
}

void WhisperDecoder::forward(const int* d_tokens, int num_tokens,
                              const float* d_encoder_out, int enc_len,
                              float* d_logits) {
    int d = config.d_model;
    float alpha = 1.0f, beta_val = 0.0f;

    // Token embedding + positional encoding
    launch_embedding_lookup(d_tokens, d_token_embed, d_embed_out, num_tokens, d);
    launch_add_pe(d_embed_out, d_pe, num_tokens, d);

    float* current = d_embed_out;

    for (int i = 0; i < config.n_layers; i++) {
        // Masked self-attention
        launch_copy(current, d_residual, num_tokens * d);
        masked_self_attention(d_residual, current, layers_[i], i, num_tokens, false, -1);

        // Cross-attention
        launch_copy(current, d_residual, num_tokens * d);
        cross_attention(d_residual, current, layers_[i], i,
                        d_encoder_out, enc_len, num_tokens);

        // FFN
        launch_copy(current, d_residual, num_tokens * d);
        ffn(d_residual, current, layers_[i], num_tokens);
    }

    // Mark cross-attention as cached
    cross_attn_cached_ = true;
    cached_enc_len_ = enc_len;

    // Final LayerNorm
    launch_layer_norm(current, d_ln_final_gamma, d_ln_final_beta,
                      current, num_tokens, d);

    // Output projection: [num_tokens, d] @ [d, vocab] -> [num_tokens, vocab]
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              config.vocab_size, num_tokens, d, &alpha,
                              d_out_proj, config.vocab_size, current, d,
                              &beta_val, d_logits, config.vocab_size));
    launch_add_bias(d_logits, d_out_bias, num_tokens, config.vocab_size);
}

void WhisperDecoder::forward_step(int token, int step,
                                   const float* d_encoder_out, int enc_len,
                                   float* d_logits) {
    int d = config.d_model;
    float alpha = 1.0f, beta_val = 0.0f;

    // Single token embedding
    int h_token = token;
    int* d_token_ptr;
    CUDA_CHECK(cudaMalloc(&d_token_ptr, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token_ptr, &h_token, sizeof(int), cudaMemcpyHostToDevice));

    launch_embedding_lookup(d_token_ptr, d_token_embed, d_step_buf, 1, d);
    // Add PE for this position
    launch_add_pe(d_step_buf, d_pe + step * d, 1, d);

    float* current = d_step_buf;

    for (int i = 0; i < config.n_layers; i++) {
        launch_copy(current, d_residual, d);
        masked_self_attention(d_residual, current, layers_[i], i, 1, true, step);

        launch_copy(current, d_residual, d);
        cross_attention(d_residual, current, layers_[i], i,
                        d_encoder_out, enc_len, 1);

        launch_copy(current, d_residual, d);
        ffn(d_residual, current, layers_[i], 1);
    }

    cross_attn_cached_ = true;
    cached_enc_len_ = enc_len;

    launch_layer_norm(current, d_ln_final_gamma, d_ln_final_beta, current, 1, d);

    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              config.vocab_size, 1, d, &alpha,
                              d_out_proj, config.vocab_size, current, d,
                              &beta_val, d_logits, config.vocab_size));
    launch_add_bias(d_logits, d_out_bias, 1, config.vocab_size);

    cudaFree(d_token_ptr);
}
