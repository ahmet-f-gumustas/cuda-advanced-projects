#include "encoder.h"
#include "whisper_kernels.cuh"
#include "cuda_utils.h"
#include <cstdlib>
#include <cmath>
#include <vector>

WhisperEncoder::WhisperEncoder(const EncoderConfig& cfg) : config(cfg) {
    CUBLAS_CHECK(cublasCreate(&cublas_));

    // Conv stem
    int conv1_size = cfg.d_model * cfg.n_mels * cfg.conv1_kernel;
    int conv2_size = cfg.d_model * cfg.d_model * cfg.conv2_kernel;
    CUDA_CHECK(cudaMalloc(&d_conv1_weight, conv1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias, cfg.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_weight, conv2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias, cfg.d_model * sizeof(float)));

    // Positional encoding
    CUDA_CHECK(cudaMalloc(&d_pe, cfg.max_seq_len * cfg.d_model * sizeof(float)));
    launch_sinusoidal_pe(d_pe, cfg.max_seq_len, cfg.d_model);

    // Final LayerNorm
    CUDA_CHECK(cudaMalloc(&d_ln_final_gamma, cfg.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_final_beta, cfg.d_model * sizeof(float)));

    // Layers
    layers_ = new EncoderLayerWeights[cfg.n_layers];
    for (int i = 0; i < cfg.n_layers; i++) {
        allocate_layer_weights(layers_[i]);
    }

    allocate_workspace(cfg.max_seq_len);
}

WhisperEncoder::~WhisperEncoder() {
    cudaFree(d_conv1_weight); cudaFree(d_conv1_bias);
    cudaFree(d_conv2_weight); cudaFree(d_conv2_bias);
    cudaFree(d_pe);
    cudaFree(d_ln_final_gamma); cudaFree(d_ln_final_beta);

    for (int i = 0; i < config.n_layers; i++) {
        free_layer_weights(layers_[i]);
    }
    delete[] layers_;

    cudaFree(d_conv1_out); cudaFree(d_conv2_out);
    cudaFree(d_ln_out); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_q_heads); cudaFree(d_k_heads); cudaFree(d_v_heads);
    cudaFree(d_attn_scores); cudaFree(d_attn_out); cudaFree(d_attn_proj);
    cudaFree(d_residual); cudaFree(d_ffn_mid); cudaFree(d_ffn_out);

    cublasDestroy(cublas_);
}

void WhisperEncoder::allocate_layer_weights(EncoderLayerWeights& l) {
    int d = config.d_model;
    int ff = config.ffn_dim;
    CUDA_CHECK(cudaMalloc(&l.d_wq, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_wk, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_wv, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_wo, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_bq, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_bk, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_bv, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_bo, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln1_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln1_beta, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_w1, d * ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_b1, ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_w2, ff * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_b2, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln2_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_ln2_beta, d * sizeof(float)));
}

void WhisperEncoder::free_layer_weights(EncoderLayerWeights& l) {
    cudaFree(l.d_wq); cudaFree(l.d_wk); cudaFree(l.d_wv); cudaFree(l.d_wo);
    cudaFree(l.d_bq); cudaFree(l.d_bk); cudaFree(l.d_bv); cudaFree(l.d_bo);
    cudaFree(l.d_ln1_gamma); cudaFree(l.d_ln1_beta);
    cudaFree(l.d_w1); cudaFree(l.d_b1); cudaFree(l.d_w2); cudaFree(l.d_b2);
    cudaFree(l.d_ln2_gamma); cudaFree(l.d_ln2_beta);
}

void WhisperEncoder::allocate_workspace(int max_seq) {
    int d = config.d_model;
    int ff = config.ffn_dim;
    int heads = config.n_heads;

    CUDA_CHECK(cudaMalloc(&d_conv1_out, d * max_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, d * max_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_out, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_heads, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_heads, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_heads, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_scores, heads * max_seq * max_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_proj, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_mid, max_seq * ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out, max_seq * d * sizeof(float)));
}

int WhisperEncoder::get_output_length(int num_frames) const {
    // Conv1: same length (stride=1, padding=1)
    int len1 = num_frames;
    // Conv2: downsample by stride
    int len2 = (len1 + 2 * 1 - config.conv2_kernel) / config.conv2_stride + 1;
    return len2;
}

void WhisperEncoder::init_random_weights() {
    srand(42);
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
    float conv1_scale = sqrtf(2.0f / (config.n_mels * config.conv1_kernel));
    float conv2_scale = sqrtf(2.0f / (d * config.conv2_kernel));
    float attn_scale = sqrtf(2.0f / (d + d));
    float ffn_scale = sqrtf(2.0f / (d + config.ffn_dim));

    fill_random(d_conv1_weight, d * config.n_mels * config.conv1_kernel, conv1_scale);
    fill_zeros(d_conv1_bias, d);
    fill_random(d_conv2_weight, d * d * config.conv2_kernel, conv2_scale);
    fill_zeros(d_conv2_bias, d);
    fill_ones(d_ln_final_gamma, d);
    fill_zeros(d_ln_final_beta, d);

    for (int i = 0; i < config.n_layers; i++) {
        auto& l = layers_[i];
        fill_random(l.d_wq, d * d, attn_scale);
        fill_random(l.d_wk, d * d, attn_scale);
        fill_random(l.d_wv, d * d, attn_scale);
        fill_random(l.d_wo, d * d, attn_scale);
        fill_zeros(l.d_bq, d); fill_zeros(l.d_bk, d);
        fill_zeros(l.d_bv, d); fill_zeros(l.d_bo, d);
        fill_ones(l.d_ln1_gamma, d); fill_zeros(l.d_ln1_beta, d);
        fill_random(l.d_w1, d * config.ffn_dim, ffn_scale);
        fill_zeros(l.d_b1, config.ffn_dim);
        fill_random(l.d_w2, config.ffn_dim * d, ffn_scale);
        fill_zeros(l.d_b2, d);
        fill_ones(l.d_ln2_gamma, d); fill_zeros(l.d_ln2_beta, d);
    }
}

void WhisperEncoder::self_attention(const float* d_input, float* d_output,
                                     EncoderLayerWeights& layer, int seq_len) {
    int d = config.d_model;
    int heads = config.n_heads;
    int head_dim = d / heads;
    float alpha = 1.0f, beta_val = 0.0f;

    // LayerNorm
    launch_layer_norm(d_input, layer.d_ln1_gamma, layer.d_ln1_beta,
                      d_ln_out, seq_len, d);

    // Q, K, V projections: [seq_len, d] @ [d, d] -> [seq_len, d]
    // cuBLAS uses column-major, so we compute: C = B^T * A^T, then C^T = A * B
    // For row-major GEMM: C[m,n] = A[m,k] * B[k,n]
    // cuBLAS call: cublasSgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n)
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_wq, d, d_ln_out, d,
                              &beta_val, d_q, d));
    launch_add_bias(d_q, layer.d_bq, seq_len, d);

    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_wk, d, d_ln_out, d,
                              &beta_val, d_k, d));
    launch_add_bias(d_k, layer.d_bk, seq_len, d);

    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_wv, d, d_ln_out, d,
                              &beta_val, d_v, d));
    launch_add_bias(d_v, layer.d_bv, seq_len, d);

    // Transpose to multi-head format: [seq, heads, head_dim] -> [heads, seq, head_dim]
    launch_transpose_0213(d_q, d_q_heads, seq_len, heads, head_dim);
    launch_transpose_0213(d_k, d_k_heads, seq_len, heads, head_dim);
    launch_transpose_0213(d_v, d_v_heads, seq_len, heads, head_dim);

    // Batched GEMM: scores = Q @ K^T for each head
    // scores[h]: [seq, head_dim] @ [head_dim, seq] -> [seq, seq]
    float scale = 1.0f / sqrtf((float)head_dim);
    long long stride_q = seq_len * head_dim;
    long long stride_k = seq_len * head_dim;
    long long stride_s = seq_len * seq_len;
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                            seq_len, seq_len, head_dim,
                                            &scale,
                                            d_k_heads, head_dim, stride_k,
                                            d_q_heads, head_dim, stride_q,
                                            &beta_val,
                                            d_attn_scores, seq_len, stride_s,
                                            heads));

    // Softmax over attention scores
    launch_softmax(d_attn_scores, heads * seq_len, seq_len);

    // Batched GEMM: context = attn @ V for each head
    // context[h]: [seq, seq] @ [seq, head_dim] -> [seq, head_dim]
    long long stride_v = seq_len * head_dim;
    long long stride_o = seq_len * head_dim;
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                            head_dim, seq_len, seq_len,
                                            &alpha,
                                            d_v_heads, head_dim, stride_v,
                                            d_attn_scores, seq_len, stride_s,
                                            &beta_val,
                                            d_attn_out, head_dim, stride_o,
                                            heads));

    // Transpose back: [heads, seq, head_dim] -> [seq, heads, head_dim] = [seq, d_model]
    launch_transpose_1023(d_attn_out, d_attn_proj, heads, seq_len, head_dim);

    // Output projection
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, d, &alpha,
                              layer.d_wo, d, d_attn_proj, d,
                              &beta_val, d_output, d));
    launch_add_bias(d_output, layer.d_bo, seq_len, d);

    // Residual connection
    launch_add_residual(d_output, d_input, d_output, seq_len * d);
}

void WhisperEncoder::ffn(const float* d_input, float* d_output,
                          EncoderLayerWeights& layer, int seq_len) {
    int d = config.d_model;
    int ff = config.ffn_dim;
    float alpha = 1.0f, beta_val = 0.0f;

    // LayerNorm
    launch_layer_norm(d_input, layer.d_ln2_gamma, layer.d_ln2_beta,
                      d_ln_out, seq_len, d);

    // FFN layer 1: [seq, d] @ [d, ff] -> [seq, ff]
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              ff, seq_len, d, &alpha,
                              layer.d_w1, ff, d_ln_out, d,
                              &beta_val, d_ffn_mid, ff));
    launch_add_bias(d_ffn_mid, layer.d_b1, seq_len, ff);
    launch_gelu(d_ffn_mid, seq_len * ff);

    // FFN layer 2: [seq, ff] @ [ff, d] -> [seq, d]
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                              d, seq_len, ff, &alpha,
                              layer.d_w2, d, d_ffn_mid, ff,
                              &beta_val, d_output, d));
    launch_add_bias(d_output, layer.d_b2, seq_len, d);

    // Residual
    launch_add_residual(d_output, d_input, d_output, seq_len * d);
}

void WhisperEncoder::forward(const float* d_mel, int num_frames,
                              float* d_output, int& out_len) {
    int d = config.d_model;

    // Conv1: [n_mels, num_frames] -> [d_model, num_frames]
    launch_conv1d(d_mel, d_conv1_weight, d_conv1_bias, d_conv1_out,
                  config.n_mels, d, num_frames, config.conv1_kernel, 1, 1);
    launch_gelu(d_conv1_out, d * num_frames);

    // Conv2: [d_model, num_frames] -> [d_model, out_len]
    out_len = get_output_length(num_frames);
    launch_conv1d(d_conv1_out, d_conv2_weight, d_conv2_bias, d_conv2_out,
                  d, d, num_frames, config.conv2_kernel, config.conv2_stride, 1);
    launch_gelu(d_conv2_out, d * out_len);

    // Transpose from [d_model, out_len] to [out_len, d_model] for transformer
    // We need a transpose kernel for this (channels, length) -> (length, channels)
    // Reuse transpose_0213 with n_heads=d_model, seq_len=1 won't work.
    // Instead, use a cublas transpose (geam)
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasSgeam(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                              d, out_len,
                              &alpha, d_conv2_out, out_len,
                              &beta_val, d_conv2_out, d,
                              d_output, d));

    // Add positional encoding
    launch_add_pe(d_output, d_pe, out_len, d);

    // Transformer encoder layers
    for (int i = 0; i < config.n_layers; i++) {
        // Copy to residual buffer for in-place operations
        launch_copy(d_output, d_residual, out_len * d);

        // Self-attention with residual
        self_attention(d_residual, d_output, layers_[i], out_len);

        // Copy for FFN
        launch_copy(d_output, d_residual, out_len * d);

        // FFN with residual
        ffn(d_residual, d_output, layers_[i], out_len);
    }

    // Final LayerNorm
    launch_layer_norm(d_output, d_ln_final_gamma, d_ln_final_beta,
                      d_output, out_len, d);

    CUDA_CHECK(cudaDeviceSynchronize());
}
