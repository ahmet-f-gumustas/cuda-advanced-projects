#include "clip.h"
#include "diffusion_kernels.cuh"

CLIPEncoder::CLIPEncoder(const CLIPConfig& cfg) : cfg_(cfg) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH));
    allocWeights();
}

CLIPEncoder::~CLIPEncoder() {
    freeWeights();
    cublasDestroy(cublas_handle_);
}

void CLIPEncoder::allocWeights() {
    int D = cfg_.d_model;
    int V = cfg_.vocab_size;
    int S = cfg_.max_seq_len;
    int F = cfg_.ff_dim;

    // Embeddings
    weights_.d_token_embed = cudaMallocDevice<half>(V * D);
    weights_.d_position_embed = cudaMallocDevice<half>(S * D);

    // Layers
    weights_.layers.resize(cfg_.n_layers);
    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& lw = weights_.layers[l];
        lw.d_ln1_gamma = cudaMallocDevice<half>(D);
        lw.d_ln1_beta = cudaMallocDevice<half>(D);
        lw.d_wq = cudaMallocDevice<half>(D * D);
        lw.d_wk = cudaMallocDevice<half>(D * D);
        lw.d_wv = cudaMallocDevice<half>(D * D);
        lw.d_wo = cudaMallocDevice<half>(D * D);
        lw.d_ln2_gamma = cudaMallocDevice<half>(D);
        lw.d_ln2_beta = cudaMallocDevice<half>(D);
        lw.d_ff1 = cudaMallocDevice<half>(D * F);
        lw.d_ff2 = cudaMallocDevice<half>(F * D);
    }

    // Final layer norm
    weights_.d_final_ln_gamma = cudaMallocDevice<half>(D);
    weights_.d_final_ln_beta = cudaMallocDevice<half>(D);

    // Activation buffers
    d_hidden_ = cudaMallocDevice<half>(S * D);
    d_normed_ = cudaMallocDevice<half>(S * D);
    d_q_ = cudaMallocDevice<half>(S * D);
    d_k_ = cudaMallocDevice<half>(S * D);
    d_v_ = cudaMallocDevice<half>(S * D);
    d_attn_out_ = cudaMallocDevice<half>(S * D);
    d_ff_hidden_ = cudaMallocDevice<half>(S * F);
    d_attn_scores_ = cudaMallocDevice<float>(cfg_.n_heads * S * S);
}

void CLIPEncoder::freeWeights() {
    cudaFree(weights_.d_token_embed);
    cudaFree(weights_.d_position_embed);
    for (auto& lw : weights_.layers) {
        cudaFree(lw.d_ln1_gamma); cudaFree(lw.d_ln1_beta);
        cudaFree(lw.d_wq); cudaFree(lw.d_wk); cudaFree(lw.d_wv); cudaFree(lw.d_wo);
        cudaFree(lw.d_ln2_gamma); cudaFree(lw.d_ln2_beta);
        cudaFree(lw.d_ff1); cudaFree(lw.d_ff2);
    }
    cudaFree(weights_.d_final_ln_gamma);
    cudaFree(weights_.d_final_ln_beta);
    cudaFree(d_hidden_); cudaFree(d_normed_);
    cudaFree(d_q_); cudaFree(d_k_); cudaFree(d_v_);
    cudaFree(d_attn_out_); cudaFree(d_ff_hidden_);
    cudaFree(d_attn_scores_);
}

void CLIPEncoder::initRandom(unsigned long long seed) {
    int D = cfg_.d_model;
    int F = cfg_.ff_dim;
    float embed_scale = 0.02f;
    float weight_scale = 0.02f;

    unsigned s = (unsigned)seed;
    init_random_fp16(weights_.d_token_embed, cfg_.vocab_size * D, embed_scale, s++);
    init_random_fp16(weights_.d_position_embed, cfg_.max_seq_len * D, embed_scale, s++);

    // Layer norm: gamma=1, beta=0
    std::vector<half> ones(D), zeros(D);
    for (int i = 0; i < D; i++) { ones[i] = __float2half(1.0f); zeros[i] = __float2half(0.0f); }

    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& lw = weights_.layers[l];
        cudaMemcpyH2D(lw.d_ln1_gamma, ones.data(), D);
        cudaMemcpyH2D(lw.d_ln1_beta, zeros.data(), D);
        init_random_fp16(lw.d_wq, D * D, weight_scale, s++);
        init_random_fp16(lw.d_wk, D * D, weight_scale, s++);
        init_random_fp16(lw.d_wv, D * D, weight_scale, s++);
        init_random_fp16(lw.d_wo, D * D, weight_scale, s++);
        cudaMemcpyH2D(lw.d_ln2_gamma, ones.data(), D);
        cudaMemcpyH2D(lw.d_ln2_beta, zeros.data(), D);
        init_random_fp16(lw.d_ff1, D * F, weight_scale, s++);
        init_random_fp16(lw.d_ff2, F * D, weight_scale, s++);
    }

    cudaMemcpyH2D(weights_.d_final_ln_gamma, ones.data(), D);
    cudaMemcpyH2D(weights_.d_final_ln_beta, zeros.data(), D);

    printf("[CLIP] Initialized with random weights (d=%d, L=%d, H=%d)\n",
           D, cfg_.n_layers, cfg_.n_heads);
}

void CLIPEncoder::gemm(const half* A, const half* B, half* C,
                       int m, int n, int k, float alpha, float beta) {
    // Row-major: C[m,n] = A[m,k] * B[n,k]^T
    // cuBLAS col-major: C = B * A^T -> cublasHgemm(N, T, n, m, k, B, n, A, k, C, n)
    half alpha_h = __float2half(alpha);
    half beta_h = __float2half(beta);
    CUBLAS_CHECK(cublasHgemm(cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha_h,
        B, k,
        A, k,
        &beta_h,
        C, n));
}

void CLIPEncoder::encode(const std::vector<int>& token_ids, half* d_output) {
    int seq_len = (int)token_ids.size();
    if (seq_len > cfg_.max_seq_len) seq_len = cfg_.max_seq_len;
    int D = cfg_.d_model;

    // Token embedding + positional embedding
    // Copy token embeddings to d_hidden_
    std::vector<half> h_hidden(seq_len * D);
    std::vector<half> h_token_embed(cfg_.vocab_size * D);
    std::vector<half> h_pos_embed(cfg_.max_seq_len * D);
    cudaMemcpyD2H(h_token_embed.data(), weights_.d_token_embed, cfg_.vocab_size * D);
    cudaMemcpyD2H(h_pos_embed.data(), weights_.d_position_embed, cfg_.max_seq_len * D);

    for (int t = 0; t < seq_len; t++) {
        int tok = token_ids[t];
        if (tok < 0 || tok >= cfg_.vocab_size) tok = 0;
        for (int d = 0; d < D; d++) {
            float val = __half2float(h_token_embed[tok * D + d])
                      + __half2float(h_pos_embed[t * D + d]);
            h_hidden[t * D + d] = __float2half(val);
        }
    }
    cudaMemcpyH2D(d_hidden_, h_hidden.data(), seq_len * D);

    // Forward through transformer layers
    for (int l = 0; l < cfg_.n_layers; l++) {
        forwardLayer(l, d_hidden_, seq_len);
    }

    // Final layer norm
    int threads = std::min(D, 1024);
    layer_norm_kernel<<<seq_len, threads>>>(
        d_output, d_hidden_,
        weights_.d_final_ln_gamma, weights_.d_final_ln_beta,
        D, cfg_.layer_norm_eps);
    CUDA_CHECK_LAST_ERROR();
}

void CLIPEncoder::forwardLayer(int layer_idx, half* d_hidden, int seq_len) {
    int D = cfg_.d_model;
    int head_dim = cfg_.head_dim();
    int n_heads = cfg_.n_heads;
    auto& lw = weights_.layers[layer_idx];

    // === Self-Attention ===

    // LayerNorm
    int threads = std::min(D, 1024);
    layer_norm_kernel<<<seq_len, threads>>>(
        d_normed_, d_hidden,
        lw.d_ln1_gamma, lw.d_ln1_beta,
        D, cfg_.layer_norm_eps);
    CUDA_CHECK_LAST_ERROR();

    // Q, K, V projections: [seq_len, D] * [D, D] -> [seq_len, D]
    gemm(d_normed_, lw.d_wq, d_q_, seq_len, D, D);
    gemm(d_normed_, lw.d_wk, d_k_, seq_len, D, D);
    gemm(d_normed_, lw.d_wv, d_v_, seq_len, D, D);

    // Reshape Q, K, V for multi-head attention
    // Current layout: [seq_len, n_heads * head_dim] (contiguous per token)
    // Need: [n_heads, seq_len, head_dim]
    // For simplicity, compute attention scores using the kernel which handles this layout

    // Attention scores: Q * K^T / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 score_grid(n_heads, seq_len);
    int score_threads = std::min(seq_len, 256);
    attention_scores_2d_kernel<<<score_grid, score_threads>>>(
        d_attn_scores_,
        d_q_, d_k_,
        n_heads, seq_len, seq_len, head_dim, scale);
    CUDA_CHECK_LAST_ERROR();

    // Softmax
    dim3 soft_grid(n_heads, seq_len);
    softmax_2d_kernel<<<soft_grid, std::min(seq_len, 256)>>>(
        d_attn_scores_, n_heads, seq_len, seq_len);
    CUDA_CHECK_LAST_ERROR();

    // Attention output: scores * V
    dim3 out_grid(n_heads, seq_len);
    attention_output_2d_kernel<<<out_grid, head_dim>>>(
        d_attn_out_,
        d_attn_scores_, d_v_,
        n_heads, seq_len, seq_len, head_dim);
    CUDA_CHECK_LAST_ERROR();

    // Output projection
    gemm(d_attn_out_, lw.d_wo, d_normed_, seq_len, D, D);

    // Residual connection
    int total = seq_len * D;
    add_tensors_inplace_kernel<<<(total + 255) / 256, 256>>>(d_hidden, d_normed_, total);
    CUDA_CHECK_LAST_ERROR();

    // === FFN ===

    // LayerNorm
    layer_norm_kernel<<<seq_len, threads>>>(
        d_normed_, d_hidden,
        lw.d_ln2_gamma, lw.d_ln2_beta,
        D, cfg_.layer_norm_eps);
    CUDA_CHECK_LAST_ERROR();

    // FFN: Linear(D -> ff_dim) -> GELU -> Linear(ff_dim -> D)
    int F = cfg_.ff_dim;
    gemm(d_normed_, lw.d_ff1, d_ff_hidden_, seq_len, F, D);

    int ff_total = seq_len * F;
    gelu_kernel<<<(ff_total + 255) / 256, 256>>>(d_ff_hidden_, d_ff_hidden_, ff_total);
    CUDA_CHECK_LAST_ERROR();

    gemm(d_ff_hidden_, lw.d_ff2, d_normed_, seq_len, D, F);

    // Residual connection
    add_tensors_inplace_kernel<<<(total + 255) / 256, 256>>>(d_hidden, d_normed_, total);
    CUDA_CHECK_LAST_ERROR();
}
