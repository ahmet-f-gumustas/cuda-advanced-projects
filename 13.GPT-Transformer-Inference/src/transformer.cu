#include "../include/transformer.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

// ============================================================================
// cuBLAS error checking
// ============================================================================

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// TransformerModel: constructor / destructor
// ============================================================================

TransformerModel::TransformerModel(const TransformerConfig& cfg)
    : cfg_(cfg)
    , d_residual_(nullptr), d_normed_(nullptr)
    , d_q_(nullptr), d_k_(nullptr), d_v_(nullptr)
    , d_attn_out_(nullptr), d_proj_out_(nullptr)
    , d_ffn_gate_(nullptr), d_ffn_up_(nullptr)
    , d_ffn_hidden_(nullptr), d_ffn_out_(nullptr)
    , d_logits_fp16_(nullptr)
    , d_attn_scores_(nullptr), d_rng_(nullptr)
{
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));

    // Enable Tensor Core math
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH));

    weights_.layers.resize(cfg_.n_layers);
    allocWeights();

    // Allocate activation buffers
    CUDA_CHECK(cudaMalloc(&d_residual_,    cfg_.d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_normed_,      cfg_.d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q_,           cfg_.q_dim()  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_,           cfg_.kv_dim() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_,           cfg_.kv_dim() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_attn_out_,    cfg_.q_dim()  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_proj_out_,    cfg_.d_model  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ffn_gate_,    cfg_.ff_dim   * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ffn_up_,      cfg_.ff_dim   * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ffn_hidden_,  cfg_.ff_dim   * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out_,     cfg_.d_model  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_logits_fp16_, cfg_.vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_attn_scores_,
        (size_t)cfg_.n_heads * cfg_.max_seq_len * sizeof(float)));

    // curand state (1 state for sampling)
    CUDA_CHECK(cudaMalloc(&d_rng_, sizeof(curandState)));
    init_curand_state_kernel<<<1, 1>>>(d_rng_, 12345ULL, 1);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

TransformerModel::~TransformerModel()
{
    freeWeights();

    cudaFree(d_residual_);
    cudaFree(d_normed_);
    cudaFree(d_q_);
    cudaFree(d_k_);
    cudaFree(d_v_);
    cudaFree(d_attn_out_);
    cudaFree(d_proj_out_);
    cudaFree(d_ffn_gate_);
    cudaFree(d_ffn_up_);
    cudaFree(d_ffn_hidden_);
    cudaFree(d_ffn_out_);
    cudaFree(d_logits_fp16_);
    cudaFree(d_attn_scores_);
    cudaFree(d_rng_);

    cublasDestroy(cublas_handle_);
}

// ============================================================================
// Weight allocation
// ============================================================================

void TransformerModel::allocWeights()
{
    auto& w = weights_;
    const auto& c = cfg_;

    CUDA_CHECK(cudaMalloc(&w.d_embed,      (size_t)c.vocab_size * c.d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w.d_final_norm, c.d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w.d_lm_head,   (size_t)c.vocab_size * c.d_model * sizeof(half)));

    for (int l = 0; l < c.n_layers; ++l) {
        auto& lw = w.layers[l];
        lw.d_wq_scale = lw.d_wk_scale = lw.d_wv_scale = lw.d_wo_scale   = nullptr;
        lw.d_gate_scale = lw.d_up_scale = lw.d_down_scale                 = nullptr;
        lw.d_wq_int8 = lw.d_wk_int8 = lw.d_wv_int8 = lw.d_wo_int8       = nullptr;
        lw.d_gate_int8 = lw.d_up_int8 = lw.d_down_int8                   = nullptr;

        CUDA_CHECK(cudaMalloc(&lw.d_attn_norm,  c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_wq,  (size_t)c.q_dim()  * c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_wk,  (size_t)c.kv_dim() * c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_wv,  (size_t)c.kv_dim() * c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_wo,  (size_t)c.d_model  * c.q_dim() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_ffn_norm,   c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_gate_proj, (size_t)c.ff_dim  * c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_up_proj,   (size_t)c.ff_dim  * c.d_model * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&lw.d_down_proj, (size_t)c.d_model * c.ff_dim  * sizeof(half)));
    }
}

void TransformerModel::freeWeights()
{
    auto& w = weights_;
    cudaFree(w.d_embed);
    cudaFree(w.d_final_norm);
    cudaFree(w.d_lm_head);

    for (auto& lw : w.layers) {
        cudaFree(lw.d_attn_norm);
        cudaFree(lw.d_wq); cudaFree(lw.d_wk); cudaFree(lw.d_wv); cudaFree(lw.d_wo);
        cudaFree(lw.d_ffn_norm);
        cudaFree(lw.d_gate_proj); cudaFree(lw.d_up_proj); cudaFree(lw.d_down_proj);
        // INT8 (may be nullptr)
        cudaFree(lw.d_wq_scale);  cudaFree(lw.d_wk_scale);
        cudaFree(lw.d_wv_scale);  cudaFree(lw.d_wo_scale);
        cudaFree(lw.d_gate_scale); cudaFree(lw.d_up_scale); cudaFree(lw.d_down_scale);
        cudaFree(lw.d_wq_int8);   cudaFree(lw.d_wk_int8);
        cudaFree(lw.d_wv_int8);   cudaFree(lw.d_wo_int8);
        cudaFree(lw.d_gate_int8); cudaFree(lw.d_up_int8);  cudaFree(lw.d_down_int8);
    }
}

// ============================================================================
// Helper: upload a FP32 host vector → FP16 device matrix
// ============================================================================

static void upload_fp32_as_fp16(const std::vector<float>& h_buf, half* d_dst, size_t n)
{
    // Convert FP32 → FP16 on host, upload
    std::vector<half> h_half(n);
    for (size_t i = 0; i < n; ++i) h_half[i] = __float2half(h_buf[i]);
    CUDA_CHECK(cudaMemcpy(d_dst, h_half.data(), n * sizeof(half), cudaMemcpyHostToDevice));
}

// ============================================================================
// Random weight initialization (Glorot uniform)
// ============================================================================

void TransformerModel::initRandom(unsigned long long seed)
{
    std::mt19937 rng(seed);
    const auto& c = cfg_;

    auto fill = [&](half* d_ptr, size_t rows, size_t cols) {
        float limit = sqrtf(6.0f / (float)(rows + cols));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::vector<float> h_buf(rows * cols);
        for (auto& v : h_buf) v = dist(rng);
        upload_fp32_as_fp16(h_buf, d_ptr, rows * cols);
    };

    auto fill_ones = [&](half* d_ptr, size_t n) {
        std::vector<half> h_buf(n, __float2half(1.0f));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    };

    fill(weights_.d_embed,      c.vocab_size, c.d_model);
    fill_ones(weights_.d_final_norm, c.d_model);
    fill(weights_.d_lm_head,    c.vocab_size, c.d_model);

    for (int l = 0; l < c.n_layers; ++l) {
        auto& lw = weights_.layers[l];
        fill_ones(lw.d_attn_norm, c.d_model);
        fill(lw.d_wq, c.q_dim(),  c.d_model);
        fill(lw.d_wk, c.kv_dim(), c.d_model);
        fill(lw.d_wv, c.kv_dim(), c.d_model);
        fill(lw.d_wo, c.d_model,  c.q_dim());
        fill_ones(lw.d_ffn_norm, c.d_model);
        fill(lw.d_gate_proj, c.ff_dim, c.d_model);
        fill(lw.d_up_proj,   c.ff_dim, c.d_model);
        fill(lw.d_down_proj, c.d_model, c.ff_dim);
    }

    // Quantize if INT8 mode
    if (cfg_.quant == QuantMode::INT8) {
        quantizeWeightsINT8();
    }
}

// ============================================================================
// Weight load / save (simple binary format)
// ============================================================================

// File format:
//   magic:       uint32  = 0x47505432  ("GPT2")
//   n_layers:    int32
//   d_model:     int32
//   n_heads:     int32
//   n_kv_heads:  int32
//   ff_dim:      int32
//   vocab_size:  int32
//   [data blocks in order, all float32]:
//     embed [vocab_size * d_model]
//     for each layer: attn_norm, wq, wk, wv, wo, ffn_norm, gate_proj, up_proj, down_proj
//     final_norm [d_model]
//     lm_head [vocab_size * d_model]

bool TransformerModel::loadWeights(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[TransformerModel] Cannot open: " << path << std::endl;
        return false;
    }

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x47505432u) {
        std::cerr << "[TransformerModel] Bad magic: " << std::hex << magic << std::endl;
        return false;
    }

    int file_layers, file_dmodel, file_nheads, file_nkvheads, file_ffdim, file_vocab;
    f.read(reinterpret_cast<char*>(&file_layers),   4);
    f.read(reinterpret_cast<char*>(&file_dmodel),   4);
    f.read(reinterpret_cast<char*>(&file_nheads),   4);
    f.read(reinterpret_cast<char*>(&file_nkvheads), 4);
    f.read(reinterpret_cast<char*>(&file_ffdim),    4);
    f.read(reinterpret_cast<char*>(&file_vocab),    4);

    const auto& c = cfg_;
    if (file_layers != c.n_layers || file_dmodel != c.d_model ||
        file_nheads != c.n_heads  || file_nkvheads != c.n_kv_heads ||
        file_ffdim  != c.ff_dim   || file_vocab != c.vocab_size) {
        std::cerr << "[TransformerModel] Config mismatch in weight file!" << std::endl;
        return false;
    }

    auto read_matrix = [&](half* d_ptr, size_t n) {
        std::vector<float> h_buf(n);
        f.read(reinterpret_cast<char*>(h_buf.data()), n * sizeof(float));
        upload_fp32_as_fp16(h_buf, d_ptr, n);
    };

    read_matrix(weights_.d_embed, (size_t)c.vocab_size * c.d_model);
    for (int l = 0; l < c.n_layers; ++l) {
        auto& lw = weights_.layers[l];
        read_matrix(lw.d_attn_norm, c.d_model);
        read_matrix(lw.d_wq,  (size_t)c.q_dim()  * c.d_model);
        read_matrix(lw.d_wk,  (size_t)c.kv_dim() * c.d_model);
        read_matrix(lw.d_wv,  (size_t)c.kv_dim() * c.d_model);
        read_matrix(lw.d_wo,  (size_t)c.d_model  * c.q_dim());
        read_matrix(lw.d_ffn_norm, c.d_model);
        read_matrix(lw.d_gate_proj, (size_t)c.ff_dim * c.d_model);
        read_matrix(lw.d_up_proj,   (size_t)c.ff_dim * c.d_model);
        read_matrix(lw.d_down_proj, (size_t)c.d_model * c.ff_dim);
    }
    read_matrix(weights_.d_final_norm, c.d_model);
    read_matrix(weights_.d_lm_head, (size_t)c.vocab_size * c.d_model);

    std::cout << "[TransformerModel] Loaded weights from: " << path << std::endl;

    if (cfg_.quant == QuantMode::INT8) quantizeWeightsINT8();
    return true;
}

bool TransformerModel::saveWeights(const std::string& path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    const auto& c = cfg_;
    uint32_t magic = 0x47505432u;
    f.write(reinterpret_cast<const char*>(&magic),        4);
    f.write(reinterpret_cast<const char*>(&c.n_layers),   4);
    f.write(reinterpret_cast<const char*>(&c.d_model),    4);
    f.write(reinterpret_cast<const char*>(&c.n_heads),    4);
    f.write(reinterpret_cast<const char*>(&c.n_kv_heads), 4);
    f.write(reinterpret_cast<const char*>(&c.ff_dim),     4);
    f.write(reinterpret_cast<const char*>(&c.vocab_size), 4);

    auto write_matrix = [&](const half* d_ptr, size_t n) {
        std::vector<half> h_half(n);
        CUDA_CHECK(cudaMemcpy(h_half.data(), d_ptr, n * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<float> h_float(n);
        for (size_t i = 0; i < n; ++i) h_float[i] = __half2float(h_half[i]);
        f.write(reinterpret_cast<const char*>(h_float.data()), n * sizeof(float));
    };

    write_matrix(weights_.d_embed, (size_t)c.vocab_size * c.d_model);
    for (int l = 0; l < c.n_layers; ++l) {
        const auto& lw = weights_.layers[l];
        write_matrix(lw.d_attn_norm, c.d_model);
        write_matrix(lw.d_wq,  (size_t)c.q_dim()  * c.d_model);
        write_matrix(lw.d_wk,  (size_t)c.kv_dim() * c.d_model);
        write_matrix(lw.d_wv,  (size_t)c.kv_dim() * c.d_model);
        write_matrix(lw.d_wo,  (size_t)c.d_model  * c.q_dim());
        write_matrix(lw.d_ffn_norm, c.d_model);
        write_matrix(lw.d_gate_proj, (size_t)c.ff_dim * c.d_model);
        write_matrix(lw.d_up_proj,   (size_t)c.ff_dim * c.d_model);
        write_matrix(lw.d_down_proj, (size_t)c.d_model * c.ff_dim);
    }
    write_matrix(weights_.d_final_norm, c.d_model);
    write_matrix(weights_.d_lm_head, (size_t)c.vocab_size * c.d_model);

    std::cout << "[TransformerModel] Saved weights to: " << path << std::endl;
    return true;
}

// ============================================================================
// INT8 quantization (post-init)
// ============================================================================

// Forward declaration of host helper (defined in quantization.cu)
void quantize_matrix(const half* d_fp16, int8_t** d_int8_out,
                     float** d_scales_out, int rows, int cols);

void TransformerModel::quantizeWeightsINT8()
{
    const auto& c = cfg_;
    for (int l = 0; l < c.n_layers; ++l) {
        auto& lw = weights_.layers[l];
        quantize_matrix(lw.d_wq, &lw.d_wq_int8, &lw.d_wq_scale, c.q_dim(),  c.d_model);
        quantize_matrix(lw.d_wk, &lw.d_wk_int8, &lw.d_wk_scale, c.kv_dim(), c.d_model);
        quantize_matrix(lw.d_wv, &lw.d_wv_int8, &lw.d_wv_scale, c.kv_dim(), c.d_model);
        quantize_matrix(lw.d_wo, &lw.d_wo_int8, &lw.d_wo_scale, c.d_model,  c.q_dim());
        quantize_matrix(lw.d_gate_proj, &lw.d_gate_int8, &lw.d_gate_scale, c.ff_dim, c.d_model);
        quantize_matrix(lw.d_up_proj,   &lw.d_up_int8,   &lw.d_up_scale,   c.ff_dim, c.d_model);
        quantize_matrix(lw.d_down_proj, &lw.d_down_int8, &lw.d_down_scale, c.d_model, c.ff_dim);
    }
    std::cout << "[TransformerModel] INT8 quantization complete." << std::endl;
}

// ============================================================================
// GEMM helper: C[m,n] = alpha * A[m,k] * B[n,k]^T + beta * C[m,n]
// cuBLAS is column-major: we compute B * A^T in column-major = A * B^T in row-major
// For single-token: m=1 always
// ============================================================================

void TransformerModel::gemm_fp16(const half* d_A, const half* d_B, half* d_C,
                                  int m, int n, int k,
                                  float alpha, float beta)
{
    // cuBLAS column-major: C[n,m] = B[n,k] * A[k,m]
    // We call: C = B * A^T  →  in row-major: C[m,n] = A[m,k] * B[n,k]^T ✓
    __half alpha_h = __float2half(alpha);
    __half beta_h  = __float2half(beta);

    CUBLAS_CHECK(cublasHgemm(cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha_h,
        d_B, k,    // B: [n, k] leading dim = k
        d_A, k,    // A: [m, k] leading dim = k
        &beta_h,
        d_C, n));  // C: [m, n] leading dim = n
}

// ============================================================================
// Single-layer forward pass
// ============================================================================

void TransformerModel::forward_layer(int layer_idx, KVCache& kv_cache, int pos)
{
    const auto& c  = cfg_;
    const auto& lw = weights_.layers[layer_idx];

    int seq_len = pos + 1;  // Tokens in cache including current
    float scale = 1.0f / sqrtf((float)c.head_dim);

    // ── 1. Attention pre-norm ──────────────────────────────────────────────
    int smem_rmsnorm = ((PT_BLOCK_SIZE + PT_WARP_SIZE - 1) / PT_WARP_SIZE) * sizeof(float);
    rmsnorm_kernel<<<1, PT_BLOCK_SIZE, smem_rmsnorm>>>(
        d_normed_, d_residual_, lw.d_attn_norm, c.d_model, PT_RMSNORM_EPS);
    CUDA_CHECK_LAST_ERROR();

    // ── 2. QKV projections (FP16 cuBLAS) ─────────────────────────────────
    // Q = d_normed_ @ Wq^T   [1, d_model] x [d_model, q_dim] → [1, q_dim]
    gemm_fp16(d_normed_, lw.d_wq, d_q_, 1, c.q_dim(), c.d_model);
    // K = d_normed_ @ Wk^T   → [1, kv_dim]
    gemm_fp16(d_normed_, lw.d_wk, d_k_, 1, c.kv_dim(), c.d_model);
    // V = d_normed_ @ Wv^T   → [1, kv_dim]
    gemm_fp16(d_normed_, lw.d_wv, d_v_, 1, c.kv_dim(), c.d_model);

    // ── 3. RoPE ──────────────────────────────────────────────────────────
    rope_kernel<<<c.n_heads,    c.head_dim / 2>>>(d_q_, pos, c.n_heads,    c.head_dim, c.rope_theta);
    rope_kernel<<<c.n_kv_heads, c.head_dim / 2>>>(d_k_, pos, c.n_kv_heads, c.head_dim, c.rope_theta);
    CUDA_CHECK_LAST_ERROR();

    // ── 4. Write KV cache ────────────────────────────────────────────────
    write_kv_cache_kernel<<<c.n_kv_heads, c.head_dim>>>(
        kv_cache.k_ptr(layer_idx), kv_cache.v_ptr(layer_idx),
        d_k_, d_v_, pos, c.n_kv_heads, c.max_seq_len, c.head_dim);
    CUDA_CHECK_LAST_ERROR();

    // ── 5. Attention scores (GQA) ────────────────────────────────────────
    int score_block = (seq_len < 256) ? seq_len : 256;
    dim3 score_grid(c.n_heads, (seq_len + score_block - 1) / score_block);
    attention_scores_kernel<<<score_grid, score_block>>>(
        d_attn_scores_, d_q_, kv_cache.k_ptr(layer_idx),
        c.n_heads, c.n_kv_heads, c.head_dim, seq_len, c.max_seq_len, scale);
    CUDA_CHECK_LAST_ERROR();

    // ── 6. Softmax over seq_len ──────────────────────────────────────────
    int smem_soft = ((min(seq_len, 512) + PT_WARP_SIZE - 1) / PT_WARP_SIZE) * sizeof(float);
    int soft_block = (seq_len < 512) ? seq_len : 512;
    softmax_kernel<<<c.n_heads, soft_block, smem_soft>>>(
        d_attn_scores_, c.n_heads, seq_len);
    CUDA_CHECK_LAST_ERROR();

    // ── 7. Weighted sum V ────────────────────────────────────────────────
    attention_output_kernel<<<c.n_heads, c.head_dim>>>(
        d_attn_out_, d_attn_scores_, kv_cache.v_ptr(layer_idx),
        c.n_heads, c.n_kv_heads, seq_len, c.max_seq_len, c.head_dim);
    CUDA_CHECK_LAST_ERROR();

    // ── 8. Output projection + residual ─────────────────────────────────
    // proj_out = attn_out @ Wo^T   [1, q_dim] x [q_dim, d_model] → [1, d_model]
    gemm_fp16(d_attn_out_, lw.d_wo, d_proj_out_, 1, c.d_model, c.q_dim());

    int blocks1d = (c.d_model + PT_BLOCK_SIZE - 1) / PT_BLOCK_SIZE;
    add_residual_kernel<<<blocks1d, PT_BLOCK_SIZE>>>(d_residual_, d_proj_out_, c.d_model);
    CUDA_CHECK_LAST_ERROR();

    // ── 9. FFN pre-norm ──────────────────────────────────────────────────
    rmsnorm_kernel<<<1, PT_BLOCK_SIZE, smem_rmsnorm>>>(
        d_normed_, d_residual_, lw.d_ffn_norm, c.d_model, PT_RMSNORM_EPS);
    CUDA_CHECK_LAST_ERROR();

    // ── 10. SwiGLU FFN ──────────────────────────────────────────────────
    // gate = d_normed_ @ Wgate^T  → [1, ff_dim]
    gemm_fp16(d_normed_, lw.d_gate_proj, d_ffn_gate_, 1, c.ff_dim, c.d_model);
    // up   = d_normed_ @ Wup^T    → [1, ff_dim]
    gemm_fp16(d_normed_, lw.d_up_proj, d_ffn_up_, 1, c.ff_dim, c.d_model);

    int ffn_blocks = (c.ff_dim + PT_BLOCK_SIZE - 1) / PT_BLOCK_SIZE;
    swiglu_kernel<<<ffn_blocks, PT_BLOCK_SIZE>>>(d_ffn_hidden_, d_ffn_gate_, d_ffn_up_, c.ff_dim);
    CUDA_CHECK_LAST_ERROR();

    // down = hidden @ Wdown^T  → [1, d_model]
    gemm_fp16(d_ffn_hidden_, lw.d_down_proj, d_ffn_out_, 1, c.d_model, c.ff_dim);

    add_residual_kernel<<<blocks1d, PT_BLOCK_SIZE>>>(d_residual_, d_ffn_out_, c.d_model);
    CUDA_CHECK_LAST_ERROR();
}

// ============================================================================
// Single-token forward pass
// ============================================================================

void TransformerModel::forward(int token_id, float* d_logits, KVCache& kv_cache, int pos)
{
    const auto& c = cfg_;

    // ── Embedding lookup ──────────────────────────────────────────────────
    int emb_blocks = (c.d_model + PT_BLOCK_SIZE - 1) / PT_BLOCK_SIZE;
    embedding_lookup_kernel<<<emb_blocks, PT_BLOCK_SIZE>>>(
        d_residual_, weights_.d_embed, token_id, c.d_model);
    CUDA_CHECK_LAST_ERROR();

    // ── Decoder layers ───────────────────────────────────────────────────
    for (int l = 0; l < c.n_layers; ++l) {
        forward_layer(l, kv_cache, pos);
    }

    // ── Final norm ───────────────────────────────────────────────────────
    int smem = ((PT_BLOCK_SIZE + PT_WARP_SIZE - 1) / PT_WARP_SIZE) * sizeof(float);
    rmsnorm_kernel<<<1, PT_BLOCK_SIZE, smem>>>(
        d_normed_, d_residual_, weights_.d_final_norm, c.d_model, PT_RMSNORM_EPS);
    CUDA_CHECK_LAST_ERROR();

    // ── LM head: logits = d_normed_ @ lm_head^T  → [vocab_size] ─────────
    gemm_fp16(d_normed_, weights_.d_lm_head, d_logits_fp16_, 1, c.vocab_size, c.d_model);

    // ── Cast FP16 → FP32 for sampling ────────────────────────────────────
    int vocab_blocks = (c.vocab_size + PT_BLOCK_SIZE - 1) / PT_BLOCK_SIZE;
    logits_fp16_to_fp32_kernel<<<vocab_blocks, PT_BLOCK_SIZE>>>(
        d_logits_fp16_, d_logits, c.vocab_size);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Prefill: process prompt tokens
// ============================================================================

void TransformerModel::prefill(const std::vector<int>& token_ids, KVCache& kv_cache)
{
    // Allocate temporary logit buffer (discarded during prefill)
    float* d_tmp_logits = cudaMallocDevice<float>(cfg_.vocab_size);

    for (int i = 0; i < (int)token_ids.size(); ++i) {
        forward(token_ids[i], d_tmp_logits, kv_cache, kv_cache.current_pos);
        kv_cache.current_pos++;
    }

    cudaFree(d_tmp_logits);
}
