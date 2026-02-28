#include "../include/transformer_kernels.cuh"
#include "../include/cuda_utils.h"
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

// ============================================================================
// CPU reference implementations for testing
// ============================================================================

// CPU RMSNorm
void cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w,
                 std::vector<float>& out, float eps = 1e-6f)
{
    int d = (int)x.size();
    float sum_sq = 0.f;
    for (float v : x) sum_sq += v * v;
    float rms = sqrtf(sum_sq / d + eps);
    out.resize(d);
    for (int i = 0; i < d; ++i) out[i] = x[i] / rms * w[i];
}

// CPU SwiGLU
void cpu_swiglu(const std::vector<float>& gate, const std::vector<float>& up,
                std::vector<float>& out)
{
    int n = (int)gate.size();
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        float silu = g / (1.f + expf(-g));
        out[i] = silu * up[i];
    }
}

// CPU attention scores (no GQA, n_kv_heads = n_heads for simplicity)
void cpu_attention_scores(const std::vector<float>& Q,   // [n_heads, head_dim]
                           const std::vector<float>& K,   // [seq_len, n_heads, head_dim]
                           std::vector<float>& scores,    // [n_heads, seq_len]
                           int n_heads, int head_dim, int seq_len, float scale)
{
    scores.resize(n_heads * seq_len, 0.f);
    for (int h = 0; h < n_heads; ++h) {
        for (int t = 0; t < seq_len; ++t) {
            float dot = 0.f;
            for (int i = 0; i < head_dim; ++i)
                dot += Q[h * head_dim + i] * K[t * n_heads * head_dim + h * head_dim + i];
            scores[h * seq_len + t] = dot * scale;
        }
    }
}

// CPU softmax (per row)
void cpu_softmax(std::vector<float>& x, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        float max_v = -1e30f;
        for (int c = 0; c < cols; ++c) max_v = std::max(max_v, x[r * cols + c]);
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) { x[r * cols + c] = expf(x[r * cols + c] - max_v); sum += x[r * cols + c]; }
        for (int c = 0; c < cols; ++c) x[r * cols + c] /= sum;
    }
}

// ============================================================================
// Helpers
// ============================================================================

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    assert(a.size() == b.size());
    float d = 0.f;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

static void randfill(std::vector<float>& v, std::mt19937& rng, float lo = -1.f, float hi = 1.f)
{
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

static std::vector<half> to_half(const std::vector<float>& v)
{
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

static std::vector<float> from_half_device(const half* d_ptr, size_t n)
{
    std::vector<half> h_half(n);
    CUDA_CHECK(cudaMemcpy(h_half.data(), d_ptr, n * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __half2float(h_half[i]);
    return out;
}

static std::vector<float> from_float_device(const float* d_ptr, size_t n)
{
    std::vector<float> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

// ============================================================================
// Test 1: RMSNorm
// ============================================================================

bool test_rmsnorm()
{
    std::cout << "[Test] RMSNorm... " << std::flush;
    std::mt19937 rng(1);
    int d = 512;

    std::vector<float> h_x(d), h_w(d);
    randfill(h_x, rng); randfill(h_w, rng, 0.5f, 1.5f);

    std::vector<float> cpu_out;
    cpu_rmsnorm(h_x, h_w, cpu_out);

    auto h_x_half = to_half(h_x);
    auto h_w_half = to_half(h_w);

    half *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   d * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,   d * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, d * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x_half.data(), d * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w_half.data(), d * sizeof(half), cudaMemcpyHostToDevice));

    int smem = ((256 + 32 - 1) / 32) * sizeof(float);
    rmsnorm_kernel<<<1, 256, smem>>>(d_out, d_x, d_w, d, 1e-6f);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_out = from_half_device(d_out, d);
    float err = max_abs_diff(cpu_out, gpu_out);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);

    // FP16 has ~1e-3 precision
    bool pass = err < 2e-2f;
    std::cout << (pass ? "PASS" : "FAIL") << "  max_err=" << err << "\n";
    return pass;
}

// ============================================================================
// Test 2: SwiGLU
// ============================================================================

bool test_swiglu()
{
    std::cout << "[Test] SwiGLU... " << std::flush;
    std::mt19937 rng(2);
    int ff = 2048;

    std::vector<float> h_gate(ff), h_up(ff);
    randfill(h_gate, rng); randfill(h_up, rng);

    std::vector<float> cpu_out;
    cpu_swiglu(h_gate, h_up, cpu_out);

    auto hg = to_half(h_gate), hu = to_half(h_up);
    half *d_gate, *d_up, *d_hidden;
    CUDA_CHECK(cudaMalloc(&d_gate,   ff * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up,     ff * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_hidden, ff * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_gate, hg.data(), ff * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up,   hu.data(), ff * sizeof(half), cudaMemcpyHostToDevice));

    swiglu_kernel<<<(ff + 255) / 256, 256>>>(d_hidden, d_gate, d_up, ff);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_out = from_half_device(d_hidden, ff);
    float err = max_abs_diff(cpu_out, gpu_out);

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_hidden);

    bool pass = err < 2e-2f;
    std::cout << (pass ? "PASS" : "FAIL") << "  max_err=" << err << "\n";
    return pass;
}

// ============================================================================
// Test 3: Attention scores + softmax (no GQA, n_kv_heads = n_heads)
// ============================================================================

bool test_attention()
{
    std::cout << "[Test] Attention scores + softmax... " << std::flush;
    std::mt19937 rng(3);
    int n_heads = 4, head_dim = 32, seq_len = 64;
    float scale = 1.f / sqrtf((float)head_dim);

    // Q: [n_heads, head_dim]
    std::vector<float> h_Q(n_heads * head_dim);
    randfill(h_Q, rng);

    // K_cache: [n_heads, seq_len, head_dim]  (n_kv_heads = n_heads for this test)
    std::vector<float> h_K(n_heads * seq_len * head_dim);
    randfill(h_K, rng);

    // CPU scores: [n_heads, seq_len]
    // Reformat K to [seq_len, n_heads, head_dim] for cpu_attention_scores
    std::vector<float> h_K_cpu(seq_len * n_heads * head_dim);
    for (int h = 0; h < n_heads; ++h)
        for (int t = 0; t < seq_len; ++t)
            for (int i = 0; i < head_dim; ++i)
                h_K_cpu[t * n_heads * head_dim + h * head_dim + i] =
                    h_K[h * seq_len * head_dim + t * head_dim + i];

    std::vector<float> cpu_scores;
    cpu_attention_scores(h_Q, h_K_cpu, cpu_scores, n_heads, head_dim, seq_len, scale);
    cpu_softmax(cpu_scores, n_heads, seq_len);

    // GPU
    auto hQ = to_half(h_Q);
    auto hK = to_half(h_K);

    half  *d_q, *d_k;
    float *d_scores;
    CUDA_CHECK(cudaMalloc(&d_q, n_heads * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, (size_t)n_heads * seq_len * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)n_heads * seq_len * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, hQ.data(), n_heads * head_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, hK.data(), (size_t)n_heads * seq_len * head_dim * sizeof(half), cudaMemcpyHostToDevice));

    // max_seq_len = seq_len (cache is exactly filled)
    dim3 sgrid(n_heads, (seq_len + 31) / 32);
    attention_scores_kernel<<<sgrid, 32>>>(
        d_scores, d_q, d_k, n_heads, n_heads /*n_kv_heads = n_heads, GQA ratio=1*/,
        head_dim, seq_len, seq_len /*max_seq_len*/, scale);
    CUDA_CHECK_LAST_ERROR();

    int smem = ((min(seq_len, 512) + 31) / 32) * sizeof(float);
    softmax_kernel<<<n_heads, min(seq_len, 512), smem>>>(d_scores, n_heads, seq_len);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_scores = from_float_device(d_scores, n_heads * seq_len);
    float err = max_abs_diff(cpu_scores, gpu_scores);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_scores);

    bool pass = err < 1e-3f;
    std::cout << (pass ? "PASS" : "FAIL") << "  max_err=" << err << "\n";
    return pass;
}

// ============================================================================
// Test 4: Embedding lookup
// ============================================================================

bool test_embedding()
{
    std::cout << "[Test] Embedding lookup... " << std::flush;
    std::mt19937 rng(4);
    int vocab = 256, d_model = 128, token_id = 65; // 'A'

    std::vector<float> h_table(vocab * d_model);
    randfill(h_table, rng);

    // CPU: just copy row token_id
    std::vector<float> cpu_out(h_table.begin() + token_id * d_model,
                                h_table.begin() + (token_id + 1) * d_model);

    auto ht = to_half(h_table);
    half *d_table, *d_out;
    CUDA_CHECK(cudaMalloc(&d_table, vocab * d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,   d_model * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_table, ht.data(), vocab * d_model * sizeof(half), cudaMemcpyHostToDevice));

    embedding_lookup_kernel<<<(d_model + 255) / 256, 256>>>(d_out, d_table, token_id, d_model);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_out = from_half_device(d_out, d_model);
    float err = max_abs_diff(cpu_out, gpu_out);

    cudaFree(d_table); cudaFree(d_out);

    // FP16 precision
    bool pass = err < 1e-2f;
    std::cout << (pass ? "PASS" : "FAIL") << "  max_err=" << err << "\n";
    return pass;
}

// ============================================================================
// Test 5: RoPE — verify that rotating twice by -pos undoes the rotation
// ============================================================================

bool test_rope()
{
    std::cout << "[Test] RoPE invertibility... " << std::flush;
    std::mt19937 rng(5);
    int n_heads = 4, head_dim = 32, pos = 17;

    std::vector<float> h_x(n_heads * head_dim);
    randfill(h_x, rng);
    std::vector<float> original = h_x;

    auto hx = to_half(h_x);
    half* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, n_heads * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), n_heads * head_dim * sizeof(half), cudaMemcpyHostToDevice));

    // Apply RoPE at pos
    rope_kernel<<<n_heads, head_dim / 2>>>(d_x, pos, n_heads, head_dim, 10000.f);
    CUDA_CHECK_LAST_ERROR();
    // Apply RoPE at -pos (should cancel out)
    rope_kernel<<<n_heads, head_dim / 2>>>(d_x, -pos, n_heads, head_dim, 10000.f);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_out = from_half_device(d_x, n_heads * head_dim);
    float err = max_abs_diff(original, gpu_out);
    cudaFree(d_x);

    // FP16 round-trip error is typically < 0.01
    bool pass = err < 5e-2f;
    std::cout << (pass ? "PASS" : "FAIL") << "  max_err=" << err << "\n";
    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    printDeviceInfo();

    std::cout << "\n=== GPT Transformer Kernel Tests ===\n\n";

    int pass_count = 0, total = 0;
    auto run = [&](bool (*fn)()) { ++total; if (fn()) ++pass_count; };

    run(test_rmsnorm);
    run(test_swiglu);
    run(test_attention);
    run(test_embedding);
    run(test_rope);

    std::cout << "\n";
    if (pass_count == total) {
        std::cout << "All " << total << " tests PASSED!\n";
    } else {
        std::cout << pass_count << "/" << total << " tests passed.\n";
    }

    return (pass_count == total) ? 0 : 1;
}
