// ============================================================================
// FlashAttention-CUDA — correctness tests against the CPU reference.
// ============================================================================

#include "cuda_utils.h"
#include "flash_attention.cuh"
#include "attention_ref.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, name)                                                       \
    do {                                                                        \
        if (cond) { printf("[PASS] %s\n", name); passed++; }                    \
        else      { printf("[FAIL] %s\n", name); failed++; }                    \
    } while (0)

// ----------------------------------------------------------------------------
static void fill_random(std::vector<float>& v, unsigned seed, float scale = 1.0f)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& x : v) x = dist(gen);
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

// relative error normalised by the magnitude of the reference
static float rel_err(const std::vector<float>& got, const std::vector<float>& ref)
{
    float num = 0.0f, den = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den + 1e-12f));
}

// ----------------------------------------------------------------------------
// Run flash forward (fp32) on device and return host O and L.
static void run_flash_fwd(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V,
                          std::vector<float>& O, std::vector<float>& L,
                          const FlashAttnConfig& cfg, FlashKernel kern)
{
    int BH = cfg.batch * cfg.num_heads, N = cfg.seq_len, D = cfg.head_dim;
    size_t nelem = (size_t)BH * N * D, nL = (size_t)BH * N;
    O.resize(nelem); L.resize(nL);

    float *dQ, *dK, *dV, *dO, *dL;
    CUDA_CHECK(cudaMalloc(&dQ, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, nL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));

    flash_attention_forward(dQ, dK, dV, dO, dL, cfg, kern);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O.data(), dO, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L.data(), dL, nL * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
}

static RefDims ref_dims(const FlashAttnConfig& cfg)
{
    return RefDims{cfg.batch, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.causal, cfg.eff_scale()};
}

// ----------------------------------------------------------------------------
static void test_forward(const char* name, int B, int H, int N, int D,
                         bool causal, FlashKernel kern, float tol)
{
    FlashAttnConfig cfg{B, H, N, D, causal, 0.0f};
    size_t nelem = (size_t)B * H * N * D;
    std::vector<float> Q(nelem), K(nelem), V(nelem), O, L;
    fill_random(Q, 1, 1.0f); fill_random(K, 2, 1.0f); fill_random(V, 3, 1.0f);

    run_flash_fwd(Q, K, V, O, L, cfg, kern);

    std::vector<float> Oref, Lref;
    ref_attention_forward(Q, K, V, Oref, Lref, ref_dims(cfg));

    float eo = max_abs_diff(O, Oref);
    float el = max_abs_diff(L, Lref);
    printf("    %-30s  maxabs(O)=%.2e  maxabs(L)=%.2e\n", name, eo, el);
    CHECK(eo < tol && el < tol, name);
}

// large-magnitude scores must not overflow (online softmax stability)
static void test_softmax_stability()
{
    int B = 1, H = 2, N = 96, D = 64;
    FlashAttnConfig cfg{B, H, N, D, false, 0.0f};
    size_t nelem = (size_t)B * H * N * D;
    std::vector<float> Q(nelem), K(nelem), V(nelem), O, L;
    fill_random(Q, 7, 8.0f); fill_random(K, 8, 8.0f); fill_random(V, 9, 1.0f);  // big QK

    run_flash_fwd(Q, K, V, O, L, cfg, FLASH_WARP_ROW);

    std::vector<float> Oref, Lref;
    ref_attention_forward(Q, K, V, Oref, Lref, ref_dims(cfg));

    bool finite = true;
    for (float x : O) if (!std::isfinite(x)) finite = false;
    float eo = max_abs_diff(O, Oref);
    printf("    stability maxabs(O)=%.2e finite=%d\n", eo, (int)finite);
    CHECK(finite && eo < 1e-3f, "Softmax numerical stability (large scores)");
}

static void test_naive_equiv()
{
    int B = 1, H = 2, N = 128, D = 64;
    FlashAttnConfig cfg{B, H, N, D, true, 0.0f};
    size_t nelem = (size_t)B * H * N * D;
    std::vector<float> Q(nelem), K(nelem), V(nelem);
    fill_random(Q, 11, 1.0f); fill_random(K, 12, 1.0f); fill_random(V, 13, 1.0f);

    float *dQ, *dK, *dV, *dO, *dS;
    CUDA_CHECK(cudaMalloc(&dQ, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS, (size_t)B * H * N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));

    naive_attention_forward(dQ, dK, dV, dO, dS, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> Onaive(nelem);
    CUDA_CHECK(cudaMemcpy(Onaive.data(), dO, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS);

    std::vector<float> Oflash, Lf;
    run_flash_fwd(Q, K, V, Oflash, Lf, cfg, FLASH_WARP_ROW);

    float e = max_abs_diff(Onaive, Oflash);
    printf("    naive-vs-flash maxabs(O)=%.2e\n", e);
    CHECK(e < 1e-4f, "Naive (materialised) == Flash forward");
}

static void test_fp16_forward()
{
    int B = 1, H = 2, N = 128, D = 64;
    FlashAttnConfig cfg{B, H, N, D, true, 0.0f};
    size_t nelem = (size_t)B * H * N * D, nL = (size_t)B * H * N;
    std::vector<float> Q(nelem), K(nelem), V(nelem);
    fill_random(Q, 21, 1.0f); fill_random(K, 22, 1.0f); fill_random(V, 23, 1.0f);

    std::vector<__half> Qh(nelem), Kh(nelem), Vh(nelem);
    for (size_t i = 0; i < nelem; ++i) { Qh[i] = __float2half(Q[i]); Kh[i] = __float2half(K[i]); Vh[i] = __float2half(V[i]); }

    __half *dQ, *dK, *dV, *dO; float* dL;
    CUDA_CHECK(cudaMalloc(&dQ, nelem * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dK, nelem * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dV, nelem * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dO, nelem * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dL, nL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Qh.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, Kh.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, Vh.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));

    flash_attention_forward_fp16(dQ, dK, dV, dO, dL, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> Oh(nelem);
    CUDA_CHECK(cudaMemcpy(Oh.data(), dO, nelem * sizeof(__half), cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);

    std::vector<float> O(nelem);
    for (size_t i = 0; i < nelem; ++i) O[i] = __half2float(Oh[i]);

    std::vector<float> Oref, Lref;
    ref_attention_forward(Q, K, V, Oref, Lref, ref_dims(cfg));
    float r = rel_err(O, Oref);
    printf("    fp16 relerr(O)=%.2e\n", r);
    CHECK(r < 2e-2f, "FP16 forward (half I/O) ~ reference");
}

static void test_backward(const char* name, int B, int H, int N, int D, bool causal)
{
    FlashAttnConfig cfg{B, H, N, D, causal, 0.0f};
    size_t nelem = (size_t)B * H * N * D, nL = (size_t)B * H * N;
    std::vector<float> Q(nelem), K(nelem), V(nelem), dO(nelem);
    fill_random(Q, 31, 1.0f); fill_random(K, 32, 1.0f); fill_random(V, 33, 1.0f); fill_random(dO, 34, 1.0f);

    // forward (need O and L)
    std::vector<float> O, L;
    run_flash_fwd(Q, K, V, O, L, cfg, FLASH_WARP_ROW);

    float *dQ, *dK, *dV, *dOut, *ddO, *dL, *ddQ, *ddK, *ddV, *dws;
    CUDA_CHECK(cudaMalloc(&dQ,   nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK,   nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV,   nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddO,  nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL,   nL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddQ,  nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddK,  nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddV,  nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dws,  nL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ,   Q.data(),  nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,   K.data(),  nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,   V.data(),  nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOut, O.data(),  nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ddO,  dO.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dL,   L.data(),  nL * sizeof(float),    cudaMemcpyHostToDevice));

    flash_attention_backward(dQ, dK, dV, dOut, ddO, dL, ddQ, ddK, ddV, dws, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gQ(nelem), gK(nelem), gV(nelem);
    CUDA_CHECK(cudaMemcpy(gQ.data(), ddQ, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gK.data(), ddK, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gV.data(), ddV, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut); cudaFree(ddO);
    cudaFree(dL); cudaFree(ddQ); cudaFree(ddK); cudaFree(ddV); cudaFree(dws);

    std::vector<float> rQ, rK, rV;
    ref_attention_backward(Q, K, V, O, dO, rQ, rK, rV, ref_dims(cfg));

    float eq = rel_err(gQ, rQ), ek = rel_err(gK, rK), ev = rel_err(gV, rV);
    printf("    %-22s relerr dQ=%.2e dK=%.2e dV=%.2e\n", name, eq, ek, ev);
    CHECK(eq < 2e-3f && ek < 2e-3f && ev < 2e-3f, name);
}

// ----------------------------------------------------------------------------
int main()
{
    printf("=== FlashAttention-CUDA tests ===\n");

    test_forward("fwd thread noncaus d=64",  1, 2, 130, 64,  false, FLASH_THREAD_ROW, 1e-3f);
    test_forward("fwd warp   noncaus d=64",  1, 2, 130, 64,  false, FLASH_WARP_ROW, 1e-3f);
    test_forward("fwd warp   causal    d=64",  2, 2, 130, 64,  true,  FLASH_WARP_ROW, 1e-3f);
    test_forward("fwd warp   noncaus d=32",  1, 3,  96, 32,  false, FLASH_WARP_ROW, 1e-3f);
    test_forward("fwd warp   causal    d=128", 1, 2,  72, 128, true,  FLASH_WARP_ROW, 1e-3f);
    test_forward("fwd thread causal    d=128", 1, 2,  72, 128, true,  FLASH_THREAD_ROW, 1e-3f);

    test_softmax_stability();
    test_naive_equiv();
    test_fp16_forward();

    test_backward("bwd noncausal d=64", 1, 2, 100, 64,  false);
    test_backward("bwd causal    d=64", 2, 2, 100, 64,  true);
    test_backward("bwd causal    d=32", 1, 2,  96, 32,  true);

    printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
