// ============================================================================
// FlashAttention-CUDA — performance benchmark.
//   Flash thread-row / warp-row / fp16  vs  naive (materialised) attention.
//   Reports latency, GFLOP/s, and the O(N^2) score-matrix memory the flash
//   path avoids.
// ============================================================================

#include "cuda_utils.h"
#include "flash_attention.cuh"

#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <string>

static void fill_random(std::vector<float>& v, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// average ms over `iters` launches (after warmup)
template<typename F>
static float bench_ms(F launch, int iters)
{
    for (int i = 0; i < 3; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) launch();
    t.stop();
    return t.elapsed() / iters;
}

static double gflops(const FlashAttnConfig& cfg, float ms)
{
    double bh = (double)cfg.batch * cfg.num_heads;
    double f  = 4.0 * bh * (double)cfg.seq_len * cfg.seq_len * cfg.head_dim; // QK^T + PV
    if (cfg.causal) f *= 0.5;
    return f / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv)
{
    int  B = 1, H = 8, D = 64;
    bool causal = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--causal") causal = true;
        else if (a == "--heads" && i + 1 < argc) H = atoi(argv[++i]);
        else if (a == "--dim"   && i + 1 < argc) D = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) B = atoi(argv[++i]);
    }

    printDeviceInfo();
    printf("Config: batch=%d heads=%d head_dim=%d causal=%d\n\n", B, H, D, (int)causal);
    printf("%-7s | %-26s | %-26s | %-26s | %-12s | %s\n",
           "seq", "naive (ms / GFLOPs)", "flash thread (ms/GFLOPs)",
           "flash warp (ms/GFLOPs)", "fp16 ms", "scores buf");
    printf("--------+----------------------------+----------------------------+"
           "----------------------------+--------------+-----------\n");

    int seqs[] = {256, 512, 1024, 2048, 4096};

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    for (int N : seqs) {
        FlashAttnConfig cfg{B, H, N, D, causal, 0.0f};
        size_t BH = (size_t)B * H, nelem = BH * N * D, nL = BH * N;
        int iters = (N <= 1024) ? 50 : (N <= 2048 ? 20 : 10);

        std::vector<float> Q(nelem), K(nelem), V(nelem);
        fill_random(Q, 1); fill_random(K, 2); fill_random(V, 3);

        float *dQ, *dK, *dV, *dO, *dL;
        CUDA_CHECK(cudaMalloc(&dQ, nelem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK, nelem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV, nelem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO, nelem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dL, nL * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dQ, Q.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, K.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, V.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));

        // --- naive (skip if the N^2 score buffer doesn't fit comfortably) ---
        size_t scores_bytes = BH * (size_t)N * N * sizeof(float);
        char naive_col[40];
        if (scores_bytes < free_b / 2) {
            float* dS;
            CUDA_CHECK(cudaMalloc(&dS, scores_bytes));
            float ms = bench_ms([&]{ naive_attention_forward(dQ, dK, dV, dO, dS, cfg); }, iters);
            snprintf(naive_col, sizeof(naive_col), "%8.3f / %8.1f", ms, gflops(cfg, ms));
            cudaFree(dS);
        } else {
            snprintf(naive_col, sizeof(naive_col), "  OOM (%.0f MB)", scores_bytes / 1.0e6);
        }

        // --- flash thread-row / warp-row ---
        float ms1 = bench_ms([&]{ flash_attention_forward(dQ, dK, dV, dO, dL, cfg, FLASH_THREAD_ROW); }, iters);
        float ms2 = bench_ms([&]{ flash_attention_forward(dQ, dK, dV, dO, dL, cfg, FLASH_WARP_ROW); }, iters);

        // --- fp16 (thread-row) ---
        std::vector<__half> Qh(nelem), Kh(nelem), Vh(nelem);
        for (size_t i = 0; i < nelem; ++i) { Qh[i] = __float2half(Q[i]); Kh[i] = __float2half(K[i]); Vh[i] = __float2half(V[i]); }
        __half *hQ, *hK, *hV, *hO;
        CUDA_CHECK(cudaMalloc(&hQ, nelem * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&hK, nelem * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&hV, nelem * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&hO, nelem * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(hQ, Qh.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(hK, Kh.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(hV, Vh.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
        float msh = bench_ms([&]{ flash_attention_forward_fp16(hQ, hK, hV, hO, dL, cfg); }, iters);

        printf("%-7d | %-26s | %8.3f / %8.1f | %8.3f / %8.1f | %12.3f | %7.0f MB\n",
               N, naive_col, ms1, gflops(cfg, ms1), ms2, gflops(cfg, ms2), msh,
               scores_bytes / 1.0e6);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
        cudaFree(hQ); cudaFree(hK); cudaFree(hV); cudaFree(hO);
    }

    printf("\nFLOPs = 4 * B*H*N^2*D (halved for causal). Flash never allocates the\n"
           "'scores buf' column — that is the O(N^2) matrix the naive path materialises.\n");
    return 0;
}
