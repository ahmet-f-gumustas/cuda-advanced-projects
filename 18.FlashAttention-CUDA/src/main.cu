// ============================================================================
// FlashAttention-CUDA — demo.
//   Runs one multi-head attention forward with the fused flash kernel, checks
//   it against the CPU reference, and prints the O(N) vs O(N^2) memory picture.
// ============================================================================

#include "cuda_utils.h"
#include "flash_attention.cuh"
#include "attention_ref.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>

int main(int argc, char** argv)
{
    int  B = 1, H = 8, N = 1024, D = 64;
    bool causal = true;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--seq")    && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim")    && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads")  && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")  && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noncausal")) causal = false;
    }

    printf("=== FlashAttention-CUDA demo ===\n");
    printf("batch=%d heads=%d seq_len=%d head_dim=%d causal=%d\n\n", B, H, N, D, (int)causal);

    FlashAttnConfig cfg{B, H, N, D, causal, 0.0f};
    size_t BH = (size_t)B * H, nelem = BH * N * D, nL = BH * N;

    std::vector<float> Q(nelem), K(nelem), V(nelem);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : Q) x = dist(gen);
    for (auto& x : K) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    float *dQ, *dK, *dV, *dO, *dL;
    CUDA_CHECK(cudaMalloc(&dQ, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, nelem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, nL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));

    // warm up (first launch pays context/JIT setup), then time
    flash_attention_forward(dQ, dK, dV, dO, dL, cfg, FLASH_THREAD_ROW);
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer t;
    t.start();
    flash_attention_forward(dQ, dK, dV, dO, dL, cfg, FLASH_THREAD_ROW);
    t.stop();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> O(nelem);
    CUDA_CHECK(cudaMemcpy(O.data(), dO, nelem * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Flash forward (thread-row): %.3f ms\n", t.elapsed());
    printf("O[0, 0, 0, 0:8] = ");
    for (int d = 0; d < 8 && d < D; ++d) printf("% .4f ", O[d]);
    printf("\n\n");

    // correctness vs CPU reference on a small slice (head 0, batch 0)
    {
        FlashAttnConfig small{1, 1, (N < 128 ? N : 128), D, causal, 0.0f};
        size_t sn = (size_t)small.seq_len * D;
        std::vector<float> q(Q.begin(), Q.begin() + sn);
        std::vector<float> k(K.begin(), K.begin() + sn);
        std::vector<float> v(V.begin(), V.begin() + sn);
        std::vector<float> oref, lref;
        ref_attention_forward(q, k, v, oref, lref,
            RefDims{1, 1, small.seq_len, D, causal, small.eff_scale()});

        float *sq, *sk, *sv, *so, *sl;
        CUDA_CHECK(cudaMalloc(&sq, sn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sk, sn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv, sn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&so, sn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sl, small.seq_len * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(sq, q.data(), sn * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sk, k.data(), sn * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sv, v.data(), sn * sizeof(float), cudaMemcpyHostToDevice));
        flash_attention_forward(sq, sk, sv, so, sl, small, FLASH_THREAD_ROW);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> og(sn);
        CUDA_CHECK(cudaMemcpy(og.data(), so, sn * sizeof(float), cudaMemcpyDeviceToHost));
        float md = 0.0f;
        for (size_t i = 0; i < sn; ++i) md = fmaxf(md, fabsf(og[i] - oref[i]));
        printf("max|flash - CPU reference| = %.2e  (%s)\n\n", md, md < 1e-3f ? "OK" : "MISMATCH");
        cudaFree(sq); cudaFree(sk); cudaFree(sv); cudaFree(so); cudaFree(sl);
    }

    // memory picture
    printf("Memory: flash vs naive (B=%d H=%d D=%d)\n", B, H, D);
    printf("%-8s | %-16s | %-18s | %s\n", "seq", "flash O+L (MB)", "naive scores (MB)", "ratio");
    printf("---------+------------------+--------------------+-------\n");
    for (int n : {512, 1024, 2048, 4096, 8192}) {
        double flash_mb = (double)BH * n * D * sizeof(float) / 1e6 + (double)BH * n * sizeof(float) / 1e6;
        double naive_mb = (double)BH * (double)n * n * sizeof(float) / 1e6;
        printf("%-8d | %16.2f | %18.2f | %.0fx\n", n, flash_mb, naive_mb, naive_mb / flash_mb);
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
    return 0;
}
