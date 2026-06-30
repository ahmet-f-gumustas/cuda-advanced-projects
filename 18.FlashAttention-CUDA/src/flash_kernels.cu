// ============================================================================
// FlashAttention-CUDA — kernels + host launchers
//
//   naive_*        : materialised [N x N] attention (baseline + memory anchor)
//   flash_fwd_v1   : thread-per-query-row, shared-memory K/V tiling
//   flash_fwd_v2   : warp-per-query-row, lane-parallel head_dim, shuffle reduce
//   flash_fwd_fp16 : v2 design with half-precision I/O, float accumulate
//   flash_bwd_*    : recompute-based backward (dQ, dK, dV) — atomic-free
// ============================================================================

#include "flash_attention.cuh"
#include "cuda_utils.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

static constexpr int WARP = 32;
static constexpr int TILE = 32;   // K/V rows streamed through shared mem (v2/fp16)
static constexpr int WPB  = 8;    // warps (= query rows) per block for v2/backward

#define DISPATCH_HEAD_DIM(HD, CALL)                                              \
    switch (HD) {                                                                \
        case 32:  CALL(32);  break;                                             \
        case 64:  CALL(64);  break;                                             \
        case 128: CALL(128); break;                                             \
        default:                                                                \
            std::fprintf(stderr,                                                \
                "[flash] unsupported head_dim=%d (use 32 / 64 / 128)\n", HD);   \
            std::exit(EXIT_FAILURE);                                            \
    }

// ============================================================================
// Naive materialised attention (FP32)
// ============================================================================

__global__ void naive_scores_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     float* __restrict__ S,
                                     int N, int D, float scale, bool causal)
{
    int i  = blockIdx.x;
    int bh = blockIdx.y;
    const float* Qi  = Q + ((size_t)bh * N + i) * D;
    const float* Kb  = K + (size_t)bh * N * D;
    float*       Sr  = S + ((size_t)bh * N + i) * N;

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        if (causal && j > i) { Sr[j] = -INFINITY; continue; }
        const float* Kj = Kb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += Qi[d] * Kj[d];
        Sr[j] = dot * scale;
    }
}

__global__ void naive_softmax_kernel(float* __restrict__ S, int N)
{
    int i  = blockIdx.x;
    int bh = blockIdx.y;
    float* Sr = S + ((size_t)bh * N + i) * N;

    __shared__ float red[256];

    float m = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, Sr[j]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float l = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = __expf(Sr[j] - m);
        Sr[j] = e;
        l += e;
    }
    red[threadIdx.x] = l;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    l = red[0];
    __syncthreads();

    float inv = 1.0f / l;
    for (int j = threadIdx.x; j < N; j += blockDim.x) Sr[j] *= inv;
}

__global__ void naive_output_kernel(const float* __restrict__ P,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    int N, int D)
{
    int i  = blockIdx.x;
    int bh = blockIdx.y;
    const float* Pr = P + ((size_t)bh * N + i) * N;
    const float* Vb = V + (size_t)bh * N * D;
    float*       Oi = O + ((size_t)bh * N + i) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += Pr[j] * Vb[(size_t)j * D + d];
        Oi[d] = acc;
    }
}

void naive_attention_forward(const float* d_Q, const float* d_K, const float* d_V,
                             float* d_O, float* d_scores,
                             const FlashAttnConfig& cfg, cudaStream_t stream)
{
    int   N  = cfg.seq_len, D = cfg.head_dim, BH = cfg.batch * cfg.num_heads;
    float sc = cfg.eff_scale();
    dim3 grid(N, BH);
    naive_scores_kernel <<<grid, 256, 0, stream>>>(d_Q, d_K, d_scores, N, D, sc, cfg.causal);
    naive_softmax_kernel<<<grid, 256, 0, stream>>>(d_scores, N);
    naive_output_kernel <<<grid, 256, 0, stream>>>(d_scores, d_V, d_O, N, D);
    CUDA_CHECK_LAST_ERROR();
}

// ============================================================================
// Flash forward (thread-row) — one thread per query row, shared-memory K/V
// tiling, full head_dim accumulator in registers. Fastest design for
// head_dim <= 128 (no per-key warp-shuffle; high instruction-level parallelism).
// ============================================================================

template<int HEAD_DIM>
__global__ void flash_fwd_thread_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    float* __restrict__ L,
                                    int N, float scale, bool causal)
{
    extern __shared__ float smem[];
    const int BR = blockDim.x;            // query rows per block == K/V tile rows
    float* sK = smem;                     // [BR * HEAD_DIM]
    float* sV = smem + BR * HEAD_DIM;     // [BR * HEAD_DIM]

    int bh = blockIdx.y;
    const float* Qb = Q + (size_t)bh * N * HEAD_DIM;
    const float* Kb = K + (size_t)bh * N * HEAD_DIM;
    const float* Vb = V + (size_t)bh * N * HEAD_DIM;
    float*       Ob = O + (size_t)bh * N * HEAD_DIM;

    int  qi     = blockIdx.x * BR + threadIdx.x;
    bool active = qi < N;

    float q[HEAD_DIM], acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q[d]   = active ? Qb[(size_t)qi * HEAD_DIM + d] : 0.0f;
        acc[d] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;

    for (int j0 = 0; j0 < N; j0 += BR) {
        __syncthreads();
        int krow = j0 + threadIdx.x;
        if (krow < N) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                sK[threadIdx.x * HEAD_DIM + d] = Kb[(size_t)krow * HEAD_DIM + d];
                sV[threadIdx.x * HEAD_DIM + d] = Vb[(size_t)krow * HEAD_DIM + d];
            }
        }
        __syncthreads();

        int jmax = (N - j0 < BR) ? (N - j0) : BR;
        for (int jj = 0; jj < jmax; ++jj) {
            int kj = j0 + jj;
            if (causal && kj > qi) break;
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) s += q[d] * sK[jj * HEAD_DIM + d];
            s *= scale;
            float m_new = fmaxf(m, s);
            float corr  = __expf(m - m_new);
            float p     = __expf(s - m_new);
            l = l * corr + p;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * corr + p * sV[jj * HEAD_DIM + d];
            m = m_new;
        }
    }

    if (active) {
        float inv = 1.0f / l;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) Ob[(size_t)qi * HEAD_DIM + d] = acc[d] * inv;
        if (L) L[(size_t)bh * N + qi] = m + logf(l);
    }
}

// ============================================================================
// Flash forward (warp-row) — one warp per query row, head_dim spread across the
// 32 lanes, dot product via warp-shuffle reduction. Keeps registers tiny so it
// scales to large head_dim, but the per-key shuffle makes it slower than the
// thread-row design for head_dim <= 128. This is the layout a Tensor-Core
// (WMMA/MMA) attention would build on.
// ============================================================================

template<int HEAD_DIM>
__global__ void flash_fwd_warp_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    float* __restrict__ L,
                                    int N, float scale, bool causal)
{
    extern __shared__ float smem[];
    float* sK = smem;                       // [TILE * HEAD_DIM]
    float* sV = smem + TILE * HEAD_DIM;     // [TILE * HEAD_DIM]

    const int warps = blockDim.x / WARP;
    int lane = threadIdx.x & (WARP - 1);
    int warp = threadIdx.x >> 5;
    int bh   = blockIdx.y;
    int qi   = blockIdx.x * warps + warp;

    const float* Qb = Q + (size_t)bh * N * HEAD_DIM;
    const float* Kb = K + (size_t)bh * N * HEAD_DIM;
    const float* Vb = V + (size_t)bh * N * HEAD_DIM;
    float*       Ob = O + (size_t)bh * N * HEAD_DIM;

    constexpr int VPT = HEAD_DIM / WARP;    // values per lane
    float q[VPT], acc[VPT];
    #pragma unroll
    for (int t = 0; t < VPT; ++t) {
        int d = lane + t * WARP;
        q[t]   = (qi < N) ? Qb[(size_t)qi * HEAD_DIM + d] : 0.0f;
        acc[t] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;

    for (int j0 = 0; j0 < N; j0 += TILE) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < TILE * HEAD_DIM; idx += blockDim.x) {
            int r = idx / HEAD_DIM, c = idx - r * HEAD_DIM, krow = j0 + r;
            float kk = 0.0f, vv = 0.0f;
            if (krow < N) { kk = Kb[(size_t)krow * HEAD_DIM + c]; vv = Vb[(size_t)krow * HEAD_DIM + c]; }
            sK[idx] = kk; sV[idx] = vv;
        }
        __syncthreads();

        int jmax = (N - j0 < TILE) ? (N - j0) : TILE;
        for (int jj = 0; jj < jmax; ++jj) {
            int kj = j0 + jj;
            if (causal && kj > qi) break;
            float partial = 0.0f;
            #pragma unroll
            for (int t = 0; t < VPT; ++t) partial += q[t] * sK[jj * HEAD_DIM + lane + t * WARP];
            #pragma unroll
            for (int off = WARP / 2; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            float s = __shfl_sync(0xffffffffu, partial, 0) * scale;

            float m_new = fmaxf(m, s);
            float corr  = __expf(m - m_new);
            float p     = __expf(s - m_new);
            l = l * corr + p;
            #pragma unroll
            for (int t = 0; t < VPT; ++t)
                acc[t] = acc[t] * corr + p * sV[jj * HEAD_DIM + lane + t * WARP];
            m = m_new;
        }
    }

    if (qi < N) {
        float inv = 1.0f / l;
        #pragma unroll
        for (int t = 0; t < VPT; ++t) Ob[(size_t)qi * HEAD_DIM + lane + t * WARP] = acc[t] * inv;
        if (L && lane == 0) L[(size_t)bh * N + qi] = m + logf(l);
    }
}

// ============================================================================
// Flash forward — FP16 I/O, FP32 accumulate (thread-row design). Half storage
// halves K/V/Q bandwidth and the shared-memory footprint vs FP32.
// ============================================================================

template<int HEAD_DIM>
__global__ void flash_fwd_fp16_kernel(const __half* __restrict__ Q,
                                      const __half* __restrict__ K,
                                      const __half* __restrict__ V,
                                      __half* __restrict__ O,
                                      float* __restrict__ L,
                                      int N, float scale, bool causal)
{
    extern __shared__ __half smemh[];
    const int BR = blockDim.x;              // query rows per block == K/V tile rows
    __half* sK = smemh;                     // [BR * HEAD_DIM]
    __half* sV = smemh + BR * HEAD_DIM;     // [BR * HEAD_DIM]

    int bh = blockIdx.y;
    const __half* Qb = Q + (size_t)bh * N * HEAD_DIM;
    const __half* Kb = K + (size_t)bh * N * HEAD_DIM;
    const __half* Vb = V + (size_t)bh * N * HEAD_DIM;
    __half*       Ob = O + (size_t)bh * N * HEAD_DIM;

    int  qi     = blockIdx.x * BR + threadIdx.x;
    bool active = qi < N;

    float q[HEAD_DIM], acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q[d]   = active ? __half2float(Qb[(size_t)qi * HEAD_DIM + d]) : 0.0f;
        acc[d] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;

    for (int j0 = 0; j0 < N; j0 += BR) {
        __syncthreads();
        int krow = j0 + threadIdx.x;
        if (krow < N) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                sK[threadIdx.x * HEAD_DIM + d] = Kb[(size_t)krow * HEAD_DIM + d];
                sV[threadIdx.x * HEAD_DIM + d] = Vb[(size_t)krow * HEAD_DIM + d];
            }
        }
        __syncthreads();

        int jmax = (N - j0 < BR) ? (N - j0) : BR;
        for (int jj = 0; jj < jmax; ++jj) {
            int kj = j0 + jj;
            if (causal && kj > qi) break;
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) s += q[d] * __half2float(sK[jj * HEAD_DIM + d]);
            s *= scale;
            float m_new = fmaxf(m, s);
            float corr  = __expf(m - m_new);
            float p     = __expf(s - m_new);
            l = l * corr + p;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d)
                acc[d] = acc[d] * corr + p * __half2float(sV[jj * HEAD_DIM + d]);
            m = m_new;
        }
    }

    if (active) {
        float inv = 1.0f / l;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) Ob[(size_t)qi * HEAD_DIM + d] = __float2half(acc[d] * inv);
        if (L) L[(size_t)bh * N + qi] = m + logf(l);
    }
}

// ============================================================================
// Backward
// ============================================================================

// Delta[bh,i] = sum_d dO[bh,i,d] * O[bh,i,d]
__global__ void bwd_preprocess_kernel(const float* __restrict__ dO,
                                      const float* __restrict__ O,
                                      float* __restrict__ Delta,
                                      int D, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* dOr = dO + (size_t)row * D;
    const float* Or  = O  + (size_t)row * D;
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += dOr[d] * Or[d];
    Delta[row] = s;
}

// dQ_i = scale * sum_j P_ij (dO_i . V_j - Delta_i) K_j   — one warp per query row i
template<int HEAD_DIM>
__global__ void flash_bwd_dq_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    const float* __restrict__ dO,
                                    const float* __restrict__ L,
                                    const float* __restrict__ Delta,
                                    float* __restrict__ dQ,
                                    int N, float scale, bool causal)
{
    const int warps = blockDim.x / WARP;
    int lane = threadIdx.x & (WARP - 1);
    int warp = threadIdx.x >> 5;
    int bh   = blockIdx.y;
    int i    = blockIdx.x * warps + warp;
    if (i >= N) return;

    const float* Qb  = Q  + (size_t)bh * N * HEAD_DIM;
    const float* Kb  = K  + (size_t)bh * N * HEAD_DIM;
    const float* Vb  = V  + (size_t)bh * N * HEAD_DIM;
    const float* dOb = dO + (size_t)bh * N * HEAD_DIM;
    float*       dQb = dQ + (size_t)bh * N * HEAD_DIM;
    float Li = L[(size_t)bh * N + i];
    float Di = Delta[(size_t)bh * N + i];

    constexpr int VPT = HEAD_DIM / WARP;
    float q[VPT], dOi[VPT], dq[VPT];
    #pragma unroll
    for (int t = 0; t < VPT; ++t) {
        int d = lane + t * WARP;
        q[t]   = Qb [(size_t)i * HEAD_DIM + d];
        dOi[t] = dOb[(size_t)i * HEAD_DIM + d];
        dq[t]  = 0.0f;
    }

    int jmax = causal ? i : (N - 1);
    for (int j = 0; j <= jmax; ++j) {
        float ps = 0.0f, pd = 0.0f;
        #pragma unroll
        for (int t = 0; t < VPT; ++t) {
            int d = lane + t * WARP;
            ps += q[t]   * Kb[(size_t)j * HEAD_DIM + d];
            pd += dOi[t] * Vb[(size_t)j * HEAD_DIM + d];
        }
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) {
            ps += __shfl_down_sync(0xffffffffu, ps, off);
            pd += __shfl_down_sync(0xffffffffu, pd, off);
        }
        float s  = __shfl_sync(0xffffffffu, ps, 0) * scale;
        float dp = __shfl_sync(0xffffffffu, pd, 0);
        float p  = __expf(s - Li);
        float ds = p * (dp - Di) * scale;
        #pragma unroll
        for (int t = 0; t < VPT; ++t) dq[t] += ds * Kb[(size_t)j * HEAD_DIM + lane + t * WARP];
    }
    #pragma unroll
    for (int t = 0; t < VPT; ++t) dQb[(size_t)i * HEAD_DIM + lane + t * WARP] = dq[t];
}

// dK_j, dV_j  — one warp per key row j
template<int HEAD_DIM>
__global__ void flash_bwd_dkv_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     const float* __restrict__ dO,
                                     const float* __restrict__ L,
                                     const float* __restrict__ Delta,
                                     float* __restrict__ dK,
                                     float* __restrict__ dV,
                                     int N, float scale, bool causal)
{
    const int warps = blockDim.x / WARP;
    int lane = threadIdx.x & (WARP - 1);
    int warp = threadIdx.x >> 5;
    int bh   = blockIdx.y;
    int j    = blockIdx.x * warps + warp;
    if (j >= N) return;

    const float* Qb  = Q  + (size_t)bh * N * HEAD_DIM;
    const float* Kb  = K  + (size_t)bh * N * HEAD_DIM;
    const float* Vb  = V  + (size_t)bh * N * HEAD_DIM;
    const float* dOb = dO + (size_t)bh * N * HEAD_DIM;
    const float* Lb  = L     + (size_t)bh * N;
    const float* Db  = Delta + (size_t)bh * N;
    float*       dKb = dK + (size_t)bh * N * HEAD_DIM;
    float*       dVb = dV + (size_t)bh * N * HEAD_DIM;

    constexpr int VPT = HEAD_DIM / WARP;
    float k[VPT], v[VPT], dk[VPT], dv[VPT];
    #pragma unroll
    for (int t = 0; t < VPT; ++t) {
        int d = lane + t * WARP;
        k[t]  = Kb[(size_t)j * HEAD_DIM + d];
        v[t]  = Vb[(size_t)j * HEAD_DIM + d];
        dk[t] = 0.0f; dv[t] = 0.0f;
    }

    int imin = causal ? j : 0;
    for (int i = imin; i < N; ++i) {
        float ps = 0.0f, pd = 0.0f;
        #pragma unroll
        for (int t = 0; t < VPT; ++t) {
            int d = lane + t * WARP;
            ps += Qb [(size_t)i * HEAD_DIM + d] * k[t];
            pd += dOb[(size_t)i * HEAD_DIM + d] * v[t];
        }
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) {
            ps += __shfl_down_sync(0xffffffffu, ps, off);
            pd += __shfl_down_sync(0xffffffffu, pd, off);
        }
        float s  = __shfl_sync(0xffffffffu, ps, 0) * scale;
        float dp = __shfl_sync(0xffffffffu, pd, 0);
        float p  = __expf(s - Lb[i]);
        float ds = p * (dp - Db[i]) * scale;
        #pragma unroll
        for (int t = 0; t < VPT; ++t) {
            int d = lane + t * WARP;
            dv[t] += p  * dOb[(size_t)i * HEAD_DIM + d];
            dk[t] += ds * Qb [(size_t)i * HEAD_DIM + d];
        }
    }
    #pragma unroll
    for (int t = 0; t < VPT; ++t) {
        int d = lane + t * WARP;
        dKb[(size_t)j * HEAD_DIM + d] = dk[t];
        dVb[(size_t)j * HEAD_DIM + d] = dv[t];
    }
}

// ============================================================================
// Host launchers
// ============================================================================

void flash_attention_forward(const float* d_Q, const float* d_K, const float* d_V,
                             float* d_O, float* d_L,
                             const FlashAttnConfig& cfg, FlashKernel kernel,
                             cudaStream_t stream)
{
    int   N  = cfg.seq_len, D = cfg.head_dim, BH = cfg.batch * cfg.num_heads;
    float sc = cfg.eff_scale();

    if (kernel == FLASH_THREAD_ROW) {
        int    BR    = (D <= 64) ? 64 : 32;
        size_t shmem = (size_t)2 * BR * D * sizeof(float);
        dim3 block(BR), grid((N + BR - 1) / BR, BH);
        #define CALL_THREAD(HD) flash_fwd_thread_kernel<HD><<<grid, block, shmem, stream>>>( \
            d_Q, d_K, d_V, d_O, d_L, N, sc, cfg.causal)
        DISPATCH_HEAD_DIM(D, CALL_THREAD);
        #undef CALL_THREAD
    } else {
        size_t shmem = (size_t)2 * TILE * D * sizeof(float);
        dim3 block(WPB * WARP), grid((N + WPB - 1) / WPB, BH);
        #define CALL_WARP(HD) flash_fwd_warp_kernel<HD><<<grid, block, shmem, stream>>>( \
            d_Q, d_K, d_V, d_O, d_L, N, sc, cfg.causal)
        DISPATCH_HEAD_DIM(D, CALL_WARP);
        #undef CALL_WARP
    }
    CUDA_CHECK_LAST_ERROR();
}

void flash_attention_forward_fp16(const __half* d_Q, const __half* d_K, const __half* d_V,
                                  __half* d_O, float* d_L,
                                  const FlashAttnConfig& cfg, cudaStream_t stream)
{
    int   N  = cfg.seq_len, D = cfg.head_dim, BH = cfg.batch * cfg.num_heads;
    float sc = cfg.eff_scale();
    int    BR    = (D <= 64) ? 64 : 32;
    size_t shmem = (size_t)2 * BR * D * sizeof(__half);
    dim3 block(BR), grid((N + BR - 1) / BR, BH);
    #define CALL_H(HD) flash_fwd_fp16_kernel<HD><<<grid, block, shmem, stream>>>( \
        d_Q, d_K, d_V, d_O, d_L, N, sc, cfg.causal)
    DISPATCH_HEAD_DIM(D, CALL_H);
    #undef CALL_H
    CUDA_CHECK_LAST_ERROR();
}

void flash_attention_backward(const float* d_Q, const float* d_K, const float* d_V,
                              const float* d_O, const float* d_dO, const float* d_L,
                              float* d_dQ, float* d_dK, float* d_dV,
                              float* d_workspace,
                              const FlashAttnConfig& cfg, cudaStream_t stream)
{
    int   N  = cfg.seq_len, D = cfg.head_dim, BH = cfg.batch * cfg.num_heads;
    float sc = cfg.eff_scale();
    int   rows = BH * N;

    bwd_preprocess_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(d_dO, d_O, d_workspace, D, rows);

    dim3 block(WPB * WARP), grid((N + WPB - 1) / WPB, BH);
    #define CALL_DQ(HD) flash_bwd_dq_kernel<HD><<<grid, block, 0, stream>>>( \
        d_Q, d_K, d_V, d_dO, d_L, d_workspace, d_dQ, N, sc, cfg.causal)
    DISPATCH_HEAD_DIM(D, CALL_DQ);
    #undef CALL_DQ
    #define CALL_DKV(HD) flash_bwd_dkv_kernel<HD><<<grid, block, 0, stream>>>( \
        d_Q, d_K, d_V, d_dO, d_L, d_workspace, d_dK, d_dV, N, sc, cfg.causal)
    DISPATCH_HEAD_DIM(D, CALL_DKV);
    #undef CALL_DKV
    CUDA_CHECK_LAST_ERROR();
}
