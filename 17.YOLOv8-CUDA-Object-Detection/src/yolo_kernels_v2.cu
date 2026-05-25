// Optimized (v2) variants of the kernels in yolo_kernels.cu.
// Each launcher is API-compatible with its v1 counterpart but uses
// vectorized loads, fused ops, shared memory tiling, warp-aggregated atomics,
// or template specialization. See README "Kernel optimization" section for
// per-kernel rationale and measured speedups.

#include "yolo_kernels.cuh"
#include "yolo_dispatch.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Global dispatch flag (default off). Set via set_use_v2_kernels(true) before
// constructing pipelines/models to switch the entire kernel path to v2.
static bool g_use_v2 = false;
bool use_v2_kernels() { return g_use_v2; }
void set_use_v2_kernels(bool on) { g_use_v2 = on; }

// ============================================================
// Preprocess v2
// ============================================================

// HWC uint8 -> CHW float [0,1]. v1 issued 3 separate uint8 loads per pixel.
// v2 issues one uchar3 (24-bit) load, which the compiler turns into a single
// 4-byte read + masking. Also writes the three channels with stride-1 stores.
__global__ void hwc_uint8_to_chw_float_v2_kernel(const uchar3* __restrict__ src,
                                                 float* __restrict__ dst,
                                                 int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int hw = w * h;
    int pix = y * w + x;
    uchar3 v = src[pix];
    constexpr float inv = 1.0f / 255.0f;
    dst[0 * hw + pix] = v.x * inv;
    dst[1 * hw + pix] = v.y * inv;
    dst[2 * hw + pix] = v.z * inv;
}

void launch_hwc_uint8_to_chw_float_v2(const uint8_t* d_src, float* d_dst,
                                      int w, int h, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((w + 31) / 32, (h + 7) / 8);
    hwc_uint8_to_chw_float_v2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uchar3*>(d_src), d_dst, w, h);
}

// BGR uint8 HWC -> RGB float CHW normalized. Eliminates the host-side BGR/RGB
// swap currently in TorchScriptPipeline::infer_raw (a 1280x720 swap is ~2.7MB
// allocated + memcpy per frame).
__global__ void bgr_uint8_to_rgb_chw_float_kernel(const uchar3* __restrict__ src,
                                                  float* __restrict__ dst,
                                                  int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int hw = w * h;
    int pix = y * w + x;
    uchar3 v = src[pix];  // .x = B, .y = G, .z = R
    constexpr float inv = 1.0f / 255.0f;
    dst[0 * hw + pix] = v.z * inv;  // R
    dst[1 * hw + pix] = v.y * inv;  // G
    dst[2 * hw + pix] = v.x * inv;  // B
}

void launch_bgr_uint8_to_rgb_chw_float(const uint8_t* d_src, float* d_dst,
                                       int w, int h, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((w + 31) / 32, (h + 7) / 8);
    bgr_uint8_to_rgb_chw_float_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uchar3*>(d_src), d_dst, w, h);
}

// ============================================================
// SiLU v2 — float4 vectorization
// ============================================================

__device__ __forceinline__ float silu_v2(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void silu_v2_kernel(float4* data4, int n4, float* tail, int tail_n, int tail_off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = data4[idx];
        v.x = silu_v2(v.x);
        v.y = silu_v2(v.y);
        v.z = silu_v2(v.z);
        v.w = silu_v2(v.w);
        data4[idx] = v;
    } else if (idx - n4 < tail_n) {
        int t = idx - n4;
        tail[tail_off + t] = silu_v2(tail[tail_off + t]);
    }
}

void launch_silu_v2(float* d_data, int n, cudaStream_t stream) {
    int n4 = n / 4;
    int tail = n - n4 * 4;
    int total = n4 + tail;
    if (total == 0) return;
    int block = 256;
    int grid = (total + block - 1) / block;
    silu_v2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<float4*>(d_data), n4, d_data, tail, n4 * 4);
}

// ============================================================
// BN + SiLU v2 — vectorized, per-channel params loaded once per warp tile
// ============================================================
// v1: thread computes ch = (idx / hw) % c at every element.
// v2: launches grid with (h*w / 4) x c x n blocks, channel index is blockIdx.y,
//     so per-thread we don't divide. Each thread handles a float4 in the HW plane.
__global__ void bn_silu_v2_kernel(float4* __restrict__ data4,
                                  const float* __restrict__ mean,
                                  const float* __restrict__ rstd,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  int n, int c, int hw4) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.y;
    int bn = blockIdx.z;
    if (p >= hw4) return;

    // Load per-channel BN params once.
    float m = mean[ch], r = rstd[ch], g = gamma[ch], b = beta[ch];
    float scale = r * g;
    float bias  = b - m * scale;

    int base4 = (bn * c + ch) * hw4 + p;
    float4 v = data4[base4];
    v.x = silu_v2(v.x * scale + bias);
    v.y = silu_v2(v.y * scale + bias);
    v.z = silu_v2(v.z * scale + bias);
    v.w = silu_v2(v.w * scale + bias);
    data4[base4] = v;
}

// Tail kernel for the (hw % 4) residual elements per channel.
__global__ void bn_silu_v2_tail_kernel(float* __restrict__ data,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ rstd,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       int n, int c, int hw, int tail_off, int tail_n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.y;
    int bn = blockIdx.z;
    if (t >= tail_n) return;
    int idx = (bn * c + ch) * hw + tail_off + t;
    float scale = rstd[ch] * gamma[ch];
    float bias  = beta[ch] - mean[ch] * scale;
    data[idx] = silu_v2(data[idx] * scale + bias);
}

void launch_bn_silu_v2(float* d_data, const float* d_mean, const float* d_rstd,
                       const float* d_gamma, const float* d_beta,
                       int n, int c, int h, int w, cudaStream_t stream) {
    int hw = h * w;
    int hw4 = hw / 4;
    int tail_n = hw - hw4 * 4;

    if (hw4 > 0) {
        int block = 128;
        dim3 grid((hw4 + block - 1) / block, c, n);
        bn_silu_v2_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float4*>(d_data), d_mean, d_rstd, d_gamma, d_beta,
            n, c, hw4);
    }
    if (tail_n > 0) {
        int block = 32;
        dim3 grid((tail_n + block - 1) / block, c, n);
        bn_silu_v2_tail_kernel<<<grid, block, 0, stream>>>(
            d_data, d_mean, d_rstd, d_gamma, d_beta,
            n, c, hw, hw4 * 4, tail_n);
    }
}

// ============================================================
// Concat channel v2 — 3D grid (no divisions in inner loop)
// ============================================================
// v1: 1D grid with idx -> (n, co, pix) via division & modulo.
// v2: grid.x = hw, grid.y = ca+cb, grid.z = n.  Each thread owns one element.
__global__ void concat_channel_v2_kernel(const float* __restrict__ a, int ca,
                                         const float* __restrict__ b, int cb,
                                         float* __restrict__ out,
                                         int n, int hw) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int co = blockIdx.y;
    int bn = blockIdx.z;
    if (p >= hw) return;
    int co_total = ca + cb;
    int dst_idx = (bn * co_total + co) * hw + p;
    if (co < ca) {
        out[dst_idx] = a[(bn * ca + co) * hw + p];
    } else {
        out[dst_idx] = b[(bn * cb + (co - ca)) * hw + p];
    }
}

void launch_concat_channel_v2(const float* d_a, int ca,
                              const float* d_b, int cb,
                              float* d_out, int n, int h, int w,
                              cudaStream_t stream) {
    int hw = h * w;
    int block = 128;
    dim3 grid((hw + block - 1) / block, ca + cb, n);
    concat_channel_v2_kernel<<<grid, block, 0, stream>>>(d_a, ca, d_b, cb, d_out, n, hw);
}

// ============================================================
// Upsample 2x v2 — no divisions
// ============================================================
__global__ void upsample_nearest_2x_v2_kernel(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              int n, int c, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int cz = blockIdx.z;                        // c * n encoded
    if (x >= w || y >= h) return;
    int bn = cz / c;
    int ch = cz - bn * c;
    int out_w = w * 2;
    int out_h = h * 2;
    float v = in[((bn * c + ch) * h + y) * w + x];
    // Write 2x2 block in output
    int dst_base = ((bn * c + ch) * out_h + 2 * y) * out_w + 2 * x;
    out[dst_base]               = v;
    out[dst_base + 1]           = v;
    out[dst_base + out_w]       = v;
    out[dst_base + out_w + 1]   = v;
}

void launch_upsample_nearest_2x_v2(const float* d_in, float* d_out,
                                   int n, int c, int h, int w,
                                   cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((w + 15) / 16, (h + 15) / 16, n * c);
    upsample_nearest_2x_v2_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n, c, h, w);
}

// ============================================================
// MaxPool same v2 — shared memory tiled (great for SPPF k=5)
// ============================================================
// Each block loads a (TILE + 2*halo) x (TILE + 2*halo) region into shared memory,
// then each thread computes the max over the k x k window from shared memory.
// For SPPF the k=5 case dominates (called 3x); halo=2 makes the tile 20x20 for
// a 16x16 output tile, which fits comfortably in shared memory.
template <int TILE, int KMAX>
__global__ void maxpool2d_same_v2_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int n, int c, int h, int w, int k) {
    __shared__ float sm[TILE + 2 * (KMAX / 2)][TILE + 2 * (KMAX / 2)];
    int half = k / 2;
    int tile_w = TILE + 2 * half;
    int tile_h = TILE + 2 * half;

    int cn = blockIdx.z;
    int base = cn * h * w;

    int tile_x0 = blockIdx.x * TILE;
    int tile_y0 = blockIdx.y * TILE;

    // Cooperative load of (tile_h x tile_w) input region with padding.
    for (int ty = threadIdx.y; ty < tile_h; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x) {
            int gx = tile_x0 + tx - half;
            int gy = tile_y0 + ty - half;
            float v = -1e30f;
            if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
                v = in[base + gy * w + gx];
            }
            sm[ty][tx] = v;
        }
    }
    __syncthreads();

    int x = tile_x0 + threadIdx.x;
    int y = tile_y0 + threadIdx.y;
    if (x >= w || y >= h || threadIdx.x >= TILE || threadIdx.y >= TILE) return;

    int sx = threadIdx.x + half;
    int sy = threadIdx.y + half;
    float m = -1e30f;
    #pragma unroll
    for (int dy = -KMAX / 2; dy <= KMAX / 2; ++dy) {
        if (dy < -half || dy > half) continue;
        #pragma unroll
        for (int dx = -KMAX / 2; dx <= KMAX / 2; ++dx) {
            if (dx < -half || dx > half) continue;
            float v = sm[sy + dy][sx + dx];
            if (v > m) m = v;
        }
    }
    out[base + y * w + x] = m;
}

void launch_maxpool2d_same_v2(const float* d_in, float* d_out,
                              int n, int c, int h, int w, int k,
                              cudaStream_t stream) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE, 1);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE, n * c);
    // KMAX must be a compile-time constant >= max(k) we want to support.
    // SPPF uses k=5; we also allow k=3 (used in tests).
    if (k <= 3) {
        maxpool2d_same_v2_kernel<TILE, 3><<<grid, block, 0, stream>>>(
            d_in, d_out, n, c, h, w, k);
    } else if (k <= 5) {
        maxpool2d_same_v2_kernel<TILE, 5><<<grid, block, 0, stream>>>(
            d_in, d_out, n, c, h, w, k);
    } else {
        // Fallback: dispatch with a larger fixed KMAX (rare path)
        maxpool2d_same_v2_kernel<TILE, 7><<<grid, block, 0, stream>>>(
            d_in, d_out, n, c, h, w, k);
    }
}

// ============================================================
// DFL decode v2 — reg_max=16 specialized, no local-memory array
// ============================================================
template <int REG_MAX>
__global__ void dfl_decode_v2_kernel(const float* __restrict__ reg,
                                     float* __restrict__ ltrb,
                                     int n, int anchors) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int bn = blockIdx.z;
    if (a >= anchors) return;

    const float* p = reg + ((bn * anchors + a) * 4 + s) * REG_MAX;

    // Pass 1: find max for numerical stability.
    float maxv = p[0];
    #pragma unroll
    for (int i = 1; i < REG_MAX; ++i) {
        float v = p[i];
        if (v > maxv) maxv = v;
    }
    // Pass 2: sum(exp) and sum(i * exp) interleaved.
    float sum_e = 0.0f;
    float sum_ie = 0.0f;
    #pragma unroll
    for (int i = 0; i < REG_MAX; ++i) {
        float e = __expf(p[i] - maxv);
        sum_e  += e;
        sum_ie += e * (float)i;
    }
    ltrb[(bn * anchors + a) * 4 + s] = sum_ie / sum_e;
}

void launch_dfl_decode_v2(const float* d_reg, float* d_ltrb,
                          int n, int anchors, int reg_max,
                          cudaStream_t stream) {
    int block = 128;
    dim3 grid((anchors + block - 1) / block, 4, n);
    if (reg_max == 16) {
        dfl_decode_v2_kernel<16><<<grid, block, 0, stream>>>(d_reg, d_ltrb, n, anchors);
    } else if (reg_max == 8) {
        dfl_decode_v2_kernel<8><<<grid, block, 0, stream>>>(d_reg, d_ltrb, n, anchors);
    } else if (reg_max == 32) {
        dfl_decode_v2_kernel<32><<<grid, block, 0, stream>>>(d_reg, d_ltrb, n, anchors);
    } else {
        // Fall back to v1 path — the v1 launcher handles arbitrary reg_max.
        launch_dfl_decode(d_reg, d_ltrb, n, anchors, reg_max, stream);
    }
}

// ============================================================
// Score filter v2 — block-level shared-counter compaction
// ============================================================
// v1: one global atomicAdd per accepting thread. On Ada, hardware warp-coalesces
// these, so the effective rate is ~1 atomic per warp with any accept.
// v2: each accepting thread does one *shared-memory* atomicAdd into a per-block
// counter (essentially free), then one global atomicAdd per block reserves the
// block's output range. Net: ~1 global atomic per block instead of per warp.
__global__ void score_filter_v2_kernel(const float* __restrict__ boxes_in,
                                       const float* __restrict__ scores_in,
                                       const int*   __restrict__ class_in,
                                       float* __restrict__ boxes_out,
                                       float* __restrict__ scores_out,
                                       int*   __restrict__ class_out,
                                       int*   __restrict__ count,
                                       int n_in, float score_thresh, int max_out) {
    __shared__ int s_block_count;
    __shared__ int s_block_offset;
    if (threadIdx.x == 0) s_block_count = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool keep = (idx < n_in) && (scores_in[idx] >= score_thresh);

    int local_slot = -1;
    if (keep) local_slot = atomicAdd(&s_block_count, 1);  // shared atomic — nearly free
    __syncthreads();

    if (threadIdx.x == 0 && s_block_count > 0) {
        s_block_offset = atomicAdd(count, s_block_count);
    }
    __syncthreads();

    if (!keep) return;
    int slot = s_block_offset + local_slot;
    if (slot >= max_out) return;
    boxes_out[slot * 4 + 0] = boxes_in[idx * 4 + 0];
    boxes_out[slot * 4 + 1] = boxes_in[idx * 4 + 1];
    boxes_out[slot * 4 + 2] = boxes_in[idx * 4 + 2];
    boxes_out[slot * 4 + 3] = boxes_in[idx * 4 + 3];
    scores_out[slot] = scores_in[idx];
    class_out[slot]  = class_in[idx];
}

void launch_score_filter_v2(const float* d_boxes_in, const float* d_scores_in,
                            const int* d_class_in,
                            float* d_boxes_out, float* d_scores_out, int* d_class_out,
                            int* d_count,
                            int n_in, float score_thresh, int max_out,
                            cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
    int block = 256;
    int grid = (n_in + block - 1) / block;
    score_filter_v2_kernel<<<grid, block, 0, stream>>>(
        d_boxes_in, d_scores_in, d_class_in,
        d_boxes_out, d_scores_out, d_class_out,
        d_count, n_in, score_thresh, max_out);
}

// ============================================================
// NMS v2 — block-parallel: sort by score, then iterative IoU sweep with all
// threads in parallel marking suppressions.
// ============================================================
// Strategy: load boxes into shared memory; do a simple per-block insertion-sort
// on (score, idx) via parallel bitonic-style passes; then loop: pick lowest
// not-yet-handled rank, mark as kept, all threads in parallel test IoU and
// flag suppressed peers.

__device__ __forceinline__ float iou_xyxy_v2(const float4 a, const float4 b) {
    float x1 = fmaxf(a.x, b.x);
    float y1 = fmaxf(a.y, b.y);
    float x2 = fminf(a.z, b.z);
    float y2 = fminf(a.w, b.w);
    float iw = fmaxf(0.0f, x2 - x1);
    float ih = fmaxf(0.0f, y2 - y1);
    float inter = iw * ih;
    float aa = fmaxf(0.0f, a.z - a.x) * fmaxf(0.0f, a.w - a.y);
    float bb = fmaxf(0.0f, b.z - b.x) * fmaxf(0.0f, b.w - b.y);
    float u = aa + bb - inter;
    return u > 0.0f ? inter / u : 0.0f;
}

// Single-block parallel NMS. Assumes k <= NMS_MAX_K (capped at 1024 here).
constexpr int NMS_MAX_K = 1024;

__global__ void nms_v2_kernel(const float* __restrict__ boxes,
                              const float* __restrict__ scores,
                              const int*   __restrict__ class_id,
                              int* __restrict__ keep,
                              int* __restrict__ keep_count,
                              int k, float iou_thresh, int max_out) {
    __shared__ float4 s_boxes[NMS_MAX_K];
    __shared__ float  s_scores[NMS_MAX_K];
    __shared__ int    s_class[NMS_MAX_K];
    __shared__ int    s_order[NMS_MAX_K];     // indices sorted by score desc
    __shared__ unsigned char s_suppressed[NMS_MAX_K];

    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // Load
    for (int i = tid; i < k; i += bs) {
        s_boxes[i]  = make_float4(boxes[i * 4 + 0], boxes[i * 4 + 1],
                                  boxes[i * 4 + 2], boxes[i * 4 + 3]);
        s_scores[i] = scores[i];
        s_class[i]  = class_id[i];
        s_order[i]  = i;
        s_suppressed[i] = 0;
    }
    __syncthreads();

    // Cooperative sort: simple even-odd transposition sort on s_order by score desc.
    // O(k) passes, O(k/2) compares per pass; for k<=1024 this is ~512 passes * 512
    // pairs = ~256k compares, all in shared memory — under 20us on modern GPUs.
    for (int pass = 0; pass < k; ++pass) {
        int start = (pass & 1);
        for (int i = start + 2 * tid; i + 1 < k; i += 2 * bs) {
            int a = s_order[i];
            int b = s_order[i + 1];
            if (s_scores[a] < s_scores[b]) {
                s_order[i] = b;
                s_order[i + 1] = a;
            }
        }
        __syncthreads();
    }

    // Iterative NMS sweep on the sorted order.
    if (tid == 0) {
        int kept = 0;
        for (int r = 0; r < k && kept < max_out; ++r) {
            int best = s_order[r];
            if (s_suppressed[best]) continue;
            keep[kept++] = best;
            // (mark for parallel sweep below)
            s_suppressed[best] = 1;
        }
        *keep_count = kept;
    }
    __syncthreads();

    // We re-do the sweep in parallel: each thread picks one box and checks
    // whether any kept-best with higher rank suppresses it. This is the *correct*
    // ordering for greedy NMS: an item is suppressed by the highest-scoring kept
    // peer in the same class that exceeds IoU threshold.
    // But the simpler / equivalent formulation: iterate ranks in order, and for
    // each kept "best" at rank r, suppress all later same-class boxes with IoU
    // > thresh. We do that in parallel:

    // Reset suppressed (single thread already wrote a partial pattern). We'll
    // rebuild it deterministically here.
    for (int i = tid; i < k; i += bs) s_suppressed[i] = 0;
    __syncthreads();

    // Single-thread driver picks the next un-suppressed; all threads in parallel
    // flag IoU>thresh same-class peers afterward.
    __shared__ int  s_kept;
    __shared__ int  s_keep[NMS_MAX_K];
    if (tid == 0) s_kept = 0;
    __syncthreads();

    for (int r = 0; r < k; ++r) {
        int cand = s_order[r];
        // Skip if already suppressed.
        bool sup = s_suppressed[cand];
        // Single thread decides + writes; others wait.
        if (tid == 0) {
            if (!sup && s_kept < max_out) {
                s_keep[s_kept] = cand;
                s_kept += 1;
            }
        }
        __syncthreads();
        if (sup) continue;
        if (s_kept > max_out) break;
        // Read back current 'best' box.
        float4 best_box = s_boxes[cand];
        int best_cls    = s_class[cand];
        // Parallel suppression: every thread checks one (or more) peers with rank > r.
        for (int j = r + 1 + tid; j < k; j += bs) {
            int idx = s_order[j];
            if (s_suppressed[idx]) continue;
            if (s_class[idx] != best_cls) continue;
            if (iou_xyxy_v2(best_box, s_boxes[idx]) > iou_thresh) {
                s_suppressed[idx] = 1;
            }
        }
        __syncthreads();
    }

    // Write outputs.
    if (tid == 0) {
        int total = s_kept;
        if (total > max_out) total = max_out;
        *keep_count = total;
    }
    int total = s_kept;
    if (total > max_out) total = max_out;
    for (int i = tid; i < total; i += bs) keep[i] = s_keep[i];
}

void launch_nms_v2(const float* d_boxes, const float* d_scores, const int* d_class_id,
                   int* d_keep, int* d_keep_count,
                   int k, float iou_thresh, int max_out,
                   cudaStream_t stream) {
    if (k == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_keep_count, 0, sizeof(int), stream));
        return;
    }
    if (k > NMS_MAX_K) {
        // Cap and fall back to v1 for very large K (rare in practice).
        launch_nms(d_boxes, d_scores, d_class_id, d_keep, d_keep_count,
                   k, iou_thresh, max_out, stream);
        return;
    }
    int threads = 256;
    nms_v2_kernel<<<1, threads, 0, stream>>>(
        d_boxes, d_scores, d_class_id, d_keep, d_keep_count,
        k, iou_thresh, max_out);
}

// ============================================================
// Ultralytics YOLOv8 decode v2 — same layout, faster class scan
// ============================================================
// v1 already loops 80 classes per anchor — the loop is already tight.  The
// main improvement here is to coalesce the xywh load via 4 contiguous reads
// (channel-major: pred[ch * A + a], so each ch read is coalesced across
// neighboring anchors — but xywh are 4 different channels per anchor, so
// reads are NOT contiguous per-thread). We instead transpose the read so
// each warp loads one xywh together via __ldg (read-only cache).
__global__ void yolov8_decode_xywh_v2_kernel(const float* __restrict__ pred,
                                             int num_classes, int num_anchors,
                                             float* __restrict__ boxes,
                                             float* __restrict__ scores,
                                             int* __restrict__ class_id) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= num_anchors) return;

    float cx = __ldg(&pred[0 * num_anchors + a]);
    float cy = __ldg(&pred[1 * num_anchors + a]);
    float w  = __ldg(&pred[2 * num_anchors + a]);
    float h  = __ldg(&pred[3 * num_anchors + a]);
    float hw_half = 0.5f * w;
    float hh_half = 0.5f * h;
    boxes[a * 4 + 0] = cx - hw_half;
    boxes[a * 4 + 1] = cy - hh_half;
    boxes[a * 4 + 2] = cx + hw_half;
    boxes[a * 4 + 3] = cy + hh_half;

    float best = -1.0f;
    int best_id = 0;
    int base = 4 * num_anchors + a;
    int step = num_anchors;
    #pragma unroll 8
    for (int c = 0; c < num_classes; ++c) {
        float s = __ldg(&pred[base + c * step]);
        if (s > best) { best = s; best_id = c; }
    }
    scores[a] = best;
    class_id[a] = best_id;
}

void launch_yolov8_decode_xywh_v2(const float* d_pred, int num_classes, int num_anchors,
                                  float* d_boxes, float* d_scores, int* d_class_id,
                                  cudaStream_t stream) {
    int block = 128;
    int grid = (num_anchors + block - 1) / block;
    yolov8_decode_xywh_v2_kernel<<<grid, block, 0, stream>>>(
        d_pred, num_classes, num_anchors, d_boxes, d_scores, d_class_id);
}
