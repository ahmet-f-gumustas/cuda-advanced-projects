#include "yolo_kernels.cuh"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// ============================================================
// Preprocessing
// ============================================================

LetterboxParams compute_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
    float r = fminf((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)roundf(src_w * r);
    int new_h = (int)roundf(src_h * r);
    LetterboxParams p;
    p.scale = r;
    p.pad_x = (dst_w - new_w) / 2;
    p.pad_y = (dst_h - new_h) / 2;
    return p;
}

__global__ void letterbox_kernel(const uint8_t* __restrict__ src, int src_w, int src_h,
                                 uint8_t* __restrict__ dst, int dst_w, int dst_h,
                                 float inv_scale, int pad_x, int pad_y,
                                 int new_w, int new_h, uint8_t pad_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int local_x = x - pad_x;
    int local_y = y - pad_y;
    uint8_t r, g, b;
    if (local_x < 0 || local_x >= new_w || local_y < 0 || local_y >= new_h) {
        r = g = b = pad_value;
    } else {
        // Map dst pixel back to src coords (nearest neighbor — fast preprocessing)
        int sx = min((int)(local_x * inv_scale), src_w - 1);
        int sy = min((int)(local_y * inv_scale), src_h - 1);
        int sidx = (sy * src_w + sx) * 3;
        r = src[sidx + 0];
        g = src[sidx + 1];
        b = src[sidx + 2];
    }
    int didx = (y * dst_w + x) * 3;
    dst[didx + 0] = r;
    dst[didx + 1] = g;
    dst[didx + 2] = b;
}

void launch_letterbox_uint8(const uint8_t* d_src, int src_w, int src_h,
                            uint8_t* d_dst, int dst_w, int dst_h,
                            const LetterboxParams& p, uint8_t pad_value,
                            cudaStream_t stream) {
    int new_w = (int)roundf(src_w * p.scale);
    int new_h = (int)roundf(src_h * p.scale);
    float inv_scale = 1.0f / p.scale;
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    letterbox_kernel<<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, d_dst, dst_w, dst_h,
        inv_scale, p.pad_x, p.pad_y, new_w, new_h, pad_value);
}

__global__ void hwc_uint8_to_chw_float_kernel(const uint8_t* __restrict__ src,
                                              float* __restrict__ dst,
                                              int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int hw = w * h;
    int sidx = (y * w + x) * 3;
    int didx = y * w + x;
    dst[0 * hw + didx] = src[sidx + 0] / 255.0f;
    dst[1 * hw + didx] = src[sidx + 1] / 255.0f;
    dst[2 * hw + didx] = src[sidx + 2] / 255.0f;
}

void launch_hwc_uint8_to_chw_float(const uint8_t* d_src, float* d_dst,
                                   int w, int h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    hwc_uint8_to_chw_float_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, w, h);
}

// ============================================================
// SiLU and fused BN+SiLU
// ============================================================

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void silu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = silu(data[idx]);
}

void launch_silu(float* d_data, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_kernel<<<grid, block, 0, stream>>>(d_data, n);
}

__global__ void bn_silu_kernel(float* __restrict__ data,
                               const float* __restrict__ mean,
                               const float* __restrict__ rstd,
                               const float* __restrict__ gamma,
                               const float* __restrict__ beta,
                               int n, int c, int hw) {
    int total = n * c * hw;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ch = (idx / hw) % c;
    float v = (data[idx] - mean[ch]) * rstd[ch] * gamma[ch] + beta[ch];
    data[idx] = silu(v);
}

void launch_bn_silu(float* d_data, const float* d_mean, const float* d_rstd,
                    const float* d_gamma, const float* d_beta,
                    int n, int c, int h, int w, cudaStream_t stream) {
    int total = n * c * h * w;
    int block = 256;
    int grid = (total + block - 1) / block;
    bn_silu_kernel<<<grid, block, 0, stream>>>(d_data, d_mean, d_rstd,
                                               d_gamma, d_beta, n, c, h * w);
}

// ============================================================
// Concat / slice along channel
// ============================================================

__global__ void concat_channel_kernel(const float* __restrict__ a, int ca,
                                      const float* __restrict__ b, int cb,
                                      float* __restrict__ out, int n, int hw) {
    int total = n * (ca + cb) * hw;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int co = (idx / hw) % (ca + cb);
    int bn = idx / (hw * (ca + cb));
    int pix = idx % hw;
    if (co < ca) {
        out[idx] = a[(bn * ca + co) * hw + pix];
    } else {
        out[idx] = b[(bn * cb + (co - ca)) * hw + pix];
    }
}

void launch_concat_channel(const float* d_a, int ca,
                           const float* d_b, int cb,
                           float* d_out, int n, int h, int w,
                           cudaStream_t stream) {
    int total = n * (ca + cb) * h * w;
    int block = 256;
    int grid = (total + block - 1) / block;
    concat_channel_kernel<<<grid, block, 0, stream>>>(d_a, ca, d_b, cb, d_out, n, h * w);
}

__global__ void slice_channel_kernel(const float* __restrict__ in, int c_total,
                                     float* __restrict__ out, int c0, int c1,
                                     int n, int hw) {
    int co_n = c1 - c0;
    int total = n * co_n * hw;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int co = (idx / hw) % co_n;
    int bn = idx / (hw * co_n);
    int pix = idx % hw;
    out[idx] = in[(bn * c_total + (co + c0)) * hw + pix];
}

void launch_slice_channel(const float* d_in, int c_total,
                          float* d_out, int c0, int c1,
                          int n, int h, int w, cudaStream_t stream) {
    int co_n = c1 - c0;
    int total = n * co_n * h * w;
    int block = 256;
    int grid = (total + block - 1) / block;
    slice_channel_kernel<<<grid, block, 0, stream>>>(d_in, c_total, d_out, c0, c1,
                                                     n, h * w);
}

// ============================================================
// 2x nearest upsample
// ============================================================

__global__ void upsample_nearest_2x_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int n, int c, int h, int w) {
    int out_w = w * 2;
    int out_h = h * 2;
    int total = n * c * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int x = idx % out_w;
    int y = (idx / out_w) % out_h;
    int ch = (idx / (out_w * out_h)) % c;
    int bn = idx / (c * out_w * out_h);
    int sx = x / 2;
    int sy = y / 2;
    out[idx] = in[((bn * c + ch) * h + sy) * w + sx];
}

void launch_upsample_nearest_2x(const float* d_in, float* d_out,
                                int n, int c, int h, int w,
                                cudaStream_t stream) {
    int total = n * c * (h * 2) * (w * 2);
    int block = 256;
    int grid = (total + block - 1) / block;
    upsample_nearest_2x_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n, c, h, w);
}

// ============================================================
// MaxPool 2D (kernel k, stride 1, padding k/2 — "same")
// ============================================================

__global__ void maxpool2d_same_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int n, int c, int h, int w, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int cn = blockIdx.z;  // c * n
    if (x >= w || y >= h) return;
    int half = k / 2;
    int base = cn * h * w;
    float m = -1e30f;
    for (int dy = -half; dy <= half; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -half; dx <= half; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            float v = in[base + yy * w + xx];
            if (v > m) m = v;
        }
    }
    out[base + y * w + x] = m;
}

void launch_maxpool2d_same(const float* d_in, float* d_out,
                           int n, int c, int h, int w, int k,
                           cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((w + 15) / 16, (h + 15) / 16, n * c);
    maxpool2d_same_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n, c, h, w, k);
}

// ============================================================
// DFL decode
// ============================================================

__global__ void dfl_decode_kernel(const float* __restrict__ reg,
                                  float* __restrict__ ltrb,
                                  int n, int anchors, int reg_max) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;  // anchor index
    int s = blockIdx.y;                              // side index 0..3
    int bn = blockIdx.z;
    if (a >= anchors) return;

    // reg layout: [N, anchors, 4, reg_max], side s at stride reg_max
    const float* p = reg + ((bn * anchors + a) * 4 + s) * reg_max;
    // softmax + expected value
    float maxv = p[0];
    for (int i = 1; i < reg_max; ++i) {
        if (p[i] > maxv) maxv = p[i];
    }
    float sum = 0.0f;
    // reg_max is typically 16 — fits in registers
    float exps[64];
    for (int i = 0; i < reg_max; ++i) {
        exps[i] = __expf(p[i] - maxv);
        sum += exps[i];
    }
    float val = 0.0f;
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < reg_max; ++i) {
        val += (exps[i] * inv_sum) * (float)i;
    }
    ltrb[(bn * anchors + a) * 4 + s] = val;
}

void launch_dfl_decode(const float* d_reg, float* d_ltrb,
                       int n, int anchors, int reg_max,
                       cudaStream_t stream) {
    int block = 128;
    dim3 grid((anchors + block - 1) / block, 4, n);
    dfl_decode_kernel<<<grid, block, 0, stream>>>(d_reg, d_ltrb, n, anchors, reg_max);
}

// ============================================================
// Anchor grid construction (host-side)
// ============================================================

void build_anchor_grid(int feat_h, int feat_w, int stride,
                       float* out_xy_cpu, float* out_stride_cpu, int offset) {
    int idx = offset;
    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            out_xy_cpu[idx * 2 + 0] = (float)x + 0.5f;
            out_xy_cpu[idx * 2 + 1] = (float)y + 0.5f;
            out_stride_cpu[idx] = (float)stride;
            idx++;
        }
    }
}

// ============================================================
// Decode predictions (anchor-free YOLOv8 head)
// ============================================================

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void decode_predictions_kernel(const float* __restrict__ ltrb,
                                          const float* __restrict__ cls,
                                          const float* __restrict__ anchor_xy,
                                          const float* __restrict__ anchor_stride,
                                          float* __restrict__ boxes,
                                          float* __restrict__ scores,
                                          int* __restrict__ class_id,
                                          int n, int anchors, int num_classes) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int bn = blockIdx.y;
    if (a >= anchors) return;

    // ltrb in feature-grid units (already DFL-decoded)
    float l = ltrb[(bn * anchors + a) * 4 + 0];
    float t = ltrb[(bn * anchors + a) * 4 + 1];
    float r = ltrb[(bn * anchors + a) * 4 + 2];
    float b = ltrb[(bn * anchors + a) * 4 + 3];

    float ax = anchor_xy[a * 2 + 0];
    float ay = anchor_xy[a * 2 + 1];
    float s = anchor_stride[a];

    float x1 = (ax - l) * s;
    float y1 = (ay - t) * s;
    float x2 = (ax + r) * s;
    float y2 = (ay + b) * s;

    boxes[(bn * anchors + a) * 4 + 0] = x1;
    boxes[(bn * anchors + a) * 4 + 1] = y1;
    boxes[(bn * anchors + a) * 4 + 2] = x2;
    boxes[(bn * anchors + a) * 4 + 3] = y2;

    // class scores (sigmoid)
    const float* clsp = cls + (bn * anchors + a) * num_classes;
    float max_score = -1.0f;
    int max_id = 0;
    for (int i = 0; i < num_classes; ++i) {
        float p = sigmoidf(clsp[i]);
        if (p > max_score) {
            max_score = p;
            max_id = i;
        }
    }
    scores[bn * anchors + a] = max_score;
    class_id[bn * anchors + a] = max_id;
}

void launch_decode_predictions(const float* d_ltrb, const float* d_cls,
                               const float* d_anchor_xy, const float* d_anchor_stride,
                               float* d_boxes, float* d_scores, int* d_class_id,
                               int n, int anchors, int num_classes,
                               cudaStream_t stream) {
    int block = 128;
    dim3 grid((anchors + block - 1) / block, n);
    decode_predictions_kernel<<<grid, block, 0, stream>>>(
        d_ltrb, d_cls, d_anchor_xy, d_anchor_stride,
        d_boxes, d_scores, d_class_id, n, anchors, num_classes);
}

// ============================================================
// Score filtering (atomic compaction)
// ============================================================

__global__ void score_filter_kernel(const float* __restrict__ boxes_in,
                                    const float* __restrict__ scores_in,
                                    const int* __restrict__ class_in,
                                    float* __restrict__ boxes_out,
                                    float* __restrict__ scores_out,
                                    int* __restrict__ class_out,
                                    int* __restrict__ count,
                                    int n_in, float score_thresh, int max_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) return;
    if (scores_in[idx] < score_thresh) return;
    int slot = atomicAdd(count, 1);
    if (slot >= max_out) return;
    boxes_out[slot * 4 + 0] = boxes_in[idx * 4 + 0];
    boxes_out[slot * 4 + 1] = boxes_in[idx * 4 + 1];
    boxes_out[slot * 4 + 2] = boxes_in[idx * 4 + 2];
    boxes_out[slot * 4 + 3] = boxes_in[idx * 4 + 3];
    scores_out[slot] = scores_in[idx];
    class_out[slot] = class_in[idx];
}

void launch_score_filter(const float* d_boxes_in, const float* d_scores_in,
                         const int* d_class_in,
                         float* d_boxes_out, float* d_scores_out, int* d_class_out,
                         int* d_count,
                         int n_in, float score_thresh, int max_out,
                         cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
    int block = 256;
    int grid = (n_in + block - 1) / block;
    score_filter_kernel<<<grid, block, 0, stream>>>(
        d_boxes_in, d_scores_in, d_class_in,
        d_boxes_out, d_scores_out, d_class_out,
        d_count, n_in, score_thresh, max_out);
}

// ============================================================
// NMS — sequential-on-GPU (simple, correct for K <= ~few thousand)
// ============================================================

__device__ __forceinline__ float iou_xyxy(const float* a, const float* b) {
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);
    float iw = fmaxf(0.0f, x2 - x1);
    float ih = fmaxf(0.0f, y2 - y1);
    float inter = iw * ih;
    float aa = fmaxf(0.0f, a[2] - a[0]) * fmaxf(0.0f, a[3] - a[1]);
    float bb = fmaxf(0.0f, b[2] - b[0]) * fmaxf(0.0f, b[3] - b[1]);
    float u = aa + bb - inter;
    return u > 0.0f ? inter / u : 0.0f;
}

// Bitonic-like sort kernel — here we'll just use a small global-memory sort on host indices.
// For simplicity and correctness, we run NMS in a single block via thrust-like
// selection: pick max-score not-yet-suppressed iteratively.
__global__ void nms_single_block_kernel(const float* __restrict__ boxes,
                                        const float* __restrict__ scores,
                                        const int* __restrict__ class_id,
                                        int* __restrict__ keep,
                                        int* __restrict__ keep_count,
                                        int k, float iou_thresh, int max_out) {
    extern __shared__ int suppressed[];  // size k bytes-as-int
    int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) suppressed[i] = 0;
    __syncthreads();

    if (tid == 0) {
        int kept = 0;
        for (int iter = 0; iter < k && kept < max_out; ++iter) {
            // find best remaining
            int best = -1;
            float best_score = -1.0f;
            for (int i = 0; i < k; ++i) {
                if (suppressed[i]) continue;
                if (scores[i] > best_score) {
                    best_score = scores[i];
                    best = i;
                }
            }
            if (best < 0) break;
            keep[kept++] = best;
            suppressed[best] = 1;
            // suppress same-class higher-IoU peers
            for (int j = 0; j < k; ++j) {
                if (suppressed[j]) continue;
                if (class_id[j] != class_id[best]) continue;
                if (iou_xyxy(boxes + best * 4, boxes + j * 4) > iou_thresh) {
                    suppressed[j] = 1;
                }
            }
        }
        *keep_count = kept;
    }
}

void launch_nms(const float* d_boxes, const float* d_scores, const int* d_class_id,
                int* d_keep, int* d_keep_count,
                int k, float iou_thresh, int max_out,
                cudaStream_t stream) {
    if (k == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_keep_count, 0, sizeof(int), stream));
        return;
    }
    int threads = 1;  // serial inside kernel for correctness; k is small post-filter
    size_t shmem = sizeof(int) * (size_t)k;
    nms_single_block_kernel<<<1, threads, shmem, stream>>>(
        d_boxes, d_scores, d_class_id, d_keep, d_keep_count,
        k, iou_thresh, max_out);
}

// ============================================================
// Undo letterbox transform on boxes (xyxy)
// ============================================================

__global__ void unletterbox_kernel(float* boxes, int n,
                                   float inv_scale, float pad_x, float pad_y,
                                   float orig_w, float orig_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x1 = (boxes[idx * 4 + 0] - pad_x) * inv_scale;
    float y1 = (boxes[idx * 4 + 1] - pad_y) * inv_scale;
    float x2 = (boxes[idx * 4 + 2] - pad_x) * inv_scale;
    float y2 = (boxes[idx * 4 + 3] - pad_y) * inv_scale;
    x1 = fmaxf(0.0f, fminf(x1, orig_w - 1.0f));
    y1 = fmaxf(0.0f, fminf(y1, orig_h - 1.0f));
    x2 = fmaxf(0.0f, fminf(x2, orig_w - 1.0f));
    y2 = fmaxf(0.0f, fminf(y2, orig_h - 1.0f));
    boxes[idx * 4 + 0] = x1;
    boxes[idx * 4 + 1] = y1;
    boxes[idx * 4 + 2] = x2;
    boxes[idx * 4 + 3] = y2;
}

void launch_unletterbox_boxes(float* d_boxes, int n,
                              float scale, int pad_x, int pad_y,
                              int orig_w, int orig_h,
                              cudaStream_t stream) {
    if (n == 0) return;
    int block = 128;
    int grid = (n + block - 1) / block;
    unletterbox_kernel<<<grid, block, 0, stream>>>(
        d_boxes, n, 1.0f / scale, (float)pad_x, (float)pad_y,
        (float)orig_w, (float)orig_h);
}
