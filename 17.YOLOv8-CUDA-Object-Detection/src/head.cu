#include "head.h"
#include "cuda_utils.h"
#include "yolo_kernels.cuh"

#include <algorithm>
#include <vector>

// Rearrange [B, C, H, W] -> [B, total_anchors, C] writing at anchor offset `a_off`.
// Each output anchor index is a_off + h * W + w, channel-major within that anchor.
__global__ void nchw_to_anchor_major_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            int total_anchors, int c, int h, int w,
                                            int a_off) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    if (x >= w || y >= h || ch >= c) return;
    int s_idx = (ch * h + y) * w + x;
    int a = a_off + y * w + x;
    int d_idx = a * c + ch;
    dst[d_idx] = src[s_idx];
}

static void launch_nchw_to_anchor_major(const float* d_src, float* d_dst,
                                        int total_anchors, int c, int h, int w,
                                        int a_off, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((w + 15) / 16, (h + 15) / 16, c);
    nchw_to_anchor_major_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, total_anchors, c, h, w, a_off);
}

DetectHead::DetectHead(cudnnHandle_t handle, int in_w, int in_h,
                       int num_classes, int reg_max, unsigned seed)
    : handle_(handle), in_w_(in_w), in_h_(in_h),
      num_classes_(num_classes), reg_max_(reg_max) {

    int h8 = in_h / 8, w8 = in_w / 8;
    int h16 = in_h / 16, w16 = in_w / 16;
    int h32 = in_h / 32, w32 = in_w / 32;
    int a3 = h8 * w8;
    int a4 = h16 * w16;
    int a5 = h32 * w32;
    total_anchors_ = a3 + a4 + a5;

    // P3 (64 ch input)
    p3_cls_a_    = std::make_unique<Conv2D>(handle_, 64, 64, 3, 1, 1, true,  seed + 200);
    p3_cls_b_    = std::make_unique<Conv2D>(handle_, 64, 64, 3, 1, 1, true,  seed + 201);
    p3_cls_pred_ = std::make_unique<Conv2D>(handle_, 64, num_classes_, 1, 1, 0, false, seed + 202);
    p3_reg_a_    = std::make_unique<Conv2D>(handle_, 64, 64, 3, 1, 1, true,  seed + 203);
    p3_reg_b_    = std::make_unique<Conv2D>(handle_, 64, 64, 3, 1, 1, true,  seed + 204);
    p3_reg_pred_ = std::make_unique<Conv2D>(handle_, 64, 4 * reg_max_, 1, 1, 0, false, seed + 205);

    // P4 (128 ch input)
    p4_cls_a_    = std::make_unique<Conv2D>(handle_, 128, 128, 3, 1, 1, true,  seed + 210);
    p4_cls_b_    = std::make_unique<Conv2D>(handle_, 128, 128, 3, 1, 1, true,  seed + 211);
    p4_cls_pred_ = std::make_unique<Conv2D>(handle_, 128, num_classes_, 1, 1, 0, false, seed + 212);
    p4_reg_a_    = std::make_unique<Conv2D>(handle_, 128, 128, 3, 1, 1, true,  seed + 213);
    p4_reg_b_    = std::make_unique<Conv2D>(handle_, 128, 128, 3, 1, 1, true,  seed + 214);
    p4_reg_pred_ = std::make_unique<Conv2D>(handle_, 128, 4 * reg_max_, 1, 1, 0, false, seed + 215);

    // P5 (256 ch input)
    p5_cls_a_    = std::make_unique<Conv2D>(handle_, 256, 256, 3, 1, 1, true,  seed + 220);
    p5_cls_b_    = std::make_unique<Conv2D>(handle_, 256, 256, 3, 1, 1, true,  seed + 221);
    p5_cls_pred_ = std::make_unique<Conv2D>(handle_, 256, num_classes_, 1, 1, 0, false, seed + 222);
    p5_reg_a_    = std::make_unique<Conv2D>(handle_, 256, 256, 3, 1, 1, true,  seed + 223);
    p5_reg_b_    = std::make_unique<Conv2D>(handle_, 256, 256, 3, 1, 1, true,  seed + 224);
    p5_reg_pred_ = std::make_unique<Conv2D>(handle_, 256, 4 * reg_max_, 1, 1, 0, false, seed + 225);

    auto alloc = [](float** ptr, size_t n) {
        CUDA_CHECK(cudaMalloc(ptr, sizeof(float) * n));
    };

    // P3 intermediates
    alloc(&d_p3_cls_a_, 64 * a3); alloc(&d_p3_cls_b_, 64 * a3); alloc(&d_p3_cls_out_, num_classes_ * a3);
    alloc(&d_p3_reg_a_, 64 * a3); alloc(&d_p3_reg_b_, 64 * a3); alloc(&d_p3_reg_out_, 4 * reg_max_ * a3);
    // P4
    alloc(&d_p4_cls_a_, 128 * a4); alloc(&d_p4_cls_b_, 128 * a4); alloc(&d_p4_cls_out_, num_classes_ * a4);
    alloc(&d_p4_reg_a_, 128 * a4); alloc(&d_p4_reg_b_, 128 * a4); alloc(&d_p4_reg_out_, 4 * reg_max_ * a4);
    // P5
    alloc(&d_p5_cls_a_, 256 * a5); alloc(&d_p5_cls_b_, 256 * a5); alloc(&d_p5_cls_out_, num_classes_ * a5);
    alloc(&d_p5_reg_a_, 256 * a5); alloc(&d_p5_reg_b_, 256 * a5); alloc(&d_p5_reg_out_, 4 * reg_max_ * a5);

    // Flat anchor-major outputs
    alloc(&d_cls_flat_, total_anchors_ * num_classes_);
    alloc(&d_reg_flat_, total_anchors_ * 4 * reg_max_);

    // Anchor grid (host build, then upload)
    std::vector<float> h_xy(total_anchors_ * 2);
    std::vector<float> h_stride(total_anchors_);
    int off = 0;
    build_anchor_grid(h8,  w8,  8,  h_xy.data(), h_stride.data(), off); off += a3;
    build_anchor_grid(h16, w16, 16, h_xy.data(), h_stride.data(), off); off += a4;
    build_anchor_grid(h32, w32, 32, h_xy.data(), h_stride.data(), off);

    alloc(&d_anchor_xy_, total_anchors_ * 2);
    alloc(&d_anchor_stride_, total_anchors_);
    CUDA_CHECK(cudaMemcpy(d_anchor_xy_, h_xy.data(),
                          sizeof(float) * total_anchors_ * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_anchor_stride_, h_stride.data(),
                          sizeof(float) * total_anchors_,
                          cudaMemcpyHostToDevice));

    conv_ws_bytes_ = std::max({
        p3_cls_a_->workspace_bytes(1, h8, w8),
        p3_reg_pred_->workspace_bytes(1, h8, w8),
        p4_cls_a_->workspace_bytes(1, h16, w16),
        p4_reg_pred_->workspace_bytes(1, h16, w16),
        p5_cls_a_->workspace_bytes(1, h32, w32),
        p5_reg_pred_->workspace_bytes(1, h32, w32)
    });
}

DetectHead::~DetectHead() {
    auto del = [](float* p) { if (p) cudaFree(p); };
    del(d_p3_cls_a_); del(d_p3_cls_b_); del(d_p3_cls_out_);
    del(d_p3_reg_a_); del(d_p3_reg_b_); del(d_p3_reg_out_);
    del(d_p4_cls_a_); del(d_p4_cls_b_); del(d_p4_cls_out_);
    del(d_p4_reg_a_); del(d_p4_reg_b_); del(d_p4_reg_out_);
    del(d_p5_cls_a_); del(d_p5_cls_b_); del(d_p5_cls_out_);
    del(d_p5_reg_a_); del(d_p5_reg_b_); del(d_p5_reg_out_);
    del(d_cls_flat_); del(d_reg_flat_);
    del(d_anchor_xy_); del(d_anchor_stride_);
}

void DetectHead::forward(const float* d_p3, const float* d_p4, const float* d_p5,
                         void* ws, size_t ws_bytes, cudaStream_t stream) {
    int h8 = in_h_ / 8, w8 = in_w_ / 8;
    int h16 = in_h_ / 16, w16 = in_w_ / 16;
    int h32 = in_h_ / 32, w32 = in_w_ / 32;
    int a3 = h8 * w8, a4 = h16 * w16;

    // P3
    p3_cls_a_->forward(d_p3, d_p3_cls_a_, 1, h8, w8, ws, ws_bytes, stream);
    p3_cls_b_->forward(d_p3_cls_a_, d_p3_cls_b_, 1, h8, w8, ws, ws_bytes, stream);
    p3_cls_pred_->forward(d_p3_cls_b_, d_p3_cls_out_, 1, h8, w8, ws, ws_bytes, stream);
    p3_reg_a_->forward(d_p3, d_p3_reg_a_, 1, h8, w8, ws, ws_bytes, stream);
    p3_reg_b_->forward(d_p3_reg_a_, d_p3_reg_b_, 1, h8, w8, ws, ws_bytes, stream);
    p3_reg_pred_->forward(d_p3_reg_b_, d_p3_reg_out_, 1, h8, w8, ws, ws_bytes, stream);

    // P4
    p4_cls_a_->forward(d_p4, d_p4_cls_a_, 1, h16, w16, ws, ws_bytes, stream);
    p4_cls_b_->forward(d_p4_cls_a_, d_p4_cls_b_, 1, h16, w16, ws, ws_bytes, stream);
    p4_cls_pred_->forward(d_p4_cls_b_, d_p4_cls_out_, 1, h16, w16, ws, ws_bytes, stream);
    p4_reg_a_->forward(d_p4, d_p4_reg_a_, 1, h16, w16, ws, ws_bytes, stream);
    p4_reg_b_->forward(d_p4_reg_a_, d_p4_reg_b_, 1, h16, w16, ws, ws_bytes, stream);
    p4_reg_pred_->forward(d_p4_reg_b_, d_p4_reg_out_, 1, h16, w16, ws, ws_bytes, stream);

    // P5
    p5_cls_a_->forward(d_p5, d_p5_cls_a_, 1, h32, w32, ws, ws_bytes, stream);
    p5_cls_b_->forward(d_p5_cls_a_, d_p5_cls_b_, 1, h32, w32, ws, ws_bytes, stream);
    p5_cls_pred_->forward(d_p5_cls_b_, d_p5_cls_out_, 1, h32, w32, ws, ws_bytes, stream);
    p5_reg_a_->forward(d_p5, d_p5_reg_a_, 1, h32, w32, ws, ws_bytes, stream);
    p5_reg_b_->forward(d_p5_reg_a_, d_p5_reg_b_, 1, h32, w32, ws, ws_bytes, stream);
    p5_reg_pred_->forward(d_p5_reg_b_, d_p5_reg_out_, 1, h32, w32, ws, ws_bytes, stream);

    // Pack into anchor-major flat tensors.
    launch_nchw_to_anchor_major(d_p3_cls_out_, d_cls_flat_, total_anchors_,
                                num_classes_, h8, w8, 0, stream);
    launch_nchw_to_anchor_major(d_p4_cls_out_, d_cls_flat_, total_anchors_,
                                num_classes_, h16, w16, a3, stream);
    launch_nchw_to_anchor_major(d_p5_cls_out_, d_cls_flat_, total_anchors_,
                                num_classes_, h32, w32, a3 + a4, stream);

    launch_nchw_to_anchor_major(d_p3_reg_out_, d_reg_flat_, total_anchors_,
                                4 * reg_max_, h8, w8, 0, stream);
    launch_nchw_to_anchor_major(d_p4_reg_out_, d_reg_flat_, total_anchors_,
                                4 * reg_max_, h16, w16, a3, stream);
    launch_nchw_to_anchor_major(d_p5_reg_out_, d_reg_flat_, total_anchors_,
                                4 * reg_max_, h32, w32, a3 + a4, stream);
}
