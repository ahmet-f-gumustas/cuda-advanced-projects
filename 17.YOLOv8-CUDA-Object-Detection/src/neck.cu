#include "neck.h"
#include "cuda_utils.h"
#include "yolo_kernels.cuh"

#include <algorithm>

Neck::Neck(cudnnHandle_t handle, int in_w, int in_h, unsigned seed)
    : handle_(handle), in_w_(in_w), in_h_(in_h) {

    conv_p4n_     = std::make_unique<Conv2D>(handle_, 384, 128, 3, 1, 1, true, seed + 100);
    conv_p3o_     = std::make_unique<Conv2D>(handle_, 192,  64, 3, 1, 1, true, seed + 101);
    conv_p3_down_ = std::make_unique<Conv2D>(handle_,  64,  64, 3, 2, 1, true, seed + 102);
    conv_p4o_     = std::make_unique<Conv2D>(handle_, 192, 128, 3, 1, 1, true, seed + 103);
    conv_p4_down_ = std::make_unique<Conv2D>(handle_, 128, 128, 3, 2, 1, true, seed + 104);
    conv_p5o_     = std::make_unique<Conv2D>(handle_, 384, 256, 3, 1, 1, true, seed + 105);

    int h8 = in_h / 8,  w8  = in_w / 8;
    int h16 = in_h / 16, w16 = in_w / 16;
    int h32 = in_h / 32, w32 = in_w / 32;

    auto alloc = [](float** ptr, size_t n) {
        CUDA_CHECK(cudaMalloc(ptr, sizeof(float) * n));
    };

    alloc(&d_up_p5_,    256 * h16 * w16);
    alloc(&d_cat_top1_, 384 * h16 * w16);
    alloc(&d_p4_n_,     128 * h16 * w16);
    alloc(&d_up_p4n_,   128 * h8  * w8);
    alloc(&d_cat_top2_, 192 * h8  * w8);
    alloc(&d_p3_out_,    64 * h8  * w8);
    alloc(&d_p3_down_,   64 * h16 * w16);
    alloc(&d_cat_bot1_, 192 * h16 * w16);
    alloc(&d_p4_out_,   128 * h16 * w16);
    alloc(&d_p4_down_,  128 * h32 * w32);
    alloc(&d_cat_bot2_, 384 * h32 * w32);
    alloc(&d_p5_out_,   256 * h32 * w32);

    conv_ws_bytes_ = std::max({
        conv_p4n_->workspace_bytes(1, h16, w16),
        conv_p3o_->workspace_bytes(1, h8,  w8),
        conv_p3_down_->workspace_bytes(1, h8, w8),
        conv_p4o_->workspace_bytes(1, h16, w16),
        conv_p4_down_->workspace_bytes(1, h16, w16),
        conv_p5o_->workspace_bytes(1, h32, w32)
    });
}

Neck::~Neck() {
    auto del = [](float* p) { if (p) cudaFree(p); };
    del(d_up_p5_); del(d_cat_top1_); del(d_p4_n_);
    del(d_up_p4n_); del(d_cat_top2_); del(d_p3_out_);
    del(d_p3_down_); del(d_cat_bot1_); del(d_p4_out_);
    del(d_p4_down_); del(d_cat_bot2_); del(d_p5_out_);
}

void Neck::forward(const float* d_p3, const float* d_p4, const float* d_p5,
                   void* ws, size_t ws_bytes, cudaStream_t stream) {
    int h8 = in_h_ / 8,  w8  = in_w_ / 8;
    int h16 = in_h_ / 16, w16 = in_w_ / 16;
    int h32 = in_h_ / 32, w32 = in_w_ / 32;

    // Top-down path
    launch_upsample_nearest_2x(d_p5, d_up_p5_, 1, 256, h32, w32, stream);
    launch_concat_channel(d_up_p5_, 256, d_p4, 128, d_cat_top1_, 1, h16, w16, stream);
    conv_p4n_->forward(d_cat_top1_, d_p4_n_, 1, h16, w16, ws, ws_bytes, stream);

    launch_upsample_nearest_2x(d_p4_n_, d_up_p4n_, 1, 128, h16, w16, stream);
    launch_concat_channel(d_up_p4n_, 128, d_p3, 64, d_cat_top2_, 1, h8, w8, stream);
    conv_p3o_->forward(d_cat_top2_, d_p3_out_, 1, h8, w8, ws, ws_bytes, stream);

    // Bottom-up path
    conv_p3_down_->forward(d_p3_out_, d_p3_down_, 1, h8, w8, ws, ws_bytes, stream);
    launch_concat_channel(d_p3_down_, 64, d_p4_n_, 128, d_cat_bot1_, 1, h16, w16, stream);
    conv_p4o_->forward(d_cat_bot1_, d_p4_out_, 1, h16, w16, ws, ws_bytes, stream);

    conv_p4_down_->forward(d_p4_out_, d_p4_down_, 1, h16, w16, ws, ws_bytes, stream);
    launch_concat_channel(d_p4_down_, 128, d_p5, 256, d_cat_bot2_, 1, h32, w32, stream);
    conv_p5o_->forward(d_cat_bot2_, d_p5_out_, 1, h32, w32, ws, ws_bytes, stream);
}
