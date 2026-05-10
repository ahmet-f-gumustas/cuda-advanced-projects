#include "backbone.h"
#include "cuda_utils.h"
#include "yolo_kernels.cuh"

#include <algorithm>

// Element-wise add: y = a + b (in-place into a).
__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] = a[idx] + b[idx];
}

static void launch_add_inplace(float* d_a, const float* d_b, int n,
                               cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_inplace_kernel<<<grid, block, 0, stream>>>(d_a, d_b, n);
}

Backbone::Backbone(cudnnHandle_t handle, int in_w, int in_h, unsigned seed)
    : handle_(handle), in_w_(in_w), in_h_(in_h) {

    // Layer construction (with SiLU; bottleneck second conv has no activation
    // before residual add — but we keep SiLU for simplicity and let the model
    // be slightly different from canonical).
    stem_     = std::make_unique<Conv2D>(handle_, 3,   16,  3, 2, 1, true,  seed + 0);
    down1_    = std::make_unique<Conv2D>(handle_, 16,  32,  3, 2, 1, true,  seed + 1);
    bot1_a_   = std::make_unique<Conv2D>(handle_, 32,  32,  3, 1, 1, true,  seed + 2);
    bot1_b_   = std::make_unique<Conv2D>(handle_, 32,  32,  3, 1, 1, false, seed + 3);
    down2_    = std::make_unique<Conv2D>(handle_, 32,  64,  3, 2, 1, true,  seed + 4);
    bot2_a_   = std::make_unique<Conv2D>(handle_, 64,  64,  3, 1, 1, true,  seed + 5);
    bot2_b_   = std::make_unique<Conv2D>(handle_, 64,  64,  3, 1, 1, false, seed + 6);
    down3_    = std::make_unique<Conv2D>(handle_, 64,  128, 3, 2, 1, true,  seed + 7);
    bot3_a_   = std::make_unique<Conv2D>(handle_, 128, 128, 3, 1, 1, true,  seed + 8);
    bot3_b_   = std::make_unique<Conv2D>(handle_, 128, 128, 3, 1, 1, false, seed + 9);
    down4_    = std::make_unique<Conv2D>(handle_, 128, 256, 3, 2, 1, true,  seed + 10);
    bot4_a_   = std::make_unique<Conv2D>(handle_, 256, 256, 3, 1, 1, true,  seed + 11);
    bot4_b_   = std::make_unique<Conv2D>(handle_, 256, 256, 3, 1, 1, false, seed + 12);
    sppf_in_  = std::make_unique<Conv2D>(handle_, 256, 128, 1, 1, 0, true,  seed + 13);
    sppf_out_ = std::make_unique<Conv2D>(handle_, 512, 256, 1, 1, 0, true,  seed + 14);

    int h2  = in_h / 2,  w2  = in_w / 2;
    int h4  = in_h / 4,  w4  = in_w / 4;
    int h8  = in_h / 8,  w8  = in_w / 8;
    int h16 = in_h / 16, w16 = in_w / 16;
    int h32 = in_h / 32, w32 = in_w / 32;

    auto alloc = [](float** ptr, size_t n) {
        CUDA_CHECK(cudaMalloc(ptr, sizeof(float) * n));
    };

    int B = 1;
    alloc(&d_stem_, B * 16 * h2 * w2);
    alloc(&d_d1_,   B * 32 * h4 * w4);
    alloc(&d_b1a_,  B * 32 * h4 * w4);
    alloc(&d_b1b_,  B * 32 * h4 * w4);
    alloc(&d_d2_,   B * 64 * h8 * w8);
    alloc(&d_b2a_,  B * 64 * h8 * w8);
    alloc(&d_b2b_,  B * 64 * h8 * w8);
    alloc(&d_p3_,   B * 64 * h8 * w8);
    alloc(&d_d3_,   B * 128 * h16 * w16);
    alloc(&d_b3a_,  B * 128 * h16 * w16);
    alloc(&d_b3b_,  B * 128 * h16 * w16);
    alloc(&d_p4_,   B * 128 * h16 * w16);
    alloc(&d_d4_,   B * 256 * h32 * w32);
    alloc(&d_b4a_,  B * 256 * h32 * w32);
    alloc(&d_b4b_,  B * 256 * h32 * w32);
    alloc(&d_sppf_in_,  B * 128 * h32 * w32);
    alloc(&d_sppf_y1_,  B * 128 * h32 * w32);
    alloc(&d_sppf_y2_,  B * 128 * h32 * w32);
    alloc(&d_sppf_y3_,  B * 128 * h32 * w32);
    alloc(&d_sppf_cat_, B * 512 * h32 * w32);
    alloc(&d_sppf_cat2_, B * 512 * h32 * w32);
    alloc(&d_p5_,   B * 256 * h32 * w32);

    // Worst-case workspace among all convs at their respective input sizes.
    conv_ws_bytes_ = std::max({
        stem_->workspace_bytes(B, in_h, in_w),
        down1_->workspace_bytes(B, h2, w2),
        bot1_a_->workspace_bytes(B, h4, w4),
        bot1_b_->workspace_bytes(B, h4, w4),
        down2_->workspace_bytes(B, h4, w4),
        bot2_a_->workspace_bytes(B, h8, w8),
        bot2_b_->workspace_bytes(B, h8, w8),
        down3_->workspace_bytes(B, h8, w8),
        bot3_a_->workspace_bytes(B, h16, w16),
        bot3_b_->workspace_bytes(B, h16, w16),
        down4_->workspace_bytes(B, h16, w16),
        bot4_a_->workspace_bytes(B, h32, w32),
        bot4_b_->workspace_bytes(B, h32, w32),
        sppf_in_->workspace_bytes(B, h32, w32),
        sppf_out_->workspace_bytes(B, h32, w32)
    });
}

Backbone::~Backbone() {
    auto del = [](float* p) { if (p) cudaFree(p); };
    del(d_stem_); del(d_d1_); del(d_b1a_); del(d_b1b_);
    del(d_d2_); del(d_b2a_); del(d_b2b_); del(d_p3_);
    del(d_d3_); del(d_b3a_); del(d_b3b_); del(d_p4_);
    del(d_d4_); del(d_b4a_); del(d_b4b_);
    del(d_sppf_in_); del(d_sppf_y1_); del(d_sppf_y2_); del(d_sppf_y3_);
    del(d_sppf_cat_); del(d_sppf_cat2_); del(d_p5_);
}

void Backbone::forward(const float* d_input, void* ws, size_t ws_bytes,
                       cudaStream_t stream) {
    int h2  = in_h_ / 2,  w2  = in_w_ / 2;
    int h4  = in_h_ / 4,  w4  = in_w_ / 4;
    int h8  = in_h_ / 8,  w8  = in_w_ / 8;
    int h16 = in_h_ / 16, w16 = in_w_ / 16;
    int h32 = in_h_ / 32, w32 = in_w_ / 32;

    // Stem
    stem_->forward(d_input, d_stem_, 1, in_h_, in_w_, ws, ws_bytes, stream);

    // Stage 1
    down1_->forward(d_stem_, d_d1_, 1, h2, w2, ws, ws_bytes, stream);
    bot1_a_->forward(d_d1_, d_b1a_, 1, h4, w4, ws, ws_bytes, stream);
    bot1_b_->forward(d_b1a_, d_b1b_, 1, h4, w4, ws, ws_bytes, stream);
    launch_add_inplace(d_b1b_, d_d1_, 32 * h4 * w4, stream);
    launch_silu(d_b1b_, 32 * h4 * w4, stream);

    // Stage 2 → P3
    down2_->forward(d_b1b_, d_d2_, 1, h4, w4, ws, ws_bytes, stream);
    bot2_a_->forward(d_d2_, d_b2a_, 1, h8, w8, ws, ws_bytes, stream);
    bot2_b_->forward(d_b2a_, d_b2b_, 1, h8, w8, ws, ws_bytes, stream);
    launch_add_inplace(d_b2b_, d_d2_, 64 * h8 * w8, stream);
    launch_silu(d_b2b_, 64 * h8 * w8, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_p3_, d_b2b_,
                               sizeof(float) * 64 * h8 * w8,
                               cudaMemcpyDeviceToDevice, stream));

    // Stage 3 → P4
    down3_->forward(d_b2b_, d_d3_, 1, h8, w8, ws, ws_bytes, stream);
    bot3_a_->forward(d_d3_, d_b3a_, 1, h16, w16, ws, ws_bytes, stream);
    bot3_b_->forward(d_b3a_, d_b3b_, 1, h16, w16, ws, ws_bytes, stream);
    launch_add_inplace(d_b3b_, d_d3_, 128 * h16 * w16, stream);
    launch_silu(d_b3b_, 128 * h16 * w16, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_p4_, d_b3b_,
                               sizeof(float) * 128 * h16 * w16,
                               cudaMemcpyDeviceToDevice, stream));

    // Stage 4
    down4_->forward(d_b3b_, d_d4_, 1, h16, w16, ws, ws_bytes, stream);
    bot4_a_->forward(d_d4_, d_b4a_, 1, h32, w32, ws, ws_bytes, stream);
    bot4_b_->forward(d_b4a_, d_b4b_, 1, h32, w32, ws, ws_bytes, stream);
    launch_add_inplace(d_b4b_, d_d4_, 256 * h32 * w32, stream);
    launch_silu(d_b4b_, 256 * h32 * w32, stream);

    // SPPF
    sppf_in_->forward(d_b4b_, d_sppf_in_, 1, h32, w32, ws, ws_bytes, stream);
    launch_maxpool2d_same(d_sppf_in_, d_sppf_y1_, 1, 128, h32, w32, 5, stream);
    launch_maxpool2d_same(d_sppf_y1_, d_sppf_y2_, 1, 128, h32, w32, 5, stream);
    launch_maxpool2d_same(d_sppf_y2_, d_sppf_y3_, 1, 128, h32, w32, 5, stream);
    // Four-way concat [y0, y1, y2, y3] -> 512 channels via ping-pong buffers.
    launch_concat_channel(d_sppf_in_, 128, d_sppf_y1_, 128,
                          d_sppf_cat_, 1, h32, w32, stream);            // 256 ch
    launch_concat_channel(d_sppf_cat_, 256, d_sppf_y2_, 128,
                          d_sppf_cat2_, 1, h32, w32, stream);           // 384 ch
    launch_concat_channel(d_sppf_cat2_, 384, d_sppf_y3_, 128,
                          d_sppf_cat_, 1, h32, w32, stream);            // 512 ch
    sppf_out_->forward(d_sppf_cat_, d_p5_, 1, h32, w32, ws, ws_bytes, stream);
}
