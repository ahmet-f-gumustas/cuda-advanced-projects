#include "vae.h"
#include "diffusion_kernels.cuh"

// ============================================================
// VAEResBlock implementation (simpler than UNet ResBlock: no time embedding)
// ============================================================

void VAEResBlock::init(int in_ch, int out_ch, int num_groups) {
    in_channels = in_ch;
    out_channels = out_ch;
    has_skip = (in_ch != out_ch);

    norm1.init(in_ch, num_groups);
    conv1.init(in_ch, out_ch, 3, 1, 1);

    norm2.init(out_ch, num_groups);
    conv2.init(out_ch, out_ch, 3, 1, 1);

    if (has_skip) {
        skip_conv.init(in_ch, out_ch, 1, 1, 0);
    }
}

void VAEResBlock::initRandom(float scale, unsigned& seed) {
    norm1.initRandom(seed);
    conv1.initRandom(scale, seed);
    norm2.initRandom(seed);
    conv2.initRandom(scale, seed);
    if (has_skip) skip_conv.initRandom(scale, seed);
}

void VAEResBlock::forward(cudnnHandle_t cudnn,
                          const half* d_input, half* d_output,
                          int H, int W,
                          void* d_workspace, size_t ws_size,
                          half* d_temp)
{
    int spatial = H * W;

    // norm1 -> silu -> conv1
    norm1.forward(d_input, d_temp, H, W);

    int size1 = in_channels * spatial;
    silu_inplace_kernel<<<(size1 + 255) / 256, 256>>>(d_temp, size1);
    CUDA_CHECK_LAST_ERROR();

    conv1.forward(cudnn, d_temp, d_output, H, W, d_workspace, ws_size);

    // norm2 -> silu -> conv2
    norm2.forward(d_output, d_temp, H, W);

    int size2 = out_channels * spatial;
    silu_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_temp, size2);
    CUDA_CHECK_LAST_ERROR();

    conv2.forward(cudnn, d_temp, d_output, H, W, d_workspace, ws_size);

    // Skip connection
    if (has_skip) {
        // d_output currently has conv2 result. Need to add skip_conv(input).
        skip_conv.forward(cudnn, d_input, d_temp, H, W, d_workspace, ws_size);
        add_tensors_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_output, d_temp, size2);
    } else {
        add_tensors_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_output, d_input, size2);
    }
    CUDA_CHECK_LAST_ERROR();
}

void VAEResBlock::destroy() {
    norm1.destroy(); norm2.destroy();
    conv1.destroy(); conv2.destroy();
    if (has_skip) skip_conv.destroy();
}

// ============================================================
// VAEDecoder implementation
// ============================================================

VAEDecoder::VAEDecoder(const VAEConfig& cfg)
    : cfg_(cfg), d_workspace_(nullptr), workspace_size_(0),
      d_buf1_(nullptr), d_buf2_(nullptr), d_temp_(nullptr), max_buf_size_(0)
{
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));

    int ng = cfg_.num_groups;

    // Post-quant conv: 4 -> 512 (1x1)
    post_quant_conv_.init(cfg_.latent_channels, cfg_.channels[0], 1, 1, 0);

    // Mid block: two ResBlocks
    mid_res1_.init(cfg_.channels[0], cfg_.channels[0], ng);
    mid_res2_.init(cfg_.channels[0], cfg_.channels[0], ng);

    // Up blocks: 4 levels, each with n_res_blocks + 1 ResBlocks + upsample
    up_blocks_.resize(cfg_.num_levels);
    for (int lvl = 0; lvl < cfg_.num_levels; lvl++) {
        int ch_in = (lvl == 0) ? cfg_.channels[0] : cfg_.channels[lvl - 1];
        int ch_out = cfg_.channels[lvl];
        int n_blocks = cfg_.n_res_blocks + 1;

        up_blocks_[lvl].resize(n_blocks);
        for (int r = 0; r < n_blocks; r++) {
            int in_ch = (r == 0) ? ch_in : ch_out;
            up_blocks_[lvl][r].init(in_ch, ch_out, ng);
        }
    }

    // Upsample convs (after nearest-neighbor)
    upsample_convs_.resize(cfg_.num_levels);
    for (int lvl = 0; lvl < cfg_.num_levels; lvl++) {
        int ch = cfg_.channels[lvl];
        upsample_convs_[lvl].init(ch, ch, 3, 1, 1);
    }

    // Final output: GroupNorm + SiLU + Conv(128 -> 3)
    final_norm_.init(cfg_.channels[cfg_.num_levels - 1], ng);
    final_conv_.init(cfg_.channels[cfg_.num_levels - 1], cfg_.output_channels, 3, 1, 1);
}

VAEDecoder::~VAEDecoder() {
    freeWeights();
    cudaFree(d_workspace_);
    cudaFree(d_buf1_);
    cudaFree(d_buf2_);
    cudaFree(d_temp_);
    cudnnDestroy(cudnn_handle_);
}

void VAEDecoder::freeWeights() {
    post_quant_conv_.destroy();
    mid_res1_.destroy();
    mid_res2_.destroy();
    for (auto& lvl : up_blocks_) for (auto& r : lvl) r.destroy();
    for (auto& u : upsample_convs_) u.destroy();
    final_norm_.destroy();
    final_conv_.destroy();
}

void VAEDecoder::initRandom(unsigned long long seed) {
    unsigned s = (unsigned)seed;
    float scale = 0.02f;

    post_quant_conv_.initRandom(scale, s);
    mid_res1_.initRandom(scale, s);
    mid_res2_.initRandom(scale, s);

    for (auto& lvl : up_blocks_) for (auto& r : lvl) r.initRandom(scale, s);
    for (auto& u : upsample_convs_) u.initRandom(scale, s);

    final_norm_.initRandom(s);
    final_conv_.initRandom(scale, s);

    printf("[VAE] Initialized with random weights (levels=%d, out_ch=%d)\n",
           cfg_.num_levels, cfg_.output_channels);
}

void VAEDecoder::decode(const half* d_latent, half* d_output, int H, int W) {
    // Allocate buffers based on output size
    // Output is 8x input spatial: H*8, W*8
    int out_h = H * 8;
    int out_w = W * 8;
    int max_ch = cfg_.channels[0];
    int max_spatial = out_h * out_w;
    int needed = max_ch * max_spatial;

    if (needed > max_buf_size_) {
        if (d_buf1_) cudaFree(d_buf1_);
        if (d_buf2_) cudaFree(d_buf2_);
        if (d_temp_) cudaFree(d_temp_);
        if (d_workspace_) cudaFree(d_workspace_);

        max_buf_size_ = needed;
        d_buf1_ = cudaMallocDevice<half>(max_buf_size_);
        d_buf2_ = cudaMallocDevice<half>(max_buf_size_);
        d_temp_ = cudaMallocDevice<half>(max_buf_size_);
        workspace_size_ = 64 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_size_));
    }

    int h = H, w = W;

    // Post-quant conv: [1, 4, H, W] -> [1, 512, H, W]
    post_quant_conv_.forward(cudnn_handle_, d_latent, d_buf1_, h, w,
                             d_workspace_, workspace_size_);

    // Mid block
    mid_res1_.forward(cudnn_handle_, d_buf1_, d_buf2_, h, w,
                      d_workspace_, workspace_size_, d_temp_);
    mid_res2_.forward(cudnn_handle_, d_buf2_, d_buf1_, h, w,
                      d_workspace_, workspace_size_, d_temp_);

    // Up blocks
    half* d_current = d_buf1_;
    for (int lvl = 0; lvl < cfg_.num_levels; lvl++) {
        for (int r = 0; r < (int)up_blocks_[lvl].size(); r++) {
            up_blocks_[lvl][r].forward(cudnn_handle_, d_current, d_buf2_, h, w,
                                       d_workspace_, workspace_size_, d_temp_);
            std::swap(d_current, d_buf2_);
        }

        // Upsample 2x
        int ch = cfg_.channels[lvl];
        int up_size = ch * h * 2 * w * 2;
        upsample_nearest_2x_kernel<<<(up_size + 255) / 256, 256>>>(
            d_buf2_, d_current, ch, h, w);
        CUDA_CHECK_LAST_ERROR();

        upsample_convs_[lvl].forward(cudnn_handle_, d_buf2_, d_current,
                                      h * 2, w * 2, d_workspace_, workspace_size_);
        h *= 2;
        w *= 2;
    }

    // Final: GroupNorm -> SiLU -> Conv
    int final_ch = cfg_.channels[cfg_.num_levels - 1];
    final_norm_.forward(d_current, d_buf2_, h, w);

    int final_size = final_ch * h * w;
    silu_inplace_kernel<<<(final_size + 255) / 256, 256>>>(d_buf2_, final_size);
    CUDA_CHECK_LAST_ERROR();

    final_conv_.forward(cudnn_handle_, d_buf2_, d_output, h, w,
                        d_workspace_, workspace_size_);

    // Clamp to [0, 1]
    int out_size = cfg_.output_channels * h * w;
    clamp_kernel<<<(out_size + 255) / 256, 256>>>(d_output, 0.0f, 1.0f, out_size);
    CUDA_CHECK_LAST_ERROR();
}
