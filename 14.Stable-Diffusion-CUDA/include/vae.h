#ifndef VAE_H
#define VAE_H

#include "cuda_utils.h"
#include "unet.h"  // For Conv2D, GroupNormLayer

// ============================================================
// VAE Decoder
// Decodes latent [1, 4, H, W] -> image [1, 3, 8H, 8W]
// Architecture: post_quant_conv -> mid -> up_blocks -> final_conv
// ============================================================

struct VAEConfig {
    int latent_channels = 4;
    int base_channels = 128;
    int output_channels = 3;       // RGB
    int num_groups = 32;
    float group_norm_eps = 1e-5f;

    // Decoder channel progression (reversed from encoder)
    // Level 0 (lowest res): 512
    // Level 1: 512
    // Level 2: 256
    // Level 3 (highest res): 128
    int channels[4] = {512, 512, 256, 128};
    int num_levels = 4;
    int n_res_blocks = 2;   // ResBlocks per level
};

// Simple ResBlock for VAE (no time embedding, no attention)
struct VAEResBlock {
    GroupNormLayer norm1, norm2;
    Conv2D conv1, conv2;
    Conv2D skip_conv;
    bool has_skip;
    int in_channels, out_channels;

    void init(int in_ch, int out_ch, int num_groups = 32);
    void initRandom(float scale, unsigned& seed);
    void forward(cudnnHandle_t cudnn,
                 const half* d_input, half* d_output,
                 int H, int W,
                 void* d_workspace, size_t ws_size,
                 half* d_temp);
    void destroy();
};

class VAEDecoder {
public:
    VAEDecoder(const VAEConfig& cfg);
    ~VAEDecoder();

    void initRandom(unsigned long long seed = 100);

    // Decode latent to image
    // d_latent: [1, 4, H, W] (FP16)
    // d_output: [1, 3, 8H, 8W] (FP16, clamped to [0,1])
    void decode(const half* d_latent, half* d_output, int H, int W);

    const VAEConfig& config() const { return cfg_; }

private:
    void freeWeights();

    VAEConfig cfg_;
    cudnnHandle_t cudnn_handle_;

    // Post-quantization conv
    Conv2D post_quant_conv_;    // [4 -> 512, 1x1]

    // Mid block
    VAEResBlock mid_res1_, mid_res2_;

    // Up blocks: [num_levels][n_res_blocks+1] ResBlocks + upsample conv
    std::vector<std::vector<VAEResBlock>> up_blocks_;
    std::vector<Conv2D> upsample_convs_;  // Conv after nearest upsample

    // Output
    GroupNormLayer final_norm_;
    Conv2D final_conv_;         // [128 -> 3, 3x3]

    // Workspace
    void* d_workspace_;
    size_t workspace_size_;

    // Temp buffers
    half* d_buf1_;
    half* d_buf2_;
    half* d_temp_;
    int max_buf_size_;
};

#endif // VAE_H
