#ifndef UNET_H
#define UNET_H

#include "cuda_utils.h"

// ============================================================
// UNet for Diffusion Models
// Architecture: Down blocks -> Mid block -> Up blocks
// Features: ResBlocks, Self/Cross-Attention, Time embedding
// ============================================================

struct UNetConfig {
    int latent_channels = 4;        // Input latent channels
    int base_channels = 128;        // Base channel count
    int channel_mult[3] = {1, 2, 4}; // -> 128, 256, 512
    int num_levels = 3;
    int n_res_blocks = 2;           // ResBlocks per level
    int n_heads = 8;
    int context_dim = 512;          // From CLIP encoder
    int time_embed_dim = 512;       // Time embedding dimension
    float group_norm_eps = 1e-5f;
    int num_groups = 32;            // GroupNorm groups

    int channels(int level) const { return base_channels * channel_mult[level]; }
};

// ============================================================
// Conv2D layer using cuDNN
// ============================================================

struct Conv2D {
    half* d_weight;     // [out_ch, in_ch, kH, kW]
    half* d_bias;       // [out_ch] or nullptr
    int in_channels, out_channels;
    int kernel_size, stride, padding;

    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    void init(int in_ch, int out_ch, int k, int s, int p, bool use_bias = true);
    void initRandom(float scale, unsigned& seed);
    void forward(cudnnHandle_t handle,
                 const half* d_input, half* d_output,
                 int H, int W,
                 void* d_workspace, size_t ws_size);
    size_t getWorkspaceSize(cudnnHandle_t handle, int H, int W);
    void destroy();
};

// ============================================================
// GroupNorm + optional SiLU layer
// ============================================================

struct GroupNormLayer {
    half* d_gamma;      // [C]
    half* d_beta;       // [C]
    int channels;
    int num_groups;
    float eps;

    void init(int C, int groups, float epsilon = 1e-5f);
    void initRandom(unsigned& seed);
    void forward(const half* d_input, half* d_output, int H, int W);
    void destroy();
};

// ============================================================
// ResBlock: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + skip
// With time embedding projection
// ============================================================

struct ResBlock {
    GroupNormLayer norm1, norm2;
    Conv2D conv1, conv2;

    // Time embedding projection: Linear(time_embed_dim -> out_channels)
    half* d_time_proj_w;    // [time_embed_dim, out_channels]
    half* d_time_proj_b;    // [out_channels]

    // Skip connection if in_channels != out_channels
    Conv2D skip_conv;
    bool has_skip;
    int in_channels, out_channels;
    int time_embed_dim;

    void init(int in_ch, int out_ch, int temb_dim, int num_groups = 32);
    void initRandom(float scale, unsigned& seed);
    void forward(cudnnHandle_t cudnn, cublasHandle_t cublas,
                 const half* d_input, const half* d_temb,
                 half* d_output, int H, int W,
                 void* d_workspace, size_t ws_size,
                 half* d_temp1, half* d_temp2);
    void destroy();
};

// ============================================================
// Attention Block: Self-attention + Cross-attention
// Operates on flattened spatial dims: [H*W, C]
// ============================================================

struct AttentionBlock {
    // Self-attention
    GroupNormLayer self_norm;
    half* d_self_wq;        // [C, C]
    half* d_self_wk;        // [C, C]
    half* d_self_wv;        // [C, C]
    half* d_self_wo;        // [C, C]

    // Cross-attention
    GroupNormLayer cross_norm;
    half* d_cross_wq;       // [C, C]
    half* d_cross_wk;       // [context_dim, C]
    half* d_cross_wv;       // [context_dim, C]
    half* d_cross_wo;       // [C, C]

    int channels, context_dim, n_heads;

    void init(int C, int ctx_dim, int heads, int num_groups = 32);
    void initRandom(float scale, unsigned& seed);
    void forward(cublasHandle_t cublas,
                 const half* d_input,
                 const half* d_context,   // [seq_len, context_dim]
                 int context_len,
                 half* d_output,
                 int H, int W,
                 half* d_q, half* d_k, half* d_v,
                 half* d_attn_out, float* d_scores);
    void destroy();

private:
    void gemm(cublasHandle_t cublas, const half* A, const half* B, half* C,
              int m, int n, int k, float alpha = 1.0f, float beta = 0.0f);
};

// ============================================================
// UNet Model
// ============================================================

class UNet {
public:
    UNet(const UNetConfig& cfg);
    ~UNet();

    void initRandom(unsigned long long seed = 42);

    // Forward pass: predict noise
    // d_latent: [1, latent_ch, H, W] noisy input
    // d_context: [seq_len, context_dim] text embeddings
    // timestep: current diffusion timestep
    // d_output: [1, latent_ch, H, W] predicted noise
    void forward(const half* d_latent,
                 const half* d_context, int context_len,
                 int timestep,
                 half* d_output,
                 int H, int W);

    const UNetConfig& config() const { return cfg_; }

private:
    void allocBuffers(int H, int W);
    void freeBuffers();
    void freeWeights();

    // Time embedding: timestep -> [time_embed_dim]
    void computeTimeEmbedding(int timestep);

    UNetConfig cfg_;

    // cuDNN / cuBLAS handles
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    // Input / output convolutions
    Conv2D input_conv_;     // [latent_ch, base_ch, 3, 3]
    Conv2D output_conv_;    // [base_ch, latent_ch, 3, 3]
    GroupNormLayer output_norm_;

    // Time embedding MLP
    half* d_time_mlp_w1_;   // [base_ch, time_embed_dim]
    half* d_time_mlp_b1_;   // [time_embed_dim]
    half* d_time_mlp_w2_;   // [time_embed_dim, time_embed_dim]
    half* d_time_mlp_b2_;   // [time_embed_dim]

    // Down blocks: [num_levels][n_res_blocks] ResBlocks + AttentionBlocks
    std::vector<std::vector<ResBlock>> down_res_;
    std::vector<std::vector<AttentionBlock>> down_attn_;
    std::vector<Conv2D> downsamplers_;  // stride-2 conv for each level (except last)

    // Mid block
    ResBlock mid_res1_, mid_res2_;
    AttentionBlock mid_attn_;

    // Up blocks
    std::vector<std::vector<ResBlock>> up_res_;
    std::vector<std::vector<AttentionBlock>> up_attn_;
    std::vector<Conv2D> upsample_convs_;  // Conv after nearest upsample (except last level)

    // Workspace and temp buffers
    void* d_workspace_;
    size_t workspace_size_;
    half* d_time_embed_;        // [time_embed_dim]
    half* d_time_sinusoidal_;   // [base_ch]

    // Activation buffers (allocated based on max spatial size)
    half* d_buf1_;
    half* d_buf2_;
    half* d_buf3_;
    half* d_temp1_;
    half* d_temp2_;

    // Attention scratch
    half* d_q_buf_;
    half* d_k_buf_;
    half* d_v_buf_;
    half* d_attn_out_buf_;
    float* d_scores_buf_;

    // Skip connections storage
    std::vector<half*> skip_buffers_;
    std::vector<int> skip_channels_;
    std::vector<int> skip_heights_;
    std::vector<int> skip_widths_;

    int max_buf_size_;
    bool buffers_allocated_;
};

#endif // UNET_H
