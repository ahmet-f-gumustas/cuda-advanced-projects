#ifndef DIFFUSION_KERNELS_CUH
#define DIFFUSION_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

// ============================================================
// GroupNorm: groups over channel dim, NCHW layout
// Grid: (N * num_groups), Block: min(group_size, 1024)
// ============================================================
__global__ void group_norm_kernel(
    half* d_out, const half* d_in,
    const half* d_gamma, const half* d_beta,
    int C, int H, int W, int num_groups, float eps);

// ============================================================
// LayerNorm: over last dimension
// Grid: (rows), Block: min(D, 1024)
// ============================================================
__global__ void layer_norm_kernel(
    half* d_out, const half* d_in,
    const half* d_gamma, const half* d_beta,
    int D, float eps);

// ============================================================
// Activations (element-wise, NCHW)
// ============================================================
__global__ void silu_kernel(half* d_out, const half* d_in, int size);
__global__ void silu_inplace_kernel(half* d_data, int size);
__global__ void gelu_kernel(half* d_out, const half* d_in, int size);

// ============================================================
// Element-wise operations
// ============================================================
__global__ void add_tensors_kernel(half* d_out, const half* d_a, const half* d_b, int size);
__global__ void add_tensors_inplace_kernel(half* d_a, const half* d_b, int size);
__global__ void scale_tensor_kernel(half* d_out, const half* d_in, float scale, int size);

// Scheduler: out = a * x + b * noise (float precision for numerical stability)
__global__ void linear_combine_kernel(
    half* d_out,
    const half* d_x, float a,
    const half* d_noise, float b,
    int size);

// ============================================================
// Spatial operations (NCHW)
// ============================================================

// 2x nearest-neighbor upsample
__global__ void upsample_nearest_2x_kernel(
    half* d_out, const half* d_in,
    int C, int H_in, int W_in);

// Channel concatenation: cat(a[N,Ca,H,W], b[N,Cb,H,W]) -> out[N,Ca+Cb,H,W]
__global__ void concat_channels_kernel(
    half* d_out,
    const half* d_a, int Ca,
    const half* d_b, int Cb,
    int H, int W);

// ============================================================
// Sinusoidal time embedding: timestep -> [d_model] features
// Grid: (1), Block: (d_model / 2)
// ============================================================
__global__ void sinusoidal_embedding_kernel(
    half* d_out, int timestep, int d_model, float max_period);

// ============================================================
// Attention helpers (for spatial self-attention and cross-attention)
// ============================================================

// Batched attention scores: Q[n_heads, seq_q, head_dim] x K^T -> scores[n_heads, seq_q, seq_k]
__global__ void attention_scores_2d_kernel(
    float* d_scores,
    const half* d_q, const half* d_k,
    int n_heads, int seq_q, int seq_k, int head_dim, float scale);

// Batched softmax over last dim: scores[n_heads, seq_q, seq_k]
__global__ void softmax_2d_kernel(
    float* d_scores, int n_heads, int seq_q, int seq_k);

// Batched attention output: softmax_scores x V -> out[n_heads, seq_q, head_dim]
__global__ void attention_output_2d_kernel(
    half* d_out,
    const float* d_scores, const half* d_v,
    int n_heads, int seq_q, int seq_k, int head_dim);

// ============================================================
// Noise generation
// ============================================================
__global__ void generate_gaussian_noise_kernel(
    half* d_out, int size, curandState* d_states);

__global__ void init_curand_states_kernel(
    curandState* d_states, unsigned long long seed, int count);

// ============================================================
// Output processing
// ============================================================
__global__ void clamp_kernel(half* d_data, float min_val, float max_val, int size);
__global__ void half_to_float_kernel(float* d_out, const half* d_in, int size);
__global__ void float_to_half_kernel(half* d_out, const float* d_in, int size);

#endif // DIFFUSION_KERNELS_CUH
