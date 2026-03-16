#include "diffusion_kernels.cuh"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cfloat>

// ============================================================
// GroupNorm kernel
// Each block handles one (sample, group) pair
// Layout: NCHW, C divided into num_groups
// ============================================================
__global__ void group_norm_kernel(
    half* d_out, const half* d_in,
    const half* d_gamma, const half* d_beta,
    int C, int H, int W, int num_groups, float eps)
{
    int group = blockIdx.x;  // Which group (across batch * num_groups)
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int hw = i % (H * W);
        int c = group * channels_per_group + c_local;
        int idx = c * H * W + hw;
        sum += __half2float(d_in[idx]);
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float s_partial[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_partial[warp_id] = sum;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < num_warps) sum = s_partial[threadIdx.x];
    else sum = 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float s_mean, s_var;
    if (threadIdx.x == 0) s_mean = sum / group_size;
    __syncthreads();

    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int hw = i % (H * W);
        int c = group * channels_per_group + c_local;
        int idx = c * H * W + hw;
        float val = __half2float(d_in[idx]) - mean;
        var_sum += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);

    if (lane == 0) s_partial[warp_id] = var_sum;
    __syncthreads();

    if (threadIdx.x < num_warps) var_sum = s_partial[threadIdx.x];
    else var_sum = 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);

    if (threadIdx.x == 0) s_var = var_sum / group_size;
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int hw = i % (H * W);
        int c = group * channels_per_group + c_local;
        int idx = c * H * W + hw;
        float val = (__half2float(d_in[idx]) - mean) * inv_std;
        float gamma = __half2float(d_gamma[c]);
        float beta = __half2float(d_beta[c]);
        d_out[idx] = __float2half(val * gamma + beta);
    }
}

// ============================================================
// LayerNorm kernel (for CLIP encoder)
// Each block normalizes one row of [rows, D]
// ============================================================
__global__ void layer_norm_kernel(
    half* d_out, const half* d_in,
    const half* d_gamma, const half* d_beta,
    int D, float eps)
{
    int row = blockIdx.x;
    const half* in_row = d_in + row * D;
    half* out_row = d_out + row * D;

    // Mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        sum += __half2float(in_row[i]);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float s_partial[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_partial[warp_id] = sum;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < num_warps) sum = s_partial[threadIdx.x];
    else sum = 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float s_mean, s_var;
    if (threadIdx.x == 0) s_mean = sum / D;
    __syncthreads();

    float mean = s_mean;

    // Variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = __half2float(in_row[i]) - mean;
        var_sum += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);

    if (lane == 0) s_partial[warp_id] = var_sum;
    __syncthreads();

    if (threadIdx.x < num_warps) var_sum = s_partial[threadIdx.x];
    else var_sum = 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);

    if (threadIdx.x == 0) s_var = var_sum / D;
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = (__half2float(in_row[i]) - mean) * inv_std;
        float gamma = __half2float(d_gamma[i]);
        float beta = __half2float(d_beta[i]);
        out_row[i] = __float2half(val * gamma + beta);
    }
}

// ============================================================
// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================
__global__ void silu_kernel(half* d_out, const half* d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(d_in[idx]);
        float s = x / (1.0f + expf(-x));
        d_out[idx] = __float2half(s);
    }
}

__global__ void silu_inplace_kernel(half* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(d_data[idx]);
        d_data[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

// ============================================================
// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================
__global__ void gelu_kernel(half* d_out, const half* d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(d_in[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        d_out[idx] = __float2half(x * cdf);
    }
}

// ============================================================
// Element-wise operations
// ============================================================
__global__ void add_tensors_kernel(half* d_out, const half* d_a, const half* d_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_out[idx] = __float2half(__half2float(d_a[idx]) + __half2float(d_b[idx]));
}

__global__ void add_tensors_inplace_kernel(half* d_a, const half* d_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_a[idx] = __float2half(__half2float(d_a[idx]) + __half2float(d_b[idx]));
}

__global__ void scale_tensor_kernel(half* d_out, const half* d_in, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_out[idx] = __float2half(__half2float(d_in[idx]) * scale);
}

__global__ void linear_combine_kernel(
    half* d_out,
    const half* d_x, float a,
    const half* d_noise, float b,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = a * __half2float(d_x[idx]) + b * __half2float(d_noise[idx]);
        d_out[idx] = __float2half(val);
    }
}

// ============================================================
// 2x nearest-neighbor upsample (NCHW)
// Input: [1, C, H_in, W_in] -> Output: [1, C, 2*H_in, 2*W_in]
// ============================================================
__global__ void upsample_nearest_2x_kernel(
    half* d_out, const half* d_in,
    int C, int H_in, int W_in)
{
    int H_out = H_in * 2;
    int W_out = W_in * 2;
    int total = C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx / (H_out * W_out);
        int rem = idx % (H_out * W_out);
        int h_out = rem / W_out;
        int w_out = rem % W_out;

        int h_in = h_out / 2;
        int w_in = w_out / 2;

        int in_idx = c * H_in * W_in + h_in * W_in + w_in;
        d_out[idx] = d_in[in_idx];
    }
}

// ============================================================
// Channel concatenation
// cat(a[1,Ca,H,W], b[1,Cb,H,W]) -> out[1,Ca+Cb,H,W]
// ============================================================
__global__ void concat_channels_kernel(
    half* d_out,
    const half* d_a, int Ca,
    const half* d_b, int Cb,
    int H, int W)
{
    int total = (Ca + Cb) * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx / (H * W);
        int hw = idx % (H * W);
        if (c < Ca) {
            d_out[idx] = d_a[c * H * W + hw];
        } else {
            d_out[idx] = d_b[(c - Ca) * H * W + hw];
        }
    }
}

// ============================================================
// Sinusoidal time embedding
// Produces [d_model] vector: first half = sin, second half = cos
// ============================================================
__global__ void sinusoidal_embedding_kernel(
    half* d_out, int timestep, int d_model, float max_period)
{
    int idx = threadIdx.x;
    int half_dim = d_model / 2;
    if (idx < half_dim) {
        float freq = expf(-logf(max_period) * (float)idx / (float)half_dim);
        float angle = (float)timestep * freq;
        d_out[idx] = __float2half(sinf(angle));
        d_out[idx + half_dim] = __float2half(cosf(angle));
    }
}

// ============================================================
// 2D Attention scores: Q * K^T
// Q: [n_heads, seq_q, head_dim]
// K: [n_heads, seq_k, head_dim]
// scores: [n_heads, seq_q, seq_k]
// ============================================================
__global__ void attention_scores_2d_kernel(
    float* d_scores,
    const half* d_q, const half* d_k,
    int n_heads, int seq_q, int seq_k, int head_dim, float scale)
{
    int h = blockIdx.x;     // head index
    int q = blockIdx.y;     // query position
    int k = threadIdx.x;    // key position (iterate if seq_k > blockDim.x)

    if (h >= n_heads || q >= seq_q) return;

    const half* q_ptr = d_q + h * seq_q * head_dim + q * head_dim;
    const half* k_base = d_k + h * seq_k * head_dim;

    for (int ki = k; ki < seq_k; ki += blockDim.x) {
        const half* k_ptr = k_base + ki * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        d_scores[h * seq_q * seq_k + q * seq_k + ki] = dot * scale;
    }
}

// ============================================================
// 2D Softmax over last dim
// scores: [n_heads, seq_q, seq_k]
// ============================================================
__global__ void softmax_2d_kernel(
    float* d_scores, int n_heads, int seq_q, int seq_k)
{
    int h = blockIdx.x;
    int q = blockIdx.y;
    if (h >= n_heads || q >= seq_q) return;

    float* row = d_scores + h * seq_q * seq_k + q * seq_k;

    // Find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}

// ============================================================
// Attention output: scores * V
// scores: [n_heads, seq_q, seq_k]
// V: [n_heads, seq_k, head_dim]
// out: [n_heads, seq_q, head_dim]
// ============================================================
__global__ void attention_output_2d_kernel(
    half* d_out,
    const float* d_scores, const half* d_v,
    int n_heads, int seq_q, int seq_k, int head_dim)
{
    int h = blockIdx.x;
    int q = blockIdx.y;
    int d = threadIdx.x;   // head_dim index

    if (h >= n_heads || q >= seq_q || d >= head_dim) return;

    const float* score_row = d_scores + h * seq_q * seq_k + q * seq_k;
    const half* v_base = d_v + h * seq_k * head_dim;

    float val = 0.0f;
    for (int ki = 0; ki < seq_k; ki++) {
        val += score_row[ki] * __half2float(v_base[ki * head_dim + d]);
    }

    d_out[h * seq_q * head_dim + q * head_dim + d] = __float2half(val);
}

// ============================================================
// Noise generation
// ============================================================
__global__ void init_curand_states_kernel(
    curandState* d_states, unsigned long long seed, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        curand_init(seed, idx, 0, &d_states[idx]);
}

__global__ void generate_gaussian_noise_kernel(
    half* d_out, int size, curandState* d_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState local_state = d_states[idx % 1024];
        float val = curand_normal(&local_state);
        d_out[idx] = __float2half(val);
        if (idx < 1024) d_states[idx] = local_state;
    }
}

// ============================================================
// Output processing
// ============================================================
__global__ void clamp_kernel(half* d_data, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(d_data[idx]);
        val = fmaxf(min_val, fminf(max_val, val));
        d_data[idx] = __float2half(val);
    }
}

__global__ void half_to_float_kernel(float* d_out, const half* d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_out[idx] = __half2float(d_in[idx]);
}

__global__ void float_to_half_kernel(half* d_out, const float* d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_out[idx] = __float2half(d_in[idx]);
}
