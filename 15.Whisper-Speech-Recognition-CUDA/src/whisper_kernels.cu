#include "whisper_kernels.cuh"
#include "cuda_utils.h"
#include <cfloat>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Audio Processing Kernels
// ============================================================

__global__ void hanning_window_kernel(float* frames, int num_frames, int frame_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * frame_length;
    if (idx >= total) return;

    int n = idx % frame_length;
    float window = 0.5f * (1.0f - cosf(2.0f * M_PI * n / (frame_length - 1)));
    frames[idx] *= window;
}

void launch_hanning_window(float* d_frames, int num_frames, int frame_length,
                           cudaStream_t stream) {
    int total = num_frames * frame_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hanning_window_kernel<<<blocks, threads, 0, stream>>>(d_frames, num_frames, frame_length);
}

__global__ void power_spectrum_kernel(const cufftComplex* fft, float* power,
                                      int num_frames, int freq_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * freq_bins;
    if (idx >= total) return;

    float re = fft[idx].x;
    float im = fft[idx].y;
    power[idx] = re * re + im * im;
}

void launch_power_spectrum(const cufftComplex* d_fft, float* d_power,
                           int num_frames, int freq_bins, cudaStream_t stream) {
    int total = num_frames * freq_bins;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    power_spectrum_kernel<<<blocks, threads, 0, stream>>>(d_fft, d_power, num_frames, freq_bins);
}

__global__ void mel_filterbank_kernel(const float* power, const float* filters,
                                      float* mel, int num_frames, int freq_bins, int n_mels) {
    int frame = blockIdx.x;
    int mel_bin = threadIdx.x;
    if (frame >= num_frames || mel_bin >= n_mels) return;

    float sum = 0.0f;
    const float* frame_power = power + frame * freq_bins;
    const float* filter_row = filters + mel_bin * freq_bins;
    for (int f = 0; f < freq_bins; f++) {
        sum += frame_power[f] * filter_row[f];
    }
    mel[frame * n_mels + mel_bin] = sum;
}

void launch_mel_filterbank(const float* d_power, const float* d_filters,
                           float* d_mel, int num_frames, int freq_bins,
                           int n_mels, cudaStream_t stream) {
    dim3 blocks(num_frames);
    dim3 threads(n_mels);
    mel_filterbank_kernel<<<blocks, threads, 0, stream>>>(d_power, d_filters, d_mel,
                                                           num_frames, freq_bins, n_mels);
}

__global__ void log_mel_kernel(float* mel, int num_frames, int n_mels, float floor_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * n_mels;
    if (idx >= total) return;

    mel[idx] = logf(fmaxf(mel[idx], floor_val));
}

void launch_log_mel(float* d_mel, int num_frames, int n_mels, float floor_val,
                    cudaStream_t stream) {
    int total = num_frames * n_mels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    log_mel_kernel<<<blocks, threads, 0, stream>>>(d_mel, num_frames, n_mels, floor_val);
}

__global__ void mel_normalize_kernel(float* mel, int num_frames, int n_mels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_mels) return;

    // Compute mean
    float sum = 0.0f;
    for (int r = 0; r < num_frames; r++) {
        sum += mel[r * n_mels + col];
    }
    float mean = sum / num_frames;

    // Compute variance
    float var_sum = 0.0f;
    for (int r = 0; r < num_frames; r++) {
        float diff = mel[r * n_mels + col] - mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrtf(var_sum / num_frames + 1e-6f);

    // Normalize
    for (int r = 0; r < num_frames; r++) {
        mel[r * n_mels + col] = (mel[r * n_mels + col] - mean) / std_dev;
    }
}

void launch_mel_normalize(float* d_mel, int num_frames, int n_mels,
                          cudaStream_t stream) {
    int threads = 256;
    int blocks = (n_mels + threads - 1) / threads;
    mel_normalize_kernel<<<blocks, threads, 0, stream>>>(d_mel, num_frames, n_mels);
}

// ============================================================
// Convolution Kernels
// ============================================================

__global__ void conv1d_kernel(const float* input, const float* weight, const float* bias,
                              float* output, int in_channels, int out_channels,
                              int length, int kernel_size, int stride, int padding,
                              int out_length) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y;
    if (out_x >= out_length || oc >= out_channels) return;

    float sum = bias[oc];
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_x = out_x * stride + k - padding;
            if (in_x >= 0 && in_x < length) {
                sum += input[ic * length + in_x] *
                       weight[(oc * in_channels + ic) * kernel_size + k];
            }
        }
    }
    output[oc * out_length + out_x] = sum;
}

void launch_conv1d(const float* d_input, const float* d_weight, const float* d_bias,
                   float* d_output, int in_channels, int out_channels,
                   int length, int kernel_size, int stride, int padding,
                   cudaStream_t stream) {
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    int threads = 256;
    int blocks_x = (out_length + threads - 1) / threads;
    dim3 blocks(blocks_x, out_channels);
    conv1d_kernel<<<blocks, threads, 0, stream>>>(d_input, d_weight, d_bias, d_output,
                                                    in_channels, out_channels, length,
                                                    kernel_size, stride, padding, out_length);
}

// ============================================================
// Normalization Kernels
// ============================================================

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta,
                                  float* output, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Compute mean using warp reduction
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += in_row[i];
    }
    // Block reduce
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    float mean = __shfl_sync(0xffffffff, sum, 0) / cols;
    // Broadcast mean to all threads
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = mean;
    __syncthreads();
    mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = in_row[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) shared[warp_id] = var_sum;
    __syncthreads();
    if (warp_id == 0) {
        var_sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = var_sum / cols;
    __syncthreads();
    float variance = s_var;

    float inv_std = rsqrtf(variance + eps);

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void launch_layer_norm(const float* d_input, const float* d_gamma, const float* d_beta,
                       float* d_output, int rows, int cols, float eps,
                       cudaStream_t stream) {
    int threads = min(256, ((cols + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    layer_norm_kernel<<<rows, threads, 0, stream>>>(d_input, d_gamma, d_beta,
                                                     d_output, rows, cols, eps);
}

// ============================================================
// Activation Kernels
// ============================================================

__global__ void gelu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = data[idx];
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    data[idx] = x * cdf;
}

void launch_gelu(float* d_data, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(d_data, n);
}

// ============================================================
// Attention Kernels
// ============================================================

__global__ void softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;

    // Find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }
    __shared__ float s_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    if (lane == 0) s_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        max_val = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_max_final;
    if (threadIdx.x == 0) s_max_final = max_val;
    __syncthreads();
    max_val = s_max_final;

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }
    __shared__ float s_sum[32];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) s_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float s_sum_final;
    if (threadIdx.x == 0) s_sum_final = sum;
    __syncthreads();
    sum = s_sum_final;

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

void launch_softmax(float* d_data, int rows, int cols, cudaStream_t stream) {
    int threads = min(256, ((cols + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    softmax_kernel<<<rows, threads, 0, stream>>>(d_data, rows, cols);
}

__global__ void masked_softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;

    // row index determines the mask boundary
    int head_size = rows / (rows / cols + 1); // approximate
    int token_idx = row % cols;

    // Apply causal mask: set future positions to -inf
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i > token_idx) {
            row_data[i] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Standard softmax on masked data
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }
    __shared__ float s_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    if (lane == 0) s_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        max_val = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_max_final;
    if (threadIdx.x == 0) s_max_final = max_val;
    __syncthreads();
    max_val = s_max_final;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }
    __shared__ float s_sum[32];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) s_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float s_sum_final;
    if (threadIdx.x == 0) s_sum_final = sum;
    __syncthreads();
    sum = s_sum_final;

    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

void launch_masked_softmax(float* d_data, int rows, int cols, cudaStream_t stream) {
    int threads = min(256, ((cols + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    masked_softmax_kernel<<<rows, threads, 0, stream>>>(d_data, rows, cols);
}

__global__ void scale_kernel(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] *= scale;
}

void launch_scale(float* d_data, float scale, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(d_data, scale, n);
}

// ============================================================
// Transpose Kernels
// ============================================================

__global__ void transpose_0213_kernel(const float* input, float* output,
                                       int seq_len, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_heads * head_dim;
    if (idx >= total) return;

    // output[h][s][d] = input[s][h][d]
    int h = idx / (seq_len * head_dim);
    int remainder = idx % (seq_len * head_dim);
    int s = remainder / head_dim;
    int d = remainder % head_dim;

    output[idx] = input[s * n_heads * head_dim + h * head_dim + d];
}

void launch_transpose_0213(const float* d_input, float* d_output,
                           int seq_len, int n_heads, int head_dim,
                           cudaStream_t stream) {
    int total = seq_len * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_0213_kernel<<<blocks, threads, 0, stream>>>(d_input, d_output,
                                                            seq_len, n_heads, head_dim);
}

__global__ void transpose_1023_kernel(const float* input, float* output,
                                       int n_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * seq_len * head_dim;
    if (idx >= total) return;

    // output[s][h][d] = input[h][s][d]
    int s = idx / (n_heads * head_dim);
    int remainder = idx % (n_heads * head_dim);
    int h = remainder / head_dim;
    int d = remainder % head_dim;

    output[idx] = input[h * seq_len * head_dim + s * head_dim + d];
}

void launch_transpose_1023(const float* d_input, float* d_output,
                           int n_heads, int seq_len, int head_dim,
                           cudaStream_t stream) {
    int total = n_heads * seq_len * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_1023_kernel<<<blocks, threads, 0, stream>>>(d_input, d_output,
                                                            n_heads, seq_len, head_dim);
}

// ============================================================
// Embedding & Positional Kernels
// ============================================================

__global__ void embedding_lookup_kernel(const int* tokens, const float* embed_table,
                                         float* output, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;

    int s = idx / d_model;
    int d = idx % d_model;
    int token_id = tokens[s];

    output[idx] = embed_table[token_id * d_model + d];
}

void launch_embedding_lookup(const int* d_tokens, const float* d_embed_table,
                             float* d_output, int seq_len, int d_model,
                             cudaStream_t stream) {
    int total = seq_len * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(d_tokens, d_embed_table,
                                                              d_output, seq_len, d_model);
}

__global__ void sinusoidal_pe_kernel(float* pe, int max_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_len * d_model;
    if (idx >= total) return;

    int pos = idx / d_model;
    int i = idx % d_model;
    int half = d_model / 2;

    float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / d_model);
    pe[idx] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
}

void launch_sinusoidal_pe(float* d_pe, int max_len, int d_model, cudaStream_t stream) {
    int total = max_len * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    sinusoidal_pe_kernel<<<blocks, threads, 0, stream>>>(d_pe, max_len, d_model);
}

__global__ void add_pe_kernel(float* data, const float* pe, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;
    data[idx] += pe[idx];
}

void launch_add_pe(float* d_data, const float* d_pe, int seq_len, int d_model,
                   cudaStream_t stream) {
    int total = seq_len * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_pe_kernel<<<blocks, threads, 0, stream>>>(d_data, d_pe, seq_len, d_model);
}

// ============================================================
// Residual & Utility Kernels
// ============================================================

__global__ void add_residual_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

void launch_add_residual(const float* d_a, const float* d_b, float* d_out, int n,
                         cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_residual_kernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_out, n);
}

__global__ void add_bias_kernel(float* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    data[idx] += bias[idx % cols];
}

void launch_add_bias(float* d_data, const float* d_bias, int rows, int cols,
                     cudaStream_t stream) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(d_data, d_bias, rows, cols);
}

__global__ void fill_zero_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = 0.0f;
}

void launch_fill_zero(float* d_data, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fill_zero_kernel<<<blocks, threads, 0, stream>>>(d_data, n);
}

__global__ void copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx];
}

void launch_copy(const float* d_src, float* d_dst, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copy_kernel<<<blocks, threads, 0, stream>>>(d_src, d_dst, n);
}
