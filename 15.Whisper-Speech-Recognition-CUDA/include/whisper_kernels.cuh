#ifndef WHISPER_KERNELS_CUH
#define WHISPER_KERNELS_CUH

#include <cuda_runtime.h>
#include <cufft.h>

// ============================================================
// Audio Processing Kernels
// ============================================================

// Apply Hanning window to audio frames [num_frames, frame_length]
void launch_hanning_window(float* d_frames, int num_frames, int frame_length,
                           cudaStream_t stream = 0);

// Compute power spectrum from complex FFT output
// d_fft: [num_frames, freq_bins] complex, d_power: [num_frames, freq_bins] real
void launch_power_spectrum(const cufftComplex* d_fft, float* d_power,
                           int num_frames, int freq_bins, cudaStream_t stream = 0);

// Apply mel filterbank: d_power[num_frames, freq_bins] x d_filters[n_mels, freq_bins]
// -> d_mel[num_frames, n_mels]
void launch_mel_filterbank(const float* d_power, const float* d_filters,
                           float* d_mel, int num_frames, int freq_bins,
                           int n_mels, cudaStream_t stream = 0);

// Log mel: d_mel = log(max(d_mel, floor))
void launch_log_mel(float* d_mel, int num_frames, int n_mels, float floor_val = 1e-10f,
                    cudaStream_t stream = 0);

// Normalize mel spectrogram to zero mean, unit variance per feature
void launch_mel_normalize(float* d_mel, int num_frames, int n_mels,
                          cudaStream_t stream = 0);

// ============================================================
// Convolution Kernels
// ============================================================

// 1D convolution: input[in_ch, length] -> output[out_ch, out_length]
void launch_conv1d(const float* d_input, const float* d_weight, const float* d_bias,
                   float* d_output, int in_channels, int out_channels,
                   int length, int kernel_size, int stride, int padding,
                   cudaStream_t stream = 0);

// ============================================================
// Normalization Kernels
// ============================================================

// Layer normalization: each row of [rows, cols] is normalized independently
void launch_layer_norm(const float* d_input, const float* d_gamma, const float* d_beta,
                       float* d_output, int rows, int cols, float eps = 1e-5f,
                       cudaStream_t stream = 0);

// ============================================================
// Activation Kernels
// ============================================================

// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void launch_gelu(float* d_data, int n, cudaStream_t stream = 0);

// ============================================================
// Attention Kernels
// ============================================================

// Row-wise softmax: each row of [rows, cols]
void launch_softmax(float* d_data, int rows, int cols, cudaStream_t stream = 0);

// Causal masked softmax: apply -inf mask to upper triangle then softmax
void launch_masked_softmax(float* d_data, int rows, int cols, cudaStream_t stream = 0);

// Scale attention scores by 1/sqrt(head_dim)
void launch_scale(float* d_data, float scale, int n, cudaStream_t stream = 0);

// Transpose [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
void launch_transpose_0213(const float* d_input, float* d_output,
                           int seq_len, int n_heads, int head_dim,
                           cudaStream_t stream = 0);

// Transpose back [n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
void launch_transpose_1023(const float* d_input, float* d_output,
                           int n_heads, int seq_len, int head_dim,
                           cudaStream_t stream = 0);

// ============================================================
// Embedding & Positional Kernels
// ============================================================

// Token embedding lookup: d_tokens[seq_len] -> d_output[seq_len, d_model]
void launch_embedding_lookup(const int* d_tokens, const float* d_embed_table,
                             float* d_output, int seq_len, int d_model,
                             cudaStream_t stream = 0);

// Sinusoidal positional encoding: fill d_pe[max_len, d_model]
void launch_sinusoidal_pe(float* d_pe, int max_len, int d_model,
                          cudaStream_t stream = 0);

// Add positional encoding: d_data[seq_len, d_model] += d_pe[seq_len, d_model]
void launch_add_pe(float* d_data, const float* d_pe, int seq_len, int d_model,
                   cudaStream_t stream = 0);

// ============================================================
// Residual & Utility Kernels
// ============================================================

// Add residual: d_out[i] = d_a[i] + d_b[i]
void launch_add_residual(const float* d_a, const float* d_b, float* d_out, int n,
                         cudaStream_t stream = 0);

// Add bias: d_data[row, col] += d_bias[col]
void launch_add_bias(float* d_data, const float* d_bias, int rows, int cols,
                     cudaStream_t stream = 0);

// Fill with zeros
void launch_fill_zero(float* d_data, int n, cudaStream_t stream = 0);

// Copy kernel
void launch_copy(const float* d_src, float* d_dst, int n, cudaStream_t stream = 0);

#endif // WHISPER_KERNELS_CUH
