#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../core/types.h"

namespace deep_engine {
namespace kernels {

// ReLU variants
template<typename T>
__global__ void relu_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void relu_inplace_kernel(T* data, int size);

template<typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, T negative_slope, int size);

template<typename T>
__global__ void prelu_kernel(const T* input, const T* slopes, T* output, 
                            int size, int channels, int channel_size);

// Sigmoid and Tanh
template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void tanh_kernel(const T* input, T* output, int size);

// Advanced activations
template<typename T>
__global__ void gelu_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void swish_kernel(const T* input, T* output, T beta, int size);

template<typename T>
__global__ void mish_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void hardswish_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void elu_kernel(const T* input, T* output, T alpha, int size);

template<typename T>
__global__ void selu_kernel(const T* input, T* output, int size);

// Softmax
template<typename T>
__global__ void softmax_kernel(const T* input, T* output, 
                              int batch_size, int classes);

template<typename T>
__global__ void log_softmax_kernel(const T* input, T* output,
                                  int batch_size, int classes);

// Fused operations
template<typename T>
__global__ void bias_relu_kernel(const T* input, const T* bias, T* output,
                                int batch, int channels, int spatial_size);

template<typename T>
__global__ void bias_add_kernel(const T* input, const T* bias, T* output,
                               int batch, int channels, int spatial_size);

// Dropout
template<typename T>
__global__ void dropout_kernel(const T* input, T* output, const float* mask,
                              float dropout_prob, float scale, int size);

template<typename T>
__global__ void dropout_inplace_kernel(T* data, const float* mask,
                                      float dropout_prob, float scale, int size);

// INT8 quantized activations
__global__ void quantized_relu_kernel(const int8_t* input, int8_t* output,
                                     int8_t zero_point, int size);

__global__ void quantized_leaky_relu_kernel(const int8_t* input, int8_t* output,
                                           int8_t zero_point, float negative_slope,
                                           float scale, int size);

// Mixed precision activations
__global__ void fp16_relu_kernel(const __half* input, __half* output, int size);

__global__ void fp16_gelu_kernel(const __half* input, __half* output, int size);

// Vectorized operations for better performance
template<typename T, int VECTOR_SIZE>
__global__ void vectorized_relu_kernel(const T* input, T* output, int size);

template<typename T, int VECTOR_SIZE>
__global__ void vectorized_sigmoid_kernel(const T* input, T* output, int size);

// Launcher functions
template<typename T>
void launch_relu(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_leaky_relu(const T* input, T* output, T negative_slope, 
                      int size, cudaStream_t stream);

template<typename T>
void launch_sigmoid(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_tanh(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_gelu(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_swish(const T* input, T* output, T beta, int size, cudaStream_t stream);

template<typename T>
void launch_softmax(const T* input, T* output, int batch_size, 
                   int classes, cudaStream_t stream);

template<typename T>
void launch_bias_relu(const T* input, const T* bias, T* output,
                     int batch, int channels, int spatial_size,
                     cudaStream_t stream);

// Activation function dispatcher
enum class ActivationKernelType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    GELU,
    SWISH,
    MISH,
    HARDSWISH,
    ELU,
    SELU
};

template<typename T>
void launch_activation(const T* input, T* output, int size,
                      ActivationKernelType type, float alpha, float beta,
                      cudaStream_t stream);

} // namespace kernels
} // namespace deep_engine