#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "../core/types.h"

namespace deep_engine {
namespace kernels {

// Im2col operations
template<typename T>
__global__ void im2col_kernel(const T* input, T* output,
                             int batch_size, int channels,
                             int height, int width,
                             int kernel_h, int kernel_w,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w,
                             int dilation_h, int dilation_w,
                             int output_h, int output_w);

template<typename T>
__global__ void col2im_kernel(const T* input, T* output,
                             int batch_size, int channels,
                             int height, int width,
                             int kernel_h, int kernel_w,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w,
                             int dilation_h, int dilation_w,
                             int output_h, int output_w);

// Winograd convolution (F(2,3) and F(4,3))
template<typename T>
__global__ void winograd_input_transform_2x2_3x3(const T* input, T* output,
                                                 int batch, int channels,
                                                 int height, int width,
                                                 int pad_h, int pad_w);

template<typename T>
__global__ void winograd_output_transform_2x2_3x3(const T* input, T* output,
                                                  int batch, int channels,
                                                  int height, int width);

template<typename T>
__global__ void winograd_filter_transform_2x2_3x3(const T* filter, T* output,
                                                  int out_channels, int in_channels);

// Depthwise convolution
template<typename T>
__global__ void depthwise_conv2d_kernel(const T* input, const T* filter,
                                       const T* bias, T* output,
                                       int batch, int channels,
                                       int height, int width,
                                       int kernel_h, int kernel_w,
                                       int pad_h, int pad_w,
                                       int stride_h, int stride_w,
                                       int output_h, int output_w,
                                       bool use_bias);

// Dilated convolution
template<typename T>
__global__ void dilated_conv2d_kernel(const T* input, const T* filter,
                                     const T* bias, T* output,
                                     int batch, int in_channels, int out_channels,
                                     int height, int width,
                                     int kernel_h, int kernel_w,
                                     int pad_h, int pad_w,
                                     int stride_h, int stride_w,
                                     int dilation_h, int dilation_w,
                                     int output_h, int output_w,
                                     bool use_bias);

// 1x1 convolution (optimized)
template<typename T>
__global__ void conv1x1_kernel(const T* input, const T* filter,
                              const T* bias, T* output,
                              int batch, int in_channels, int out_channels,
                              int spatial_size, bool use_bias);

// INT8 quantized convolution
__global__ void quantized_conv2d_kernel(const int8_t* input, const int8_t* filter,
                                       const int32_t* bias, int8_t* output,
                                       float input_scale, float filter_scale,
                                       float output_scale,
                                       int batch, int in_channels, int out_channels,
                                       int height, int width,
                                       int kernel_h, int kernel_w,
                                       int pad_h, int pad_w,
                                       int stride_h, int stride_w,
                                       int output_h, int output_w,
                                       bool use_bias);

// Tensor Core operations (for Ampere and newer)
#if __CUDA_ARCH__ >= 800
template<typename T>
__global__ void tensor_core_conv2d_kernel(const T* input, const T* filter,
                                         const T* bias, T* output,
                                         int batch, int in_channels, int out_channels,
                                         int height, int width,
                                         int kernel_h, int kernel_w,
                                         int pad_h, int pad_w,
                                         int stride_h, int stride_w,
                                         int output_h, int output_w,
                                         bool use_bias);
#endif

// Helper functions
template<typename T>
void launch_im2col(const T* input, T* output,
                  int batch_size, int channels,
                  int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w,
                  int dilation_h, int dilation_w,
                  int output_h, int output_w,
                  cudaStream_t stream);

template<typename T>
void launch_depthwise_conv2d(const T* input, const T* filter,
                            const T* bias, T* output,
                            int batch, int channels,
                            int height, int width,
                            int kernel_h, int kernel_w,
                            int pad_h, int pad_w,
                            int stride_h, int stride_w,
                            int output_h, int output_w,
                            bool use_bias, cudaStream_t stream);

// Winograd launcher
template<typename T>
void launch_winograd_conv2d(const T* input, const T* filter,
                           const T* bias, T* output,
                           int batch, int in_channels, int out_channels,
                           int height, int width,
                           int pad_h, int pad_w,
                           bool use_bias, cudaStream_t stream,
                           T* workspace);

} // namespace kernels
} // namespace deep_engine