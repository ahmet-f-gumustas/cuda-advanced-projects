#include "../../include/kernels/conv_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace deep_engine {
namespace kernels {

// Im2col kernel implementation
template<typename T>
__global__ void im2col_kernel(const T* input, T* output,
                             int batch_size, int channels,
                             int height, int width,
                             int kernel_h, int kernel_w,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w,
                             int dilation_h, int dilation_w,
                             int output_h, int output_w) {
    const int output_size = output_h * output_w;
    const int grid_size = gridDim.x * blockDim.x;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < batch_size * channels * kernel_h * kernel_w * output_size;
         index += grid_size) {
        
        const int w_out = index % output_w;
        const int h_out = (index / output_w) % output_h;
        const int kw = (index / output_w / output_h) % kernel_w;
        const int kh = (index / output_w / output_h / kernel_w) % kernel_h;
        const int c = (index / output_w / output_h / kernel_w / kernel_h) % channels;
        const int batch = index / output_w / output_h / kernel_w / kernel_h / channels;
        
        const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
        const int w_in = w_out * stride_w - pad_w + kw * dilation_w;
        
        const int output_idx = ((batch * channels + c) * kernel_h * kernel_w + 
                               kh * kernel_w + kw) * output_size + 
                               h_out * output_w + w_out;
        
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            const int input_idx = ((batch * channels + c) * height + h_in) * width + w_in;
            output[output_idx] = input[input_idx];
        } else {
            output[output_idx] = static_cast<T>(0);
        }
    }
}

// Depthwise convolution kernel
template<typename T>
__global__ void depthwise_conv2d_kernel(const T* input, const T* filter,
                                       const T* bias, T* output,
                                       int batch, int channels,
                                       int height, int width,
                                       int kernel_h, int kernel_w,
                                       int pad_h, int pad_w,
                                       int stride_h, int stride_w,
                                       int output_h, int output_w,
                                       bool use_bias) {
    const int out_size = output_h * output_w;
    const int total_size = batch * channels * out_size;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_size;
         idx += gridDim.x * blockDim.x) {
        
        const int w_out = idx % output_w;
        const int h_out = (idx / output_w) % output_h;
        const int c = (idx / out_size) % channels;
        const int b = idx / (channels * out_size);
        
        T sum = 0;
        
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_in = h_out * stride_h - pad_h + kh;
                const int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const int input_idx = ((b * channels + c) * height + h_in) * width + w_in;
                    const int filter_idx = (c * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * filter[filter_idx];
                }
            }
        }
        
        if (use_bias) {
            sum += bias[c];
        }
        
        output[idx] = sum;
    }
}

// 1x1 convolution optimized kernel
template<typename T>
__global__ void conv1x1_kernel(const T* input, const T* filter,
                              const T* bias, T* output,
                              int batch, int in_channels, int out_channels,
                              int spatial_size, bool use_bias) {
    extern __shared__ T shared_mem[];
    T* shared_filter = shared_mem;
    
    const int tid = threadIdx.x;
    const int spatial_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    // Load filter tile to shared memory
    for (int i = tid; i < in_channels * out_channels; i += blockDim.x) {
        shared_filter[i] = filter[i];
    }
    __syncthreads();
    
    // Compute output
    for (int oc = tid; oc < out_channels; oc += blockDim.x) {
        T sum = 0;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            const int input_idx = (batch_idx * in_channels + ic) * spatial_size + spatial_idx;
            const int filter_idx = oc * in_channels + ic;
            sum += input[input_idx] * shared_filter[filter_idx];
        }
        
        if (use_bias) {
            sum += bias[oc];
        }
        
        const int output_idx = (batch_idx * out_channels + oc) * spatial_size + spatial_idx;
        output[output_idx] = sum;
    }
}

// Winograd F(2,3) input transform
template<typename T>
__global__ void winograd_input_transform_2x2_3x3(const T* input, T* output,
                                                 int batch, int channels,
                                                 int height, int width,
                                                 int pad_h, int pad_w) {
    const int tile_h = (height + pad_h * 2 - 2) / 2;
    const int tile_w = (width + pad_w * 2 - 2) / 2;
    const int num_tiles = tile_h * tile_w;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * num_tiles) return;
    
    const int tile_idx = idx % num_tiles;
    const int c = (idx / num_tiles) % channels;
    const int b = idx / (num_tiles * channels);
    
    const int tile_y = tile_idx / tile_w;
    const int tile_x = tile_idx % tile_w;
    
    // Load 4x4 input tile
    T tile[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int y = tile_y * 2 - pad_h + i;
            int x = tile_x * 2 - pad_w + j;
            
            if (y >= 0 && y < height && x >= 0 && x < width) {
                tile[i][j] = input[((b * channels + c) * height + y) * width + x];
            } else {
                tile[i][j] = 0;
            }
        }
    }
    
    // Apply transform: B^T * tile * B
    // B^T = [1  0 -1  0]
    //       [0  1  1  0]
    //       [0 -1  1  0]
    //       [0  1  0 -1]
    
    T temp[4][4];
    
    // Compute B^T * tile
    for (int i = 0; i < 4; ++i) {
        temp[0][i] = tile[0][i] - tile[2][i];
        temp[1][i] = tile[1][i] + tile[2][i];
        temp[2][i] = -tile[1][i] + tile[2][i];
        temp[3][i] = tile[1][i] - tile[3][i];
    }
    
    // Compute temp * B
    T result[4][4];
    for (int i = 0; i < 4; ++i) {
        result[i][0] = temp[i][0] - temp[i][2];
        result[i][1] = temp[i][1] + temp[i][2];
        result[i][2] = -temp[i][1] + temp[i][2];
        result[i][3] = temp[i][1] - temp[i][3];
    }
    
    // Write output
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int out_idx = ((i * 4 + j) * batch * channels + b * channels + c) * num_tiles + tile_idx;
            output[out_idx] = result[i][j];
        }
    }
}

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
                                       bool use_bias) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_size = output_h * output_w;
    const int total_size = batch * out_channels * out_size;
    
    if (out_idx >= total_size) return;
    
    const int w_out = out_idx % output_w;
    const int h_out = (out_idx / output_w) % output_h;
    const int oc = (out_idx / out_size) % out_channels;
    const int b = out_idx / (out_channels * out_size);
    
    int32_t sum = 0;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_in = h_out * stride_h - pad_h + kh;
                const int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const int input_idx = ((b * in_channels + ic) * height + h_in) * width + w_in;
                    const int filter_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += static_cast<int32_t>(input[input_idx]) * static_cast<int32_t>(filter[filter_idx]);
                }
            }
        }
    }
    
    if (use_bias) {
        sum += bias[oc];
    }
    
    // Dequantize and requantize
    float result = sum * input_scale * filter_scale / output_scale;
    result = fmaxf(-128.0f, fminf(127.0f, roundf(result)));
    
    output[out_idx] = static_cast<int8_t>(result);
}

// Launcher functions
template<typename T>
void launch_im2col(const T* input, T* output,
                  int batch_size, int channels,
                  int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w,
                  int dilation_h, int dilation_w,
                  int output_h, int output_w,
                  cudaStream_t stream) {
    const int total_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    im2col_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, height, width,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, output_h, output_w);
}

template<typename T>
void launch_depthwise_conv2d(const T* input, const T* filter,
                            const T* bias, T* output,
                            int batch, int channels,
                            int height, int width,
                            int kernel_h, int kernel_w,
                            int pad_h, int pad_w,
                            int stride_h, int stride_w,
                            int output_h, int output_w,
                            bool use_bias, cudaStream_t stream) {
    const int total_size = batch * channels * output_h * output_w;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    depthwise_conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input, filter, bias, output,
        batch, channels, height, width,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, output_h, output_w,
        use_bias);
}

// Explicit instantiations
template void launch_im2col<float>(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_im2col<__half>(const __half*, __half*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

template void launch_depthwise_conv2d<float>(const float*, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, bool, cudaStream_t);
template void launch_depthwise_conv2d<__half>(const __half*, const __half*, const __half*, __half*, int, int, int, int, int, int, int, int, int, int, int, int, bool, cudaStream_t);

} // namespace kernels
} // namespace deep_engine