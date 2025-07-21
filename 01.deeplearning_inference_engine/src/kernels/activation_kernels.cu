#include "../../include/kernels/activation_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace deep_engine {
namespace kernels {

// ReLU implementations
template<typename T>
__global__ void relu_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

template<typename T>
__global__ void relu_inplace_kernel(T* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

template<typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, T negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T val = input[idx];
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

// Sigmoid
template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Tanh
template<typename T>
__global__ void tanh_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// GELU - Gaussian Error Linear Unit
template<typename T>
__global__ void gelu_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const T c1 = 0.7978845608f; // sqrt(2/pi)
        const T c2 = 0.044715f;
        T x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x3)));
    }
}

// Swish: x * sigmoid(beta * x)
template<typename T>
__global__ void swish_kernel(const T* input, T* output, T beta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        output[idx] = x / (1.0f + expf(-beta * x));
    }
}

// Mish: x * tanh(softplus(x))
template<typename T>
__global__ void mish_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        T softplus = logf(1.0f + expf(x));
        output[idx] = x * tanhf(softplus);
    }
}

// HardSwish: x * ReLU6(x + 3) / 6
template<typename T>
__global__ void hardswish_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        T relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[idx] = x * relu6 / 6.0f;
    }
}

// ELU: x if x > 0, alpha * (exp(x) - 1) if x <= 0
template<typename T>
__global__ void elu_kernel(const T* input, T* output, T alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        output[idx] = x > 0 ? x : alpha * (expf(x) - 1.0f);
    }
}

// SELU: scale * (x if x > 0, alpha * (exp(x) - 1) if x <= 0)
template<typename T>
__global__ void selu_kernel(const T* input, T* output, int size) {
    const T alpha = 1.67326f;
    const T scale = 1.0507f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        output[idx] = scale * (x > 0 ? x : alpha * (expf(x) - 1.0f));
    }
}

// Softmax
template<typename T>
__global__ void softmax_kernel(const T* input, T* output, 
                              int batch_size, int classes) {
    extern __shared__ T shared_data[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* input_batch = input + batch_idx * classes;
    T* output_batch = output + batch_idx * classes;
    
    // Find max for numerical stability
    T max_val = -INFINITY;
    for (int i = tid; i < classes; i += blockDim.x) {
        max_val = fmaxf(max_val, input_batch[i]);
    }
    
    // Reduce max across threads
    shared_data[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    
    max_val = shared_data[0];
    
    // Compute exp and sum
    T sum = 0.0f;
    for (int i = tid; i < classes; i += blockDim.x) {
        T exp_val = expf(input_batch[i] - max_val);
        output_batch[i] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum
    shared_data[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    sum = shared_data[0];
    
    // Normalize
    for (int i = tid; i < classes; i += blockDim.x) {
        output_batch[i] /= sum;
    }
}

// Fused bias + ReLU
template<typename T>
__global__ void bias_relu_kernel(const T* input, const T* bias, T* output,
                                int batch, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        T val = input[idx] + bias[c];
        output[idx] = fmaxf(0.0f, val);
    }
}

// Vectorized ReLU for better performance
template<typename T, int VECTOR_SIZE>
__global__ void vectorized_relu_kernel(const T* input, T* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;
    
    if (idx + VECTOR_SIZE <= size) {
        #pragma unroll
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            output[idx + i] = fmaxf(0.0f, input[idx + i]);
        }
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

// FP16 implementations
__global__ void fp16_relu_kernel(const __half* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hmax(__float2half(0.0f), input[idx]);
    }
}

__global__ void fp16_gelu_kernel(const __half* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(input[idx]);
        const float c1 = 0.7978845608f;
        const float c2 = 0.044715f;
        float x3 = x * x * x;
        float result = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x3)));
        output[idx] = __float2half(result);
    }
}

// Launcher functions
template<typename T>
void launch_relu(const T* input, T* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

template<typename T>
void launch_leaky_relu(const T* input, T* output, T negative_slope, 
                      int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(input, output, negative_slope, size);
}

template<typename T>
void launch_sigmoid(const T* input, T* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

template<typename T>
void launch_tanh(const T* input, T* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tanh_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

template<typename T>
void launch_gelu(const T* input, T* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

template<typename T>
void launch_swish(const T* input, T* output, T beta, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    swish_kernel<<<blocks, threads, 0, stream>>>(input, output, beta, size);
}

template<typename T>
void launch_softmax(const T* input, T* output, int batch_size, 
                   int classes, cudaStream_t stream) {
    int threads = min(256, classes);
    int shared_mem_size = threads * sizeof(T);
    softmax_kernel<<<batch_size, threads, shared_mem_size, stream>>>(
        input, output, batch_size, classes);
}

template<typename T>
void launch_bias_relu(const T* input, const T* bias, T* output,
                     int batch, int channels, int spatial_size,
                     cudaStream_t stream) {
    int total_size = batch * channels * spatial_size;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    bias_relu_kernel<<<blocks, threads, 0, stream>>>(
        input, bias, output, batch, channels, spatial_size);
}

// Activation dispatcher
template<typename T>
void launch_activation(const T* input, T* output, int size,
                      ActivationKernelType type, float alpha, float beta,
                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    switch (type) {
        case ActivationKernelType::RELU:
            relu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
        case ActivationKernelType::LEAKY_RELU:
            leaky_relu_kernel<<<blocks, threads, 0, stream>>>(input, output, alpha, size);
            break;
        case ActivationKernelType::SIGMOID:
            sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
        case ActivationKernelType::TANH:
            tanh_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
        case ActivationKernelType::GELU:
            gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
        case ActivationKernelType::SWISH:
            swish_kernel<<<blocks, threads, 0, stream>>>(input, output, beta, size);
            break;
        case ActivationKernelType::MISH:
            mish_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
        case ActivationKernelType::HARDSWISH:
            hardswish_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
        case ActivationKernelType::ELU:
            elu_kernel<<<blocks, threads, 0, stream>>>(input, output, alpha, size);
            break;
        case ActivationKernelType::SELU:
            selu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
            break;
    }
}

// Explicit instantiations
template void launch_relu<float>(const float*, float*, int, cudaStream_t);
template void launch_leaky_relu<float>(const float*, float*, float, int, cudaStream_t);
template void launch_sigmoid<float>(const float*, float*, int, cudaStream_t);
template void launch_tanh<float>(const float*, float*, int, cudaStream_t);
template void launch_gelu<float>(const float*, float*, int, cudaStream_t);
template void launch_swish<float>(const float*, float*, float, int, cudaStream_t);
template void launch_softmax<float>(const float*, float*, int, int, cudaStream_t);
template void launch_bias_relu<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_activation<float>(const float*, float*, int, ActivationKernelType, float, float, cudaStream_t);

} // namespace kernels
} // namespace deep_engine