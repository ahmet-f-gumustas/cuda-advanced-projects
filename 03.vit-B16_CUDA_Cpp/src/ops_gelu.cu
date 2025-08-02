#include "ops_gelu.hpp"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>

// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// Fast approximation using tanh
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_fast(float x) {
    const float c = 0.797884560802865f; // sqrt(2/π)
    const float a = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_forward_kernel_fp16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    size_t size,
    bool use_fast) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(input[idx]);
        float result = use_fast ? gelu_fast(val) : gelu_exact(val);
        output[idx] = __float2half(result);
    }
}

__global__ void gelu_forward_kernel_fp32(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t size,
    bool use_fast) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = use_fast ? gelu_fast(input[idx]) : gelu_exact(input[idx]);
    }
}

void gelu_forward(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream) {
    
    const size_t size = input.size();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    if (input.dtype() == Tensor::DataType::FP16) {
        gelu_forward_kernel_fp16<<<blocks, threads, 0, stream>>>(
            input.data_ptr<__half>(),
            output.data_ptr<__half>(),
            size,
            false  // use exact
        );
    } else {
        gelu_forward_kernel_fp32<<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size,
            false  // use exact
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void gelu_fast_forward(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream) {
    
    const size_t size = input.size();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    if (input.dtype() == Tensor::DataType::FP16) {
        gelu_forward_kernel_fp16<<<blocks, threads, 0, stream>>>(
            input.data_ptr<__half>(),
            output.data_ptr<__half>(),
            size,
            true  // use fast
        );
    } else {
        gelu_forward_kernel_fp32<<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size,
            true  // use fast
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}