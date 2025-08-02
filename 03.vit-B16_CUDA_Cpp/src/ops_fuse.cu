#include "ops_fuse.hpp"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>
#include <curand_kernel.h>

__global__ void fused_bias_residual_kernel_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    if (idx < total_elements) {
        const int d = idx % hidden_dim;
        
        // Load values
        float val = __half2float(input[idx]);
        float bias_val = __half2float(bias[d]);
        float res_val = __half2float(residual[idx]);
        
        // Fused operation: output = input + bias + residual
        float result = val + bias_val + res_val;
        
        output[idx] = __float2half(result);
    }
}

__global__ void fused_bias_residual_dropout_kernel_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float dropout_prob,
    unsigned long long seed) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    if (idx < total_elements) {
        const int d = idx % hidden_dim;
        
        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Load values
        float val = __half2float(input[idx]);
        float bias_val = __half2float(bias[d]);
        float res_val = __half2float(residual[idx]);
        
        // Apply dropout
        float random = curand_uniform(&state);
        float scale = 1.0f / (1.0f - dropout_prob);
        float dropout_mask = (random > dropout_prob) ? scale : 0.0f;
        
        // Fused operation
        float result = val * dropout_mask + bias_val + res_val;
        
        output[idx] = __float2half(result);
    }
}

void fused_bias_residual(
    const Tensor& input,
    const Tensor& bias,
    const Tensor& residual,
    Tensor& output,
    cudaStream_t stream) {
    
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int hidden_dim = input.shape()[2];
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    if (input.dtype() == Tensor::DataType::FP16) {
        fused_bias_residual_kernel_fp16<<<blocks, threads, 0, stream>>>(
            input.data_ptr<__half>(),
            bias.data_ptr<__half>(),
            residual.data_ptr<__half>(),
            output.data_ptr<__half>(),
            batch_size, seq_len, hidden_dim
        );
    } else {
        throw std::runtime_error("Fused ops: Only FP16 supported currently");
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void fused_bias_residual_dropout(
    const Tensor& input,
    const Tensor& bias,
    const Tensor& residual,
    Tensor& output,
    float dropout_prob,
    bool training,
    cudaStream_t stream) {
    
    if (!training || dropout_prob == 0.0f) {
        fused_bias_residual(input, bias, residual, output, stream);
        return;
    }
    
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int hidden_dim = input.shape()[2];
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    unsigned long long seed = 42; // Fixed seed for reproducibility
    
    if (input.dtype() == Tensor::DataType::FP16) {
        fused_bias_residual_dropout_kernel_fp16<<<blocks, threads, 0, stream>>>(
            input.data_ptr<__half>(),
            bias.data_ptr<__half>(),
            residual.data_ptr<__half>(),
            output.data_ptr<__half>(),
            batch_size, seq_len, hidden_dim,
            dropout_prob, seed
        );
    } else {
        throw std::runtime_error("Fused ops: Only FP16 supported currently");
    }
    
    CUDA_CHECK(cudaGetLastError());
}