#include "ops_layernorm.hpp"
#include "cuda_utils.hpp"
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

template<typename T>
__device__ void welford_combine(T& mean1, T& m2_1, int& count1,
                               const T& mean2, const T& m2_2, const int& count2) {
    if (count2 == 0) return;
    
    const int count = count1 + count2;
    const T delta = mean2 - mean1;
    mean1 += delta * count2 / count;
    m2_1 += m2_2 + delta * delta * count1 * count2 / count;
    count1 = count;
}

template<typename T>
__device__ void warp_reduce_mean_var(T& mean, T& m2, int& count) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other_mean = __shfl_down_sync(0xffffffff, mean, offset);
        T other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
        int other_count = __shfl_down_sync(0xffffffff, count, offset);
        welford_combine(mean, m2, count, other_mean, other_m2, other_count);
    }
}

__global__ void layernorm_forward_kernel_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps) {
    
    const int batch_idx = blockIdx.x / seq_len;
    const int seq_idx = blockIdx.x % seq_len;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    const int offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
    
    // Welford's algorithm for mean and variance
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    int local_count = 0;
    
    // Each thread processes multiple elements
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(input[offset + i]);
        local_count++;
        float delta = val - local_mean;
        local_mean += delta / local_count;
        local_m2 += delta * (val - local_mean);
    }
    
    // Warp reduction
    warp_reduce_mean_var(local_mean, local_m2, local_count);
    
    // Broadcast from lane 0 to shared memory
    __shared__ float shared_mean[32];
    __shared__ float shared_m2[32];
    __shared__ int shared_count[32];
    
    if (lane == 0) {
        shared_mean[tid / WARP_SIZE] = local_mean;
        shared_m2[tid / WARP_SIZE] = local_m2;
        shared_count[tid / WARP_SIZE] = local_count;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < blockDim.x / WARP_SIZE) {
        local_mean = shared_mean[tid];
        local_m2 = shared_m2[tid];
        local_count = shared_count[tid];
        
        warp_reduce_mean_var(local_mean, local_m2, local_count);
        
        if (tid == 0) {
            shared_mean[0] = local_mean;
            shared_m2[0] = local_m2 / (hidden_dim - 1); // variance
        }
    }
    __syncthreads();
    
    const float mean = shared_mean[0];
    const float var = shared_m2[0];
    const float rstd = rsqrtf(var + eps);
    
    // Apply normalization
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(input[offset + i]);
        float normalized = (val - mean) * rstd;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        output[offset + i] = __float2half(scaled);
    }
}

void layernorm_forward(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    Tensor& output,
    float eps,
    cudaStream_t stream) {
    
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int hidden_dim = input.shape()[2];
    
    const int threads = std::min(hidden_dim, 1024);
    const int blocks = batch_size * seq_len;
    
    if (input.dtype() == Tensor::DataType::FP16) {
        layernorm_forward_kernel_fp16<<<blocks, threads, 0, stream>>>(
            input.data_ptr<__half>(),
            gamma.data_ptr<__half>(),
            beta.data_ptr<__half>(),
            output.data_ptr<__half>(),
            batch_size, seq_len, hidden_dim, eps
        );
    } else {
        throw std::runtime_error("LayerNorm: Only FP16 supported currently");
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// CPU reference implementation
void layernorm_cpu_reference(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps) {
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int offset = (b * seq_len + s) * hidden_dim;
            
            // Compute mean
            float mean = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                mean += input[offset + d];
            }
            mean /= hidden_dim;
            
            // Compute variance
            float var = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                float diff = input[offset + d] - mean;
                var += diff * diff;
            }
            var /= hidden_dim;
            
            // Normalize
            float rstd = 1.0f / std::sqrt(var + eps);
            for (int d = 0; d < hidden_dim; ++d) {
                float normalized = (input[offset + d] - mean) * rstd;
                output[offset + d] = normalized * gamma[d] + beta[d];
            }
        }
    }
}