#include "ops_softmax.hpp"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>
#include <float.h>

template<int BLOCK_SIZE>
__global__ void softmax_forward_kernel_fp16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len) {
    
    const int batch_head_idx = blockIdx.x;
    const int row_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (batch_head_idx >= batch_size * num_heads || row_idx >= seq_len) return;
    
    const int offset = batch_head_idx * seq_len * seq_len + row_idx * seq_len;
    
    // Find max value using warp reduction
    float local_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, __half2float(input[offset + i]));
    }
    
    // Warp reduce max
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    // Broadcast max
    __shared__ float shared_max[32];
    if (tid % WARP_SIZE == 0) {
        shared_max[tid / WARP_SIZE] = local_max;
    }
    __syncthreads();
    
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        local_max = shared_max[tid];
        #pragma unroll
        for (int offset = BLOCK_SIZE / WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
    }
    
    if (tid == 0) {
        shared_max[0] = local_max;
    }
    __syncthreads();
    
    const float max_val = shared_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        float exp_val = expf(__half2float(input[offset + i]) - max_val);
        local_sum += exp_val;
    }
    
    // Warp reduce sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Broadcast sum
    __shared__ float shared_sum[32];
    if (tid % WARP_SIZE == 0) {
        shared_sum[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();
    
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        local_sum = shared_sum[tid];
        #pragma unroll
        for (int offset = BLOCK_SIZE / WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    
    if (tid == 0) {
        shared_sum[0] = local_sum;
    }
    __syncthreads();
    
    const float sum_val = shared_sum[0];
    const float scale = 1.0f / sum_val;
    
    // Write normalized output
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        float exp_val = expf(__half2float(input[offset + i]) - max_val);
        output[offset + i] = __float2half(exp_val * scale);
    }
}

void softmax_forward(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream) {
    
    const int batch_size = input.shape()[0];
    const int num_heads = input.shape()[1];
    const int seq_len = input.shape()[2];
    
    const int threads = std::min(seq_len, 256);
    dim3 blocks(batch_size * num_heads, seq_len);
    
    if (input.dtype() == Tensor::DataType::FP16) {
        if (threads <= 256) {
            softmax_forward_kernel_fp16<256><<<blocks, threads, 0, stream>>>(
                input.data_ptr<__half>(),
                output.data_ptr<__half>(),
                batch_size, num_heads, seq_len
            );
        } else {
            softmax_forward_kernel_fp16<1024><<<blocks, 1024, 0, stream>>>(
                input.data_ptr<__half>(),
                output.data_ptr<__half>(),
                batch_size, num_heads, seq_len
            );
        }
    } else {
        throw std::runtime_error("Softmax: Only FP16 supported currently");
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// CPU reference
void softmax_cpu_reference(
    const float* input,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len) {
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                const int offset = ((b * num_heads + h) * seq_len + i) * seq_len;
                
                // Find max
                float max_val = -FLT_MAX;
                for (int j = 0; j < seq_len; ++j) {
                    max_val = std::max(max_val, input[offset + j]);
                }
                
                // Compute exp and sum
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    output[offset + j] = std::exp(input[offset + j] - max_val);
                    sum_exp += output[offset + j];
                }
                
                // Normalize
                for (int j = 0; j < seq_len; ++j) {
                    output[offset + j] /= sum_exp;
                }
            }
        }
    }
}