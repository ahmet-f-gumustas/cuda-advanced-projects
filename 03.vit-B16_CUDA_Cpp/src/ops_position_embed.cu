#include "ops_position_embed.hpp"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>

__global__ void add_positional_embedding_kernel_fp16(
    __half* __restrict__ patches,         // [B, N, D]
    const __half* __restrict__ pos_embed, // [1, N, D]
    int batch_size,
    int seq_len,
    int hidden_dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    if (idx < total_elements) {
        const int n = (idx / hidden_dim) % seq_len;
        const int d = idx % hidden_dim;
        
        const int pos_idx = n * hidden_dim + d;
        patches[idx] = __hadd(patches[idx], pos_embed[pos_idx]);
    }
}

__global__ void add_class_token_kernel_fp16(
    const __half* __restrict__ patches,      // [B, N-1, D]
    const __half* __restrict__ class_token,  // [1, 1, D]
    __half* __restrict__ output,             // [B, N, D]
    int batch_size,
    int num_patches,
    int hidden_dim) {
    
    const int b = blockIdx.z;
    const int n = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || n >= num_patches + 1 || d >= hidden_dim) return;
    
    if (n == 0) {
        // First position is class token
        output[b * (num_patches + 1) * hidden_dim + d] = class_token[d];
    } else {
        // Copy patch embeddings
        const int src_idx = b * num_patches * hidden_dim + (n - 1) * hidden_dim + d;
        const int dst_idx = b * (num_patches + 1) * hidden_dim + n * hidden_dim + d;
        output[dst_idx] = patches[src_idx];
    }
}

void add_positional_embedding(
    Tensor& patches,
    const Tensor& pos_embed,
    cudaStream_t stream) {
    
    const int batch_size = patches.shape()[0];
    const int seq_len = patches.shape()[1];
    const int hidden_dim = patches.shape()[2];
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    if (patches.dtype() == Tensor::DataType::FP16) {
        add_positional_embedding_kernel_fp16<<<blocks, threads, 0, stream>>>(
            patches.data_ptr<__half>(),
            pos_embed.data_ptr<__half>(),
            batch_size, seq_len, hidden_dim
        );
    } else {
        throw std::runtime_error("Position embedding: Only FP16 supported currently");
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void add_class_token(
    const Tensor& patches,
    const Tensor& class_token,
    Tensor& output,
    cudaStream_t stream) {
    
    const int batch_size = patches.shape()[0];
    const int num_patches = patches.shape()[1];
    const int hidden_dim = patches.shape()[2];
    
    dim3 blocks(
        (hidden_dim + 255) / 256,
        num_patches + 1,
        batch_size
    );
    dim3 threads(256);
    
    if (patches.dtype() == Tensor::DataType::FP16) {
        add_class_token_kernel_fp16<<<blocks, threads, 0, stream>>>(
            patches.data_ptr<__half>(),
            class_token.data_ptr<__half>(),
            output.data_ptr<__half>(),
            batch_size, num_patches, hidden_dim
        );
    } else {
        throw std::runtime_error("Class token: Only FP16 supported currently");
    }
    
    CUDA_CHECK(cudaGetLastError());
}