#include "ops_patch_embed.hpp"
#include "cublaslt_utils.hpp"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>

__global__ void unfold_patches_kernel_fp16(
    const __half* __restrict__ images,  // [B, 3, H, W]
    __half* __restrict__ patches,        // [B, H/P * W/P, 3*P*P]
    int batch_size,
    int channels,
    int height,
    int width,
    int patch_size) {
    
    const int num_patches_h = height / patch_size;
    const int num_patches_w = width / patch_size;
    const int patch_dim = channels * patch_size * patch_size;
    
    const int b = blockIdx.z;
    const int patch_idx = blockIdx.y * gridDim.x + blockIdx.x;
    const int elem_idx = threadIdx.x;
    
    if (b >= batch_size || patch_idx >= num_patches_h * num_patches_w || 
        elem_idx >= patch_dim) return;
    
    const int ph = patch_idx / num_patches_w;
    const int pw = patch_idx % num_patches_w;
    
    // Calculate position in patch
    const int c = elem_idx / (patch_size * patch_size);
    const int patch_pos = elem_idx % (patch_size * patch_size);
    const int py = patch_pos / patch_size;
    const int px = patch_pos % patch_size;
    
    // Calculate global position in image
    const int y = ph * patch_size + py;
    const int x = pw * patch_size + px;
    
    // Read from image and write to patches
    const int img_idx = ((b * channels + c) * height + y) * width + x;
    const int patch_out_idx = (b * num_patches_h * num_patches_w + patch_idx) * patch_dim + elem_idx;
    
    patches[patch_out_idx] = images[img_idx];
}

__global__ void add_bias_kernel_fp16(
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    int batch_size,
    int num_patches,
    int hidden_dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_patches * hidden_dim;
    
    if (idx < total_elements) {
        const int d = idx % hidden_dim;
        output[idx] = __hadd(output[idx], bias[d]);
    }
}

void patch_embed_unfold_gemm(
    const Tensor& images,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output,
    Tensor& workspace,
    int patch_size,
    cublasLtHandle_t cublaslt_handle,
    cudaStream_t stream) {
    
    const int batch_size = images.shape()[0];
    const int channels = images.shape()[1];
    const int height = images.shape()[2];
    const int width = images.shape()[3];
    const int hidden_dim = weight.shape()[0];
    
    const int num_patches_h = height / patch_size;
    const int num_patches_w = width / patch_size;
    const int num_patches = num_patches_h * num_patches_w;
    const int patch_dim = channels * patch_size * patch_size;
    
    // Unfold patches
    dim3 blocks(num_patches_w, num_patches_h, batch_size);
    int threads = std::min(patch_dim, 256);
    
    unfold_patches_kernel_fp16<<<blocks, threads, 0, stream>>>(
        images.data_ptr<__half>(),
        workspace.data_ptr<__half>(),
        batch_size, channels, height, width, patch_size
    );
    
    // GEMM: patches @ weight.T + bias
    // Input: [B * num_patches, patch_dim]
    // Weight: [hidden_dim, patch_dim]
    // Output: [B * num_patches, hidden_dim]
    
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    gemm_fp16_tensor_core(
        cublaslt_handle,
        CUBLAS_OP_T,      // weight is transposed
        CUBLAS_OP_N,      // patches not transposed
        hidden_dim,       // m
        batch_size * num_patches,  // n
        patch_dim,        // k
        &alpha,
        weight.data_ptr<__half>(), patch_dim,    // lda
        workspace.data_ptr<__half>(), patch_dim, // ldb
        &beta,
        output.data_ptr<__half>(), hidden_dim,   // ldc
        nullptr, 0,       // workspace
        stream
    );
    
    // Add bias
    const int total_elements = batch_size * num_patches * hidden_dim;
    const int bias_threads = 256;
    const int bias_blocks = (total_elements + bias_threads - 1) / bias_threads;
    
    add_bias_kernel_fp16<<<bias_blocks, bias_threads, 0, stream>>>(
        output.data_ptr<__half>(),
        bias.data_ptr<__half>(),
        batch_size, num_patches, hidden_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void patch_embed_forward(
    const Tensor& images,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output,
    int patch_size,
    cudaStream_t stream) {
    
    // For simplicity, allocate workspace here
    const int batch_size = images.shape()[0];
    const int channels = images.shape()[1];
    const int height = images.shape()[2];
    const int width = images.shape()[3];
    
    const int num_patches = (height / patch_size) * (width / patch_size);
    const int patch_dim = channels * patch_size * patch_size;
    
    Tensor workspace({batch_size * num_patches, patch_dim}, images.dtype());
    
    CublasLtHandle cublaslt_handle;
    
    patch_embed_unfold_gemm(images, weight, bias, output, workspace,
                           patch_size, cublaslt_handle.get(), stream);
}