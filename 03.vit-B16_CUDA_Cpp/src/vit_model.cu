#include "vit_model.hpp"
#include "ops_patch_embed.hpp"
#include "ops_position_embed.hpp"
#include "ops_layernorm.hpp"
#include "ops_fuse.hpp"
#include "cublaslt_utils.hpp"
#include "timers.hpp"
#include <cuda_fp16.h>

__global__ void normalize_images_kernel(
    const uint8_t* __restrict__ input,   // [B, H, W, 3]
    __half* __restrict__ output,         // [B, 3, H, W]
    int batch_size,
    int height,
    int width) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = batch_size * height * width;
    
    if (idx < total_pixels) {
        const int b = idx / (height * width);
        const int hw = idx % (height * width);
        const int h = hw / width;
        const int w = hw % width;
        
        // ImageNet mean and std
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};
        
        // Read RGB values and normalize
        for (int c = 0; c < 3; ++c) {
            const int in_idx = ((b * height + h) * width + w) * 3 + c;
            const int out_idx = ((b * 3 + c) * height + h) * width + w;
            
            float val = float(input[in_idx]) / 255.0f;
            val = (val - mean[c]) / std[c];
            output[out_idx] = __float2half(val);
        }
    }
}

__global__ void extract_cls_token_kernel(
    const __half* __restrict__ input,   // [B, S, D]
    __half* __restrict__ output,        // [B, D]
    int batch_size,
    int seq_len,
    int hidden_dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_dim) {
        const int b = idx / hidden_dim;
        const int d = idx % hidden_dim;
        const int src_idx = b * seq_len * hidden_dim + d;
        output[idx] = input[src_idx];
    }
}

ViTModel::ViTModel(const ViTConfig& config) : config_(config) {
    // Initialize encoder blocks
    for (int i = 0; i < config.num_layers; ++i) {
        blocks_.emplace_back(std::make_unique<ViTBlock>(
            config.hidden_dim, config.num_heads, config.mlp_dim));
    }
}

void ViTModel::allocate_workspace(int batch_size) {
    auto dtype = Tensor::DataType::FP16;
    
    // Allocate tensors
    patches_ = Tensor({batch_size, config_.num_patches(), config_.hidden_dim}, dtype);
    embeddings_ = Tensor({batch_size, config_.sequence_length(), config_.hidden_dim}, dtype);
    
    // Allocate block outputs
    block_outputs_.resize(config_.num_layers + 1);
    for (auto& output : block_outputs_) {
        output = Tensor({batch_size, config_.sequence_length(), config_.hidden_dim}, dtype);
    }
    
    // Allocate workspace for blocks
    for (auto& block : blocks_) {
        block->allocate_workspace(batch_size, config_.sequence_length());
    }
}

void ViTModel::preprocess(const Tensor& images_uint8, Tensor& images_normalized) {
    const int batch_size = images_uint8.shape()[0];
    const int height = images_uint8.shape()[1];
    const int width = images_uint8.shape()[2];
    
    const int threads = 256;
    const int blocks = (batch_size * height * width + threads - 1) / threads;
    
    // Cast data pointer appropriately based on actual data type
    // Since we're using FP32 tensor to store uint8 data temporarily
    normalize_images_kernel<<<blocks, threads, 0, stream_.get()>>>(
        reinterpret_cast<const uint8_t*>(images_uint8.data()),
        images_normalized.data_ptr<__half>(),
        batch_size, height, width
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void ViTModel::forward(
    const Tensor& images,
    const ViTWeights& weights,
    Tensor& logits) {
    
    const int batch_size = images.shape()[0];
    
    // 1. Patch embedding
    patch_embed_forward(images, weights.patch_embed_weight, weights.patch_embed_bias,
                       patches_, config_.patch_size, stream_.get());
    
    // 2. Add class token
    add_class_token(patches_, weights.cls_token, embeddings_, stream_.get());
    
    // 3. Add positional embeddings
    add_positional_embedding(embeddings_, weights.pos_embed, stream_.get());
    
    // 4. Encoder blocks
    // Copy embeddings to first block output
    CUDA_CHECK(cudaMemcpyAsync(
        block_outputs_[0].data(),
        embeddings_.data(),
        embeddings_.nbytes(),
        cudaMemcpyDeviceToDevice,
        stream_.get()
    ));
    
    for (int i = 0; i < config_.num_layers; ++i) {
        blocks_[i]->forward(
            block_outputs_[i],
            weights.blocks[i],
            block_outputs_[i + 1],
            cublaslt_handle_,
            stream_.get()
        );
    }
    
    // 5. Final layer norm
    Tensor ln_output({batch_size, config_.sequence_length(), config_.hidden_dim}, 
                     Tensor::DataType::FP16);
    layernorm_forward(block_outputs_[config_.num_layers], weights.ln_gamma, 
                     weights.ln_beta, ln_output, 1e-5f, stream_.get());
    
    // 6. Extract class token (first token)
    Tensor cls_output({batch_size, config_.hidden_dim}, Tensor::DataType::FP16);
    
    const int extract_threads = 256;
    const int extract_blocks = (batch_size * config_.hidden_dim + extract_threads - 1) / extract_threads;
    
    extract_cls_token_kernel<<<extract_blocks, extract_threads, 0, stream_.get()>>>(
        ln_output.data_ptr<__half>(),
        cls_output.data_ptr<__half>(),
        batch_size,
        config_.sequence_length(),
        config_.hidden_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // 7. Classification head
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    gemm_fp16_tensor_core(
        cublaslt_handle_.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        config_.num_classes,      // m
        batch_size,               // n
        config_.hidden_dim,       // k
        &alpha,
        weights.head_weight.data_ptr<__half>(), config_.hidden_dim,
        cls_output.data_ptr<__half>(), config_.hidden_dim,
        &beta,
        logits.data_ptr<__half>(), config_.num_classes,
        nullptr, 0, stream_.get()
    );
    
    // Add head bias
    fused_bias_residual(logits, weights.head_bias,
                       Tensor::zeros(logits.shape(), logits.dtype()),
                       logits, stream_.get());
    
    CUDA_CHECK(cudaGetLastError());
}