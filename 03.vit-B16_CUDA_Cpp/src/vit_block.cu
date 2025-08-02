#include "vit_block.hpp"
#include "ops_layernorm.hpp"
#include "ops_softmax.hpp"
#include "ops_gelu.hpp"
#include "ops_fuse.hpp"
#include <cmath>

ViTBlock::ViTBlock(int hidden_dim, int num_heads, int mlp_dim)
    : hidden_dim_(hidden_dim), num_heads_(num_heads), 
      mlp_dim_(mlp_dim > 0 ? mlp_dim : hidden_dim * 4) {}

void ViTBlock::allocate_workspace(int batch_size, int seq_len) {
    auto dtype = Tensor::DataType::FP16;
    
    ln1_output_ = Tensor({batch_size, seq_len, hidden_dim_}, dtype);
    qkv_ = Tensor({batch_size, seq_len, 3 * hidden_dim_}, dtype);
    q_ = Tensor({batch_size, num_heads_, seq_len, hidden_dim_ / num_heads_}, dtype);
    k_ = Tensor({batch_size, num_heads_, seq_len, hidden_dim_ / num_heads_}, dtype);
    v_ = Tensor({batch_size, num_heads_, seq_len, hidden_dim_ / num_heads_}, dtype);
    attn_scores_ = Tensor({batch_size, num_heads_, seq_len, seq_len}, dtype);
    attn_probs_ = Tensor({batch_size, num_heads_, seq_len, seq_len}, dtype);
    attn_output_ = Tensor({batch_size, num_heads_, seq_len, hidden_dim_ / num_heads_}, dtype);
    proj_output_ = Tensor({batch_size, seq_len, hidden_dim_}, dtype);
    ln2_output_ = Tensor({batch_size, seq_len, hidden_dim_}, dtype);
    mlp_hidden_ = Tensor({batch_size, seq_len, mlp_dim_}, dtype);
}

void ViTBlock::forward(
    const Tensor& input,
    const ViTBlockWeights& weights,
    Tensor& output,
    CublasLtHandle& cublaslt_handle,
    cudaStream_t stream) const {
    
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int head_dim = hidden_dim_ / num_heads_;
    
    // 1. LayerNorm
    layernorm_forward(input, weights.ln1_gamma, weights.ln1_beta, 
                     ln1_output_, 1e-5f, stream);
    
    // 2. QKV projection
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    gemm_fp16_tensor_core(
        cublaslt_handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        3 * hidden_dim_,           // m
        batch_size * seq_len,      // n
        hidden_dim_,               // k
        &alpha,
        weights.qkv_weight.data_ptr<__half>(), hidden_dim_,
        ln1_output_.data_ptr<__half>(), hidden_dim_,
        &beta,
        qkv_.data_ptr<__half>(), 3 * hidden_dim_,
        nullptr, 0, stream
    );
    
    // Add QKV bias
    fused_bias_residual(qkv_, weights.qkv_bias, 
                       Tensor::zeros(qkv_.shape(), qkv_.dtype()), 
                       qkv_, stream);
    
    // 3. Reshape QKV
    // This is a view operation, but for simplicity we'll copy
    // In production, use a custom kernel to avoid copies
    // For now, we'll process attention in a simplified way
    
    // 4. Scaled dot-product attention
    // Q @ K^T / sqrt(d_k)
    const float scale = 1.0f / std::sqrt(float(head_dim));
    const __half scale_half = __float2half(scale);
    
    // For each head in parallel
    for (int h = 0; h < num_heads_; ++h) {
        // Extract Q, K, V for this head
        // This would be done more efficiently with custom kernels
        
        // Compute attention scores
        gemm_fp16_tensor_core(
            cublaslt_handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_dim,
            &scale_half,
            k_.data_ptr<__half>() + h * seq_len * head_dim, head_dim,
            q_.data_ptr<__half>() + h * seq_len * head_dim, head_dim,
            &beta,
            attn_scores_.data_ptr<__half>() + h * seq_len * seq_len, seq_len,
            nullptr, 0, stream
        );
    }
    
    // 5. Softmax
    softmax_forward(attn_scores_, attn_probs_, stream);
    
    // 6. Attention @ V
    for (int h = 0; h < num_heads_; ++h) {
        gemm_fp16_tensor_core(
            cublaslt_handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len,
            &alpha,
            v_.data_ptr<__half>() + h * seq_len * head_dim, head_dim,
            attn_probs_.data_ptr<__half>() + h * seq_len * seq_len, seq_len,
            &beta,
            attn_output_.data_ptr<__half>() + h * seq_len * head_dim, head_dim,
            nullptr, 0, stream
        );
    }
    
    // 7. Concatenate heads and project
    // For simplicity, we'll treat this as a single GEMM
    gemm_fp16_tensor_core(
        cublaslt_handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_dim_, batch_size * seq_len, hidden_dim_,
        &alpha,
        weights.proj_weight.data_ptr<__half>(), hidden_dim_,
        attn_output_.data_ptr<__half>(), hidden_dim_,
        &beta,
        proj_output_.data_ptr<__half>(), hidden_dim_,
        nullptr, 0, stream
    );
    
    // 8. Add projection bias and residual connection
    fused_bias_residual(proj_output_, weights.proj_bias, input, proj_output_, stream);
    
    // 9. LayerNorm 2
    layernorm_forward(proj_output_, weights.ln2_gamma, weights.ln2_beta, 
                     ln2_output_, 1e-5f, stream);
    
    // 10. MLP: Linear -> GELU -> Linear
    gemm_fp16_tensor_core(
        cublaslt_handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        mlp_dim_, batch_size * seq_len, hidden_dim_,
        &alpha,
        weights.mlp_fc1_weight.data_ptr<__half>(), hidden_dim_,
        ln2_output_.data_ptr<__half>(), hidden_dim_,
        &beta,
        mlp_hidden_.data_ptr<__half>(), mlp_dim_,
        nullptr, 0, stream
    );
    
    // Add bias and apply GELU
    fused_bias_residual(mlp_hidden_, weights.mlp_fc1_bias,
                       Tensor::zeros(mlp_hidden_.shape(), mlp_hidden_.dtype()),
                       mlp_hidden_, stream);
    gelu_fast_forward(mlp_hidden_, mlp_hidden_, stream);
    
    // Second linear layer
    gemm_fp16_tensor_core(
        cublaslt_handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_dim_, batch_size * seq_len, mlp_dim_,
        &alpha,
        weights.mlp_fc2_weight.data_ptr<__half>(), mlp_dim_,
        mlp_hidden_.data_ptr<__half>(), mlp_dim_,
        &beta,
        output.data_ptr<__half>(), hidden_dim_,
        nullptr, 0, stream
    );
    
    // 11. Final residual connection
    fused_bias_residual(output, weights.mlp_fc2_bias, proj_output_, output, stream);
}