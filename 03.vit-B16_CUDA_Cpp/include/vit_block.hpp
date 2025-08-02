#pragma once
#include "tensor.hpp"
#include "cublaslt_utils.hpp"
#include <memory>
#include <cuda_fp16.h>

struct ViTBlockWeights {
    Tensor ln1_gamma, ln1_beta;           // LayerNorm 1
    Tensor qkv_weight, qkv_bias;          // QKV projection
    Tensor proj_weight, proj_bias;        // Output projection
    Tensor ln2_gamma, ln2_beta;           // LayerNorm 2
    Tensor mlp_fc1_weight, mlp_fc1_bias;  // MLP layer 1
    Tensor mlp_fc2_weight, mlp_fc2_bias;  // MLP layer 2
};

class ViTBlock {
private:
    int hidden_dim_;
    int num_heads_;
    int mlp_dim_;
    
    // Workspace tensors
    mutable Tensor ln1_output_;
    mutable Tensor qkv_;
    mutable Tensor q_, k_, v_;
    mutable Tensor attn_scores_;
    mutable Tensor attn_probs_;
    mutable Tensor attn_output_;
    mutable Tensor proj_output_;
    mutable Tensor ln2_output_;
    mutable Tensor mlp_hidden_;
    
public:
    ViTBlock(int hidden_dim, int num_heads, int mlp_dim = 0);
    
    void forward(
        const Tensor& input,              // [B, S, D]
        const ViTBlockWeights& weights,
        Tensor& output,                   // [B, S, D]
        CublasLtHandle& cublaslt_handle,
        cudaStream_t stream) const;
    
    void allocate_workspace(int batch_size, int seq_len);
};