#pragma once
#include "tensor.hpp"
#include "vit_block.hpp"
#include "cublaslt_utils.hpp"
#include <vector>
#include <memory>

struct ViTConfig {
    int image_size = 224;
    int patch_size = 16;
    int num_channels = 3;
    int hidden_dim = 768;
    int num_heads = 12;
    int num_layers = 12;
    int mlp_dim = 3072;
    int num_classes = 1000;
    float dropout = 0.0f;
    
    int num_patches() const {
        return (image_size / patch_size) * (image_size / patch_size);
    }
    
    int sequence_length() const {
        return num_patches() + 1; // +1 for class token
    }
};

struct ViTWeights {
    // Patch embedding
    Tensor patch_embed_weight;
    Tensor patch_embed_bias;
    
    // Positional and class embeddings
    Tensor cls_token;
    Tensor pos_embed;
    
    // Encoder blocks
    std::vector<ViTBlockWeights> blocks;
    
    // Head
    Tensor ln_gamma, ln_beta;
    Tensor head_weight, head_bias;
};

class ViTModel {
private:
    ViTConfig config_;
    std::vector<std::unique_ptr<ViTBlock>> blocks_;
    
    // Workspace tensors
    mutable Tensor patches_;
    mutable Tensor embeddings_;
    mutable std::vector<Tensor> block_outputs_;
    
    // CUDA resources
    CudaStream stream_;
    CublasLtHandle cublaslt_handle_;
    
public:
    explicit ViTModel(const ViTConfig& config);
    
    // Main inference function
    void forward(
        const Tensor& images,      // [B, 3, H, W]
        const ViTWeights& weights,
        Tensor& logits);          // [B, num_classes]
    
    // Preprocessing on GPU
    void preprocess(
        const Tensor& images_uint8,  // [B, H, W, 3] in uint8
        Tensor& images_normalized); // [B, 3, H, W] in FP16
    
    void allocate_workspace(int batch_size);
    
    const ViTConfig& config() const { return config_; }
};