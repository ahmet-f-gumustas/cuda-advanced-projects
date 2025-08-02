#include "weights.hpp"
#include <fstream>
#include <random>
#include <cuda_fp16.h>

ViTWeights WeightsLoader::load_binary(const std::string& path, const ViTConfig& config) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + path);
    }
    
    // Read header
    Header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(Header));
    
    if (header.magic != MAGIC) {
        throw std::runtime_error("Invalid weights file format");
    }
    
    if (header.version != VERSION) {
        throw std::runtime_error("Unsupported weights version");
    }
    
    auto dtype = header.precision == 0 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    
    ViTWeights weights;
    
    // Helper to read tensor
    auto read_tensor = [&](const std::vector<int>& shape) -> Tensor {
        Tensor tensor(shape, dtype);
        size_t nbytes = tensor.nbytes();
        std::vector<char> buffer(nbytes);
        file.read(buffer.data(), nbytes);
        tensor.copy_from_host(buffer.data());
        return tensor;
    };
    
    // Read patch embedding
    weights.patch_embed_weight = read_tensor({config.hidden_dim, 3, config.patch_size, config.patch_size});
    weights.patch_embed_bias = read_tensor({config.hidden_dim});
    
    // Read embeddings
    weights.cls_token = read_tensor({1, 1, config.hidden_dim});
    weights.pos_embed = read_tensor({1, config.sequence_length(), config.hidden_dim});
    
    // Read encoder blocks
    weights.blocks.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        auto& block = weights.blocks[i];
        
        // LayerNorm 1
        block.ln1_gamma = read_tensor({config.hidden_dim});
        block.ln1_beta = read_tensor({config.hidden_dim});
        
        // Attention
        block.qkv_weight = read_tensor({3 * config.hidden_dim, config.hidden_dim});
        block.qkv_bias = read_tensor({3 * config.hidden_dim});
        block.proj_weight = read_tensor({config.hidden_dim, config.hidden_dim});
        block.proj_bias = read_tensor({config.hidden_dim});
        
        // LayerNorm 2
        block.ln2_gamma = read_tensor({config.hidden_dim});
        block.ln2_beta = read_tensor({config.hidden_dim});
        
        // MLP
        block.mlp_fc1_weight = read_tensor({config.mlp_dim, config.hidden_dim});
        block.mlp_fc1_bias = read_tensor({config.mlp_dim});
        block.mlp_fc2_weight = read_tensor({config.hidden_dim, config.mlp_dim});
        block.mlp_fc2_bias = read_tensor({config.hidden_dim});
    }
    
    // Read head
    weights.ln_gamma = read_tensor({config.hidden_dim});
    weights.ln_beta = read_tensor({config.hidden_dim});
    weights.head_weight = read_tensor({config.num_classes, config.hidden_dim});
    weights.head_bias = read_tensor({config.num_classes});
    
    return weights;
}

void WeightsLoader::save_binary(const std::string& path, const ViTWeights& weights, 
                               const ViTConfig& config) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create weights file: " + path);
    }
    
    // Write header
    Header header;
    header.magic = MAGIC;
    header.version = VERSION;
    header.precision = (weights.patch_embed_weight.dtype() == Tensor::DataType::FP16) ? 1 : 0;
    header.hidden_dim = config.hidden_dim;
    header.num_heads = config.num_heads;
    header.num_layers = config.num_layers;
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(Header));
    
    // Helper to write tensor
    auto write_tensor = [&](const Tensor& tensor) {
        size_t nbytes = tensor.nbytes();
        std::vector<char> buffer(nbytes);
        tensor.copy_to_host(buffer.data());
        file.write(buffer.data(), nbytes);
    };
    
    // Write all weights
    write_tensor(weights.patch_embed_weight);
    write_tensor(weights.patch_embed_bias);
    write_tensor(weights.cls_token);
    write_tensor(weights.pos_embed);
    
    for (const auto& block : weights.blocks) {
        write_tensor(block.ln1_gamma);
        write_tensor(block.ln1_beta);
        write_tensor(block.qkv_weight);
        write_tensor(block.qkv_bias);
        write_tensor(block.proj_weight);
        write_tensor(block.proj_bias);
        write_tensor(block.ln2_gamma);
        write_tensor(block.ln2_beta);
        write_tensor(block.mlp_fc1_weight);
        write_tensor(block.mlp_fc1_bias);
        write_tensor(block.mlp_fc2_weight);
        write_tensor(block.mlp_fc2_bias);
    }
    
    write_tensor(weights.ln_gamma);
    write_tensor(weights.ln_beta);
    write_tensor(weights.head_weight);
    write_tensor(weights.head_bias);
}

ViTWeights WeightsLoader::generate_random(const ViTConfig& config, Tensor::DataType dtype) {
    ViTWeights weights;
    
    // Initialize random generator
    std::mt19937 gen(42);
    std::normal_distribution<float> norm_dist(0.0f, 0.02f);
    
    // Helper to create random tensor
    auto random_tensor = [&](const std::vector<int>& shape) -> Tensor {
        Tensor tensor = Tensor::randn(shape, dtype);
        
        // Scale by 0.02 (typical initialization)
        tensor.fill(0.02f);
        
        return tensor;
    };
    
    // Initialize all weights
    weights.patch_embed_weight = random_tensor({config.hidden_dim, 3, config.patch_size, config.patch_size});
    weights.patch_embed_bias = Tensor::zeros({config.hidden_dim}, dtype);
    
    weights.cls_token = random_tensor({1, 1, config.hidden_dim});
    weights.pos_embed = random_tensor({1, config.sequence_length(), config.hidden_dim});
    
    weights.blocks.resize(config.num_layers);
    for (auto& block : weights.blocks) {
        block.ln1_gamma = Tensor::ones({config.hidden_dim}, dtype);
        block.ln1_beta = Tensor::zeros({config.hidden_dim}, dtype);
        
        block.qkv_weight = random_tensor({3 * config.hidden_dim, config.hidden_dim});
        block.qkv_bias = Tensor::zeros({3 * config.hidden_dim}, dtype);
        block.proj_weight = random_tensor({config.hidden_dim, config.hidden_dim});
        block.proj_bias = Tensor::zeros({config.hidden_dim}, dtype);
        
        block.ln2_gamma = Tensor::ones({config.hidden_dim}, dtype);
        block.ln2_beta = Tensor::zeros({config.hidden_dim}, dtype);
        
        block.mlp_fc1_weight = random_tensor({config.mlp_dim, config.hidden_dim});
        block.mlp_fc1_bias = Tensor::zeros({config.mlp_dim}, dtype);
        block.mlp_fc2_weight = random_tensor({config.hidden_dim, config.mlp_dim});
        block.mlp_fc2_bias = Tensor::zeros({config.hidden_dim}, dtype);
    }
    
    weights.ln_gamma = Tensor::ones({config.hidden_dim}, dtype);
    weights.ln_beta = Tensor::zeros({config.hidden_dim}, dtype);
    weights.head_weight = random_tensor({config.num_classes, config.hidden_dim});
    weights.head_bias = Tensor::zeros({config.num_classes}, dtype);
    
    return weights;
}