#include "../include/vit_model.hpp"
#include "../include/weights.hpp"
#include <iostream>
#include <cuda_runtime.h>

int main() {
    try {
        // Initialize CUDA
        cudaSetDevice(0);
        
        std::cout << "Generating dummy ViT-B/16 weights..." << std::endl;
        
        // Create config
        ViTConfig config;
        config.image_size = 224;
        config.patch_size = 16;
        config.num_channels = 3;
        config.hidden_dim = 768;
        config.num_heads = 12;
        config.num_layers = 12;
        config.mlp_dim = 3072;
        config.num_classes = 1000;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Image size: " << config.image_size << "x" << config.image_size << std::endl;
        std::cout << "  Patch size: " << config.patch_size << std::endl;
        std::cout << "  Hidden dimension: " << config.hidden_dim << std::endl;
        std::cout << "  Number of heads: " << config.num_heads << std::endl;
        std::cout << "  Number of layers: " << config.num_layers << std::endl;
        std::cout << "  MLP dimension: " << config.mlp_dim << std::endl;
        std::cout << "  Number of classes: " << config.num_classes << std::endl;
        std::cout << "  Sequence length: " << config.sequence_length() << std::endl;
        
        // Generate random weights
        std::cout << "\nGenerating weights..." << std::endl;
        auto weights = WeightsLoader::generate_random(config, Tensor::DataType::FP16);
        
        // Save to file
        std::string output_path = "weights.bin";
        std::cout << "Saving weights to " << output_path << "..." << std::endl;
        WeightsLoader::save_binary(output_path, weights, config);
        
        // Calculate total size
        size_t total_params = 0;
        total_params += config.hidden_dim * 3 * config.patch_size * config.patch_size; // patch_embed
        total_params += config.hidden_dim; // patch_embed bias
        total_params += config.hidden_dim; // cls_token
        total_params += config.sequence_length() * config.hidden_dim; // pos_embed
        
        // Encoder blocks
        for (int i = 0; i < config.num_layers; ++i) {
            total_params += 2 * config.hidden_dim; // ln1
            total_params += 3 * config.hidden_dim * config.hidden_dim + 3 * config.hidden_dim; // qkv
            total_params += config.hidden_dim * config.hidden_dim + config.hidden_dim; // proj
            total_params += 2 * config.hidden_dim; // ln2
            total_params += config.hidden_dim * config.mlp_dim + config.mlp_dim; // mlp1
            total_params += config.mlp_dim * config.hidden_dim + config.hidden_dim; // mlp2
        }
        
        // Head
        total_params += 2 * config.hidden_dim; // ln
        total_params += config.hidden_dim * config.num_classes + config.num_classes; // head
        
        std::cout << "\nTotal parameters: " << total_params << std::endl;
        std::cout << "File size: ~" << (total_params * 2) / (1024 * 1024) << " MB (FP16)" << std::endl;
        
        std::cout << "\nDone! Weights saved to " << output_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}