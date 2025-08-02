#pragma once
#include "vit_model.hpp"
#include <string>
#include <fstream>

class WeightsLoader {
public:
    // Load from binary format
    static ViTWeights load_binary(const std::string& path, const ViTConfig& config);
    
    // Save to binary format
    static void save_binary(const std::string& path, const ViTWeights& weights, 
                          const ViTConfig& config);
    
    // Generate random weights for testing
    static ViTWeights generate_random(const ViTConfig& config, 
                                    Tensor::DataType dtype = Tensor::DataType::FP16);
    
private:
    static constexpr uint32_t MAGIC = 0x56495442; // "VITB"
    static constexpr uint32_t VERSION = 1;
    
    struct Header {
        uint32_t magic;
        uint32_t version;
        uint32_t precision; // 0=FP32, 1=FP16
        uint32_t hidden_dim;
        uint32_t num_heads;
        uint32_t num_layers;
    };
};