#ifndef MAMBA_BLOCK_H
#define MAMBA_BLOCK_H

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

struct MambaConfig {
    int batch = 1;
    int seq_len = 256;
    int model_dim = 64;
    int inner_dim = 128;
    int state_size = 16;
    int dt_rank = 8;
    int conv_width = 4;
};

struct MambaWeights {
    std::vector<float> norm;
    std::vector<float> in_proj;
    std::vector<float> conv_weight;
    std::vector<float> conv_bias;
    std::vector<float> x_proj;
    std::vector<float> dt_proj;
    std::vector<float> dt_bias;
    std::vector<float> A;
    std::vector<float> D;
    std::vector<float> out_proj;
};

MambaWeights make_mamba_weights(const MambaConfig& config, unsigned seed = 42);

class MambaBlock {
public:
    explicit MambaBlock(const MambaConfig& config);
    ~MambaBlock();

    MambaBlock(const MambaBlock&) = delete;
    MambaBlock& operator=(const MambaBlock&) = delete;

    void load_weights(const MambaWeights& weights);
    void forward(const float* d_input, float* d_output, cudaStream_t stream = 0);
    const MambaConfig& config() const { return config_; }

private:
    void allocate();
    void release();

    MambaConfig config_;

    float* d_norm_ = nullptr;
    float* d_in_proj_ = nullptr;
    float* d_conv_weight_ = nullptr;
    float* d_conv_bias_ = nullptr;
    float* d_x_proj_ = nullptr;
    float* d_dt_proj_ = nullptr;
    float* d_dt_bias_ = nullptr;
    float* d_A_ = nullptr;
    float* d_D_ = nullptr;
    float* d_out_proj_ = nullptr;

    float* d_normalized_ = nullptr;
    float* d_projected_ = nullptr;
    float* d_x_ = nullptr;
    float* d_z_ = nullptr;
    float* d_convolved_ = nullptr;
    float* d_parameters_ = nullptr;
    float* d_delta_ = nullptr;
    float* d_B_ = nullptr;
    float* d_C_ = nullptr;
    float* d_scan_output_ = nullptr;
};

#endif
