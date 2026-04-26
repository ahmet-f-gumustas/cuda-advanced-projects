#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "cuda_utils.h"

// ============================================================
// Configuration
// ============================================================

struct GaussianConfig {
    int num_gaussians = 0;
    int sh_degree = 3;          // 0, 1, 2, or 3

    int numSHCoeffs() const {
        return (sh_degree + 1) * (sh_degree + 1);
    }
};

// ============================================================
// GPU data (SoA layout for coalesced access)
// ============================================================

struct GaussianData {
    float* d_positions;     // (N x 3) world positions
    float* d_scales;        // (N x 3) log-scale
    float* d_rotations;     // (N x 4) quaternion (w, x, y, z)
    float* d_sh_coeffs;     // (N x C x 3) spherical harmonics, C = numSHCoeffs
    float* d_opacities;     // (N) sigmoid pre-activation
};

// ============================================================
// GaussianModel — manages Gaussian data lifecycle
// ============================================================

class GaussianModel {
public:
    GaussianModel();
    ~GaussianModel();

    // Generate random Gaussians in a bounding box for testing
    void generateRandom(int num_gaussians, int sh_degree,
                        float3 bbox_min, float3 bbox_max,
                        unsigned int seed = 42);

    // Upload host data to GPU
    void uploadToDevice(const std::vector<float>& positions,
                        const std::vector<float>& scales,
                        const std::vector<float>& rotations,
                        const std::vector<float>& sh_coeffs,
                        const std::vector<float>& opacities,
                        int sh_degree);

    // Load gaussians from a binary 3DGS-format .ply file
    bool loadFromPLY(const std::string& path);

    // Free GPU memory
    void free();

    // Accessors
    const GaussianConfig& getConfig() const { return config_; }
    const GaussianData& getData() const { return data_; }
    int getCount() const { return config_.num_gaussians; }
    int getSHDegree() const { return config_.sh_degree; }
    int getNumSHCoeffs() const { return config_.numSHCoeffs(); }
    bool isAllocated() const { return allocated_; }

private:
    void allocate(int num_gaussians, int sh_degree);

    GaussianConfig config_;
    GaussianData data_;
    bool allocated_;
};

#endif // GAUSSIAN_H
