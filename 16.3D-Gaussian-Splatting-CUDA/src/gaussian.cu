#include "gaussian.h"
#include "ply_reader.h"

// ============================================================
// GaussianModel implementation
// ============================================================

GaussianModel::GaussianModel() : allocated_(false) {
    memset(&data_, 0, sizeof(data_));
}

GaussianModel::~GaussianModel() {
    free();
}

void GaussianModel::allocate(int num_gaussians, int sh_degree) {
    free();

    config_.num_gaussians = num_gaussians;
    config_.sh_degree = sh_degree;
    int C = config_.numSHCoeffs();

    data_.d_positions  = cudaMallocDevice<float>(num_gaussians * 3);
    data_.d_scales     = cudaMallocDevice<float>(num_gaussians * 3);
    data_.d_rotations  = cudaMallocDevice<float>(num_gaussians * 4);
    data_.d_sh_coeffs  = cudaMallocDevice<float>(num_gaussians * C * 3);
    data_.d_opacities  = cudaMallocDevice<float>(num_gaussians);

    allocated_ = true;
}

void GaussianModel::free() {
    if (!allocated_) return;

    if (data_.d_positions)  { CUDA_CHECK(cudaFree(data_.d_positions));  data_.d_positions = nullptr; }
    if (data_.d_scales)     { CUDA_CHECK(cudaFree(data_.d_scales));     data_.d_scales = nullptr; }
    if (data_.d_rotations)  { CUDA_CHECK(cudaFree(data_.d_rotations));  data_.d_rotations = nullptr; }
    if (data_.d_sh_coeffs)  { CUDA_CHECK(cudaFree(data_.d_sh_coeffs));  data_.d_sh_coeffs = nullptr; }
    if (data_.d_opacities)  { CUDA_CHECK(cudaFree(data_.d_opacities));  data_.d_opacities = nullptr; }

    allocated_ = false;
}

void GaussianModel::generateRandom(int num_gaussians, int sh_degree,
                                    float3 bbox_min, float3 bbox_max,
                                    unsigned int seed) {
    allocate(num_gaussians, sh_degree);
    int C = config_.numSHCoeffs();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_x(bbox_min.x, bbox_max.x);
    std::uniform_real_distribution<float> pos_y(bbox_min.y, bbox_max.y);
    std::uniform_real_distribution<float> pos_z(bbox_min.z, bbox_max.z);
    std::uniform_real_distribution<float> scale_dist(-5.0f, -3.0f);  // log-scale: actual scales ~0.007-0.05
    std::normal_distribution<float> sh_dist(0.0f, 0.5f);
    std::uniform_real_distribution<float> opacity_dist(0.5f, 2.0f);  // pre-sigmoid: mostly opaque

    // Positions
    std::vector<float> h_positions(num_gaussians * 3);
    for (int i = 0; i < num_gaussians; i++) {
        h_positions[i * 3 + 0] = pos_x(rng);
        h_positions[i * 3 + 1] = pos_y(rng);
        h_positions[i * 3 + 2] = pos_z(rng);
    }

    // Scales (log-space)
    std::vector<float> h_scales(num_gaussians * 3);
    for (int i = 0; i < num_gaussians * 3; i++) {
        h_scales[i] = scale_dist(rng);
    }

    // Rotations (unit quaternions: w, x, y, z)
    std::vector<float> h_rotations(num_gaussians * 4);
    std::normal_distribution<float> rot_dist(0.0f, 1.0f);
    for (int i = 0; i < num_gaussians; i++) {
        float w = rot_dist(rng);
        float x = rot_dist(rng);
        float y = rot_dist(rng);
        float z = rot_dist(rng);
        float norm = sqrtf(w * w + x * x + y * y + z * z);
        h_rotations[i * 4 + 0] = w / norm;
        h_rotations[i * 4 + 1] = x / norm;
        h_rotations[i * 4 + 2] = y / norm;
        h_rotations[i * 4 + 3] = z / norm;
    }

    // SH coefficients (DC term gets brighter colors, rest is small)
    std::vector<float> h_sh_coeffs(num_gaussians * C * 3, 0.0f);
    std::uniform_real_distribution<float> color_dist(0.2f, 0.9f);
    for (int i = 0; i < num_gaussians; i++) {
        // DC term (degree 0, coeff 0) — base color
        h_sh_coeffs[i * C * 3 + 0] = color_dist(rng);  // R
        h_sh_coeffs[i * C * 3 + 1] = color_dist(rng);  // G
        h_sh_coeffs[i * C * 3 + 2] = color_dist(rng);  // B
        // Higher-order SH coefficients — small perturbations
        for (int j = 1; j < C; j++) {
            h_sh_coeffs[i * C * 3 + j * 3 + 0] = sh_dist(rng) * 0.1f;
            h_sh_coeffs[i * C * 3 + j * 3 + 1] = sh_dist(rng) * 0.1f;
            h_sh_coeffs[i * C * 3 + j * 3 + 2] = sh_dist(rng) * 0.1f;
        }
    }

    // Opacities (pre-sigmoid)
    std::vector<float> h_opacities(num_gaussians);
    for (int i = 0; i < num_gaussians; i++) {
        h_opacities[i] = opacity_dist(rng);
    }

    // Upload to GPU
    cudaMemcpyH2D(data_.d_positions, h_positions.data(), num_gaussians * 3);
    cudaMemcpyH2D(data_.d_scales, h_scales.data(), num_gaussians * 3);
    cudaMemcpyH2D(data_.d_rotations, h_rotations.data(), num_gaussians * 4);
    cudaMemcpyH2D(data_.d_sh_coeffs, h_sh_coeffs.data(), num_gaussians * C * 3);
    cudaMemcpyH2D(data_.d_opacities, h_opacities.data(), num_gaussians);

    printf("Generated %d random Gaussians (SH degree %d, %d coeffs)\n",
           num_gaussians, sh_degree, C);
    printf("  Bounding box: (%.1f,%.1f,%.1f) to (%.1f,%.1f,%.1f)\n",
           bbox_min.x, bbox_min.y, bbox_min.z,
           bbox_max.x, bbox_max.y, bbox_max.z);

    size_t total_bytes = num_gaussians * (3 + 3 + 4 + C * 3 + 1) * sizeof(float);
    printf("  GPU memory: %.2f MB\n\n", total_bytes / (1024.0 * 1024.0));
}

void GaussianModel::uploadToDevice(const std::vector<float>& positions,
                                    const std::vector<float>& scales,
                                    const std::vector<float>& rotations,
                                    const std::vector<float>& sh_coeffs,
                                    const std::vector<float>& opacities,
                                    int sh_degree) {
    int num_gaussians = (int)opacities.size();
    allocate(num_gaussians, sh_degree);

    cudaMemcpyH2D(data_.d_positions, positions.data(), num_gaussians * 3);
    cudaMemcpyH2D(data_.d_scales, scales.data(), num_gaussians * 3);
    cudaMemcpyH2D(data_.d_rotations, rotations.data(), num_gaussians * 4);
    cudaMemcpyH2D(data_.d_sh_coeffs, sh_coeffs.data(), (size_t)num_gaussians * config_.numSHCoeffs() * 3);
    cudaMemcpyH2D(data_.d_opacities, opacities.data(), num_gaussians);

    printf("Uploaded %d Gaussians to GPU (SH degree %d)\n\n", num_gaussians, sh_degree);
}

bool GaussianModel::loadFromPLY(const std::string& path) {
    PLYData data;
    if (!PLYReader::readFile(path, data)) return false;
    uploadToDevice(data.positions, data.scales, data.rotations,
                   data.sh_coeffs, data.opacities, data.sh_degree);
    return true;
}
