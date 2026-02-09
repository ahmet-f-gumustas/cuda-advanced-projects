#pragma once

#include "common.hpp"

namespace cuda_stabilizer {

// Gaussian pyramid structure
struct GaussianPyramid {
    float** d_levels;
    int* widths;
    int* heights;
    int num_levels;

    GaussianPyramid() : d_levels(nullptr), widths(nullptr), heights(nullptr), num_levels(0) {}
};

// Initialize Gaussian pyramid
void initGaussianPyramid(GaussianPyramid& pyramid, int width, int height, int num_levels);

// Release Gaussian pyramid resources
void releaseGaussianPyramid(GaussianPyramid& pyramid);

// Build Gaussian pyramid from image
void buildGaussianPyramid(
    const float* d_image,
    GaussianPyramid& pyramid,
    int width,
    int height
);

// Apply Gaussian blur
void gaussianBlur(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    float sigma = 1.0f
);

// Apply Gaussian blur with separable kernels (faster)
void gaussianBlurSeparable(
    const float* d_input,
    float* d_output,
    float* d_temp,
    int width,
    int height,
    float sigma = 1.0f
);

// Downsample image by factor of 2
void downsample2x(
    const float* d_input,
    float* d_output,
    int in_width,
    int in_height
);

// Upsample image by factor of 2
void upsample2x(
    const float* d_input,
    float* d_output,
    int in_width,
    int in_height
);

// Build Laplacian pyramid
void buildLaplacianPyramid(
    const float* d_image,
    GaussianPyramid& gaussian_pyramid,
    GaussianPyramid& laplacian_pyramid,
    int width,
    int height
);

// Reconstruct image from Laplacian pyramid
void reconstructFromLaplacian(
    const GaussianPyramid& laplacian_pyramid,
    float* d_output,
    int width,
    int height
);

} // namespace cuda_stabilizer
