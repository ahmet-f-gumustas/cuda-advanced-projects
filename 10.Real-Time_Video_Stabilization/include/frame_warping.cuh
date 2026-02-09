#pragma once

#include "common.hpp"

namespace cuda_stabilizer {

// Initialize frame warping
void initFrameWarping(int width, int height);

// Release frame warping resources
void releaseFrameWarping();

// Apply affine transformation to frame
void warpFrameAffine(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    const TransformParams& transform
);

// Apply affine transformation with crop
void warpFrameAffineCrop(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    const TransformParams& transform,
    float crop_ratio
);

// Apply perspective transformation
void warpFramePerspective(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    const float* d_homography
);

// Convert BGR to grayscale
void convertToGrayscale(
    const unsigned char* d_input,
    float* d_output,
    int width,
    int height,
    int channels
);

// Convert grayscale to float
void convertToFloat(
    const unsigned char* d_input,
    float* d_output,
    int width,
    int height
);

// Bilinear interpolation kernel
__device__ inline float bilinearInterpolate(
    const float* image,
    float x,
    float y,
    int width,
    int height
);

// Border handling modes
enum class BorderMode {
    CONSTANT,
    REPLICATE,
    REFLECT
};

// Apply border handling
void applyBorder(
    unsigned char* d_image,
    int width,
    int height,
    int channels,
    int border_size,
    BorderMode mode
);

} // namespace cuda_stabilizer
