#pragma once

#include "common.hpp"

namespace cuda_stabilizer {

// Motion estimation parameters
struct MotionEstimationParams {
    int block_size = 16;
    int search_range = 32;
    float threshold = 0.5f;
};

// Initialize motion estimation
void initMotionEstimation(int width, int height);

// Release motion estimation resources
void releaseMotionEstimation();

// Compute block-based motion vectors
void computeBlockMotion(
    const float* d_prev_frame,
    const float* d_curr_frame,
    MotionVector* d_motion_vectors,
    int width,
    int height,
    const MotionEstimationParams& params
);

// Compute global motion parameters from motion vectors
TransformParams computeGlobalMotion(
    const MotionVector* d_motion_vectors,
    int num_blocks_x,
    int num_blocks_y,
    int block_size,
    int width,
    int height
);

// RANSAC-based outlier rejection for motion estimation
void ransacMotionEstimation(
    const float* d_prev_points,
    const float* d_curr_points,
    int num_points,
    TransformParams& transform,
    int max_iterations = 100,
    float threshold = 3.0f
);

// Feature point detection (Harris corners)
void detectFeaturePoints(
    const float* d_image,
    float* d_corners,
    int* d_num_corners,
    int width,
    int height,
    float threshold = 0.01f,
    int max_corners = 1000
);

// Track feature points between frames
void trackFeaturePoints(
    const float* d_prev_frame,
    const float* d_curr_frame,
    const float* d_prev_points,
    float* d_curr_points,
    unsigned char* d_status,
    int num_points,
    int width,
    int height,
    int window_size = 21
);

} // namespace cuda_stabilizer
