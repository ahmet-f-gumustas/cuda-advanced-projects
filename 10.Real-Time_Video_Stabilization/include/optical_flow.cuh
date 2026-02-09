#pragma once

#include "common.hpp"

namespace cuda_stabilizer {

// Lucas-Kanade optical flow parameters
struct OpticalFlowParams {
    int window_size = 21;
    int max_iterations = 30;
    float epsilon = 0.01f;
    int pyramid_levels = 4;
};

// Initialize optical flow
void initOpticalFlow(int width, int height, int pyramid_levels);

// Release optical flow resources
void releaseOpticalFlow();

// Compute optical flow between two frames using Lucas-Kanade
void computeOpticalFlowLK(
    const float* d_prev_frame,
    const float* d_curr_frame,
    float* d_flow_x,
    float* d_flow_y,
    int width,
    int height,
    const OpticalFlowParams& params
);

// Compute dense optical flow using Horn-Schunck method
void computeOpticalFlowHS(
    const float* d_prev_frame,
    const float* d_curr_frame,
    float* d_flow_x,
    float* d_flow_y,
    int width,
    int height,
    float alpha,
    int iterations
);

// Compute image gradients (Sobel)
void computeGradients(
    const float* d_image,
    float* d_grad_x,
    float* d_grad_y,
    int width,
    int height
);

// Compute temporal gradient
void computeTemporalGradient(
    const float* d_prev_frame,
    const float* d_curr_frame,
    float* d_grad_t,
    int width,
    int height
);

} // namespace cuda_stabilizer
