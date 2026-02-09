#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Timer utility
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// Motion vector structure
struct MotionVector {
    float dx;
    float dy;
};

// Transform parameters (affine)
struct TransformParams {
    float dx;       // Translation X
    float dy;       // Translation Y
    float da;       // Rotation angle
    float ds;       // Scale

    TransformParams() : dx(0), dy(0), da(0), ds(1.0f) {}
    TransformParams(float x, float y, float a, float s) : dx(x), dy(y), da(a), ds(s) {}
};

// Trajectory point
struct Trajectory {
    float x;
    float y;
    float a;  // angle
    float s;  // scale

    Trajectory() : x(0), y(0), a(0), s(1.0f) {}
    Trajectory(float _x, float _y, float _a, float _s) : x(_x), y(_y), a(_a), s(_s) {}

    Trajectory operator+(const Trajectory& other) const {
        return Trajectory(x + other.x, y + other.y, a + other.a, s * other.s);
    }

    Trajectory operator-(const Trajectory& other) const {
        return Trajectory(x - other.x, y - other.y, a - other.a, s / other.s);
    }
};

namespace cuda_stabilizer {

// Configuration
struct StabilizerConfig {
    int smoothing_radius = 30;       // Frames for smoothing window
    float crop_ratio = 0.9f;         // Crop ratio to hide borders
    int pyramid_levels = 4;          // Gaussian pyramid levels
    int block_size = 16;             // Block size for motion estimation
    int search_range = 32;           // Search range for block matching
    float motion_threshold = 0.5f;   // Minimum motion threshold
    bool use_gpu = true;             // Use GPU acceleration
    bool show_comparison = false;    // Show original vs stabilized
};

} // namespace cuda_stabilizer
