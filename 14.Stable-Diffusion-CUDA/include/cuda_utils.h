#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>

// ============================================================
// Error checking macros
// ============================================================

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                  \
    do {                                                                         \
        cudaError_t err = cudaGetLastError();                                    \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                           \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,   \
                    (int)status);                                                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDNN_CHECK(call)                                                        \
    do {                                                                         \
        cudnnStatus_t status = (call);                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudnnGetErrorString(status));                                 \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CURAND_CHECK(call)                                                       \
    do {                                                                         \
        curandStatus_t status = (call);                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__,   \
                    (int)status);                                                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ============================================================
// Device memory helpers
// ============================================================

template <typename T>
T* cudaMallocDevice(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
T* cudaMallocHostPinned(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
void cudaMemcpyH2D(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cudaMemcpyD2H(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void cudaMemcpyD2D(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void cudaZeroDevice(T* ptr, size_t count) {
    CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
}

// ============================================================
// CUDA Timer (event-based)
// ============================================================

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

// ============================================================
// Device info
// ============================================================

inline void printDeviceInfo() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("=== GPU: %s ===\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Global memory: %.1f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Shared memory/block: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("\n");
}

// ============================================================
// FP16 conversion helpers
// ============================================================

inline half float_to_half(float f) { return __float2half(f); }
inline float half_to_float(half h) { return __half2float(h); }

inline void float_to_half_array(const float* src, half* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

inline void half_to_float_array(const half* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __half2float(src[i]);
}

// ============================================================
// Random initialization helpers
// ============================================================

inline void init_random_fp16(half* d_ptr, size_t count, float scale, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> h_buf(count);
    for (size_t i = 0; i < count; i++) h_buf[i] = dist(rng);
    std::vector<half> h_half(count);
    float_to_half_array(h_buf.data(), h_half.data(), count);
    cudaMemcpyH2D(d_ptr, h_half.data(), count);
}

inline void init_zeros_fp16(half* d_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, count * sizeof(half)));
}

#endif // CUDA_UTILS_H
