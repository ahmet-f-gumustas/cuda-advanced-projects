#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
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
// float3/float4 math helpers (host + device)
// ============================================================

inline __host__ __device__ float3 make_float3(float s) {
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline __host__ __device__ float length(float3 v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 normalize(float3 v) {
    float inv_len = 1.0f / (length(v) + 1e-8f);
    return v * inv_len;
}

#endif // CUDA_UTILS_H
