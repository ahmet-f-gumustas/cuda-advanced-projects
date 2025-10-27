#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel launch error checking
#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device query and information
inline void printDeviceInfo() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA capable devices found!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "\n=== Device " << dev << ": " << prop.name << " ===" << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
        std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Grid Dimensions: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock Rate: " << (prop.clockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 Cache Size: " << (prop.l2CacheSize / 1024) << " KB" << std::endl;
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    }
    std::cout << std::endl;
}

// Timer class for benchmarking
class CudaTimer {
private:
    cudaEvent_t start_, stop_;
    bool running_;

public:
    CudaTimer() : running_(false) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
        running_ = true;
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        running_ = false;
    }

    float elapsed() {
        if (running_) {
            std::cerr << "Timer is still running!" << std::endl;
            return 0.0f;
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Memory allocation helpers
template<typename T>
inline T* cudaMallocManaged(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(::cudaMallocManaged(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline T* cudaMallocDevice(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(::cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline T* cudaMallocHost(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(::cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline void cudaFreeWrapper(T* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(::cudaFree(ptr));
    }
}

// Memory copy helpers
template<typename T>
inline void cudaMemcpyH2D(T* dst, const T* src, size_t count) {
    CUDA_CHECK(::cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
inline void cudaMemcpyD2H(T* dst, const T* src, size_t count) {
    CUDA_CHECK(::cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void cudaMemcpyD2D(T* dst, const T* src, size_t count) {
    CUDA_CHECK(::cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

#endif // CUDA_UTILS_H
