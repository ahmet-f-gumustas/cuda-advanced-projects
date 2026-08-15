#ifndef MAMBA_CUDA_UTILS_H
#define MAMBA_CUDA_UTILS_H

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)           \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

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

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

template <typename T>
T* device_alloc(size_t count) {
    T* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

template <typename T>
void copy_to_device(T* destination, const T* source, size_t count) {
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void copy_to_host(T* destination, const T* source, size_t count) {
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}

inline void print_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::cout << "GPU: " << properties.name << " (SM " << properties.major << '.'
              << properties.minor << ")\n";
}

#endif
