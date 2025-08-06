#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <stdexcept>
#include <string>

// CUDA hata kontrolü makrosu
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << error                                    \
                      << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        }                                                                     \
    } while (0)

// cuDNN hata kontrolü makrosu
#define CUDNN_CHECK(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = call;                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << status                                   \
                      << " \"" << cudnnGetErrorString(status) << "\"" << std::endl; \
            throw std::runtime_error("cuDNN error: " + std::string(cudnnGetErrorString(status))); \
        }                                                                     \
    } while (0)

// CUDA stream sarmalayıcı
class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    
    cudaStream_t get() const { return stream_; }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    
private:
    cudaStream_t stream_;
};

// Pinned memory tahsisi için yardımcı
template<typename T>
class PinnedMemory {
public:
    explicit PinnedMemory(size_t count) : count_(count) {
        CUDA_CHECK(cudaMallocHost(&data_, count * sizeof(T)));
    }
    
    ~PinnedMemory() {
        cudaFreeHost(data_);
    }
    
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return count_; }
    
private:
    T* data_;
    size_t count_;
};

// Device memory tahsisi için yardımcı
template<typename T>
class DeviceMemory {
public:
    explicit DeviceMemory(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }
    
    ~DeviceMemory() {
        cudaFree(data_);
    }
    
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return count_; }
    
private:
    T* data_;
    size_t count_;
};