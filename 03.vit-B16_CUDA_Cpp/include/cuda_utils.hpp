#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + std::to_string(status)); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuDNN error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudnnGetErrorString(status)); \
        } \
    } while(0)

// Thread block configurations
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

// Memory alignment
constexpr size_t ALIGNMENT = 128; // 128-bit alignment for tensor cores

inline void* aligned_alloc_cuda(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

inline void aligned_free_cuda(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// RAII wrappers
class CudaStream {
private:
    cudaStream_t stream_;
    
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    operator cudaStream_t() const { return stream_; }
    cudaStream_t get() const { return stream_; }
    
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
};

class CudaEvent {
private:
    cudaEvent_t event_;
    
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    
    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    
    float elapsed_time(const CudaEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }
    
    cudaEvent_t get() const { return event_; }
};