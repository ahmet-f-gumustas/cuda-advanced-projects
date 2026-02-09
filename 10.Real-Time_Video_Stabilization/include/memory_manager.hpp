#pragma once

#include "common.hpp"
#include <unordered_map>
#include <string>

namespace cuda_stabilizer {

class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Allocate GPU memory
    template<typename T>
    T* allocate(size_t count, const std::string& name = "");

    // Allocate pinned (page-locked) host memory
    template<typename T>
    T* allocatePinned(size_t count, const std::string& name = "");

    // Free specific allocation
    void free(void* ptr);
    void freePinned(void* ptr);

    // Free all allocations
    void freeAll();

    // Memory transfer operations
    template<typename T>
    void copyToDevice(T* d_dst, const T* h_src, size_t count);

    template<typename T>
    void copyToHost(T* h_dst, const T* d_src, size_t count);

    template<typename T>
    void copyDeviceToDevice(T* d_dst, const T* d_src, size_t count);

    // Async memory operations
    template<typename T>
    void copyToDeviceAsync(T* d_dst, const T* h_src, size_t count, cudaStream_t stream);

    template<typename T>
    void copyToHostAsync(T* h_dst, const T* d_src, size_t count, cudaStream_t stream);

    // Set memory to value
    template<typename T>
    void memset(T* d_ptr, int value, size_t count);

    // Get memory statistics
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getPeakAllocated() const { return peak_allocated_; }

    // Get device memory info
    void getDeviceMemoryInfo(size_t& free_bytes, size_t& total_bytes) const;

    // Print memory statistics
    void printStats() const;

private:
    struct Allocation {
        void* ptr;
        size_t size;
        std::string name;
        bool is_pinned;
    };

    std::unordered_map<void*, Allocation> allocations_;
    size_t total_allocated_;
    size_t peak_allocated_;
};

// Template implementations
template<typename T>
T* MemoryManager::allocate(size_t count, const std::string& name) {
    size_t size = count * sizeof(T);
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));

    Allocation alloc = {ptr, size, name, false};
    allocations_[ptr] = alloc;
    total_allocated_ += size;
    peak_allocated_ = std::max(peak_allocated_, total_allocated_);

    return ptr;
}

template<typename T>
T* MemoryManager::allocatePinned(size_t count, const std::string& name) {
    size_t size = count * sizeof(T);
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));

    Allocation alloc = {ptr, size, name, true};
    allocations_[ptr] = alloc;
    total_allocated_ += size;
    peak_allocated_ = std::max(peak_allocated_, total_allocated_);

    return ptr;
}

template<typename T>
void MemoryManager::copyToDevice(T* d_dst, const T* h_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void MemoryManager::copyToHost(T* h_dst, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void MemoryManager::copyDeviceToDevice(T* d_dst, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

template<typename T>
void MemoryManager::copyToDeviceAsync(T* d_dst, const T* h_src, size_t count, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template<typename T>
void MemoryManager::copyToHostAsync(T* h_dst, const T* d_src, size_t count, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template<typename T>
void MemoryManager::memset(T* d_ptr, int value, size_t count) {
    CUDA_CHECK(cudaMemset(d_ptr, value, count * sizeof(T)));
}

} // namespace cuda_stabilizer
