#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace rtid {

struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    cudaStream_t stream;
    
    MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false), stream(0) {}
};

class GpuMemoryPool {
public:
    static GpuMemoryPool& getInstance();
    
    // Memory allocation/deallocation
    void* allocate(size_t size, size_t alignment = 256);
    bool deallocate(void* ptr);
    
    // Pool management
    void reserve(size_t total_size);
    void clear();
    size_t getTotalAllocated() const;
    size_t getTotalAvailable() const;
    size_t getFragmentation() const;
    
    // Statistics
    void printMemoryStats() const;
    size_t getAllocationCount() const { return allocation_count_; }
    size_t getDeallocationCount() const { return deallocation_count_; }
    
    // Stream-aware allocation
    void* allocateAsync(size_t size, cudaStream_t stream, size_t alignment = 256);
    bool deallocateAsync(void* ptr, cudaStream_t stream);
    
private:
    GpuMemoryPool() = default;
    ~GpuMemoryPool();
    
    // Prevent copying
    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;
    
    // Internal functions
    MemoryBlock* findFreeBlock(size_t size);
    void splitBlock(MemoryBlock* block, size_t size);
    void mergeAdjacentBlocks();
    size_t alignSize(size_t size, size_t alignment) const;
    
    // Memory tracking
    std::vector<std::unique_ptr<MemoryBlock>> blocks_;
    std::unordered_map<void*, MemoryBlock*> ptr_to_block_;
    
    // Statistics
    size_t total_allocated_;
    size_t total_reserved_;
    size_t allocation_count_;
    size_t deallocation_count_;
    
    // Thread safety
    mutable std::mutex mutex_;
};

class ScopedGpuMemory {
public:
    ScopedGpuMemory(size_t size, size_t alignment = 256);
    ScopedGpuMemory(size_t size, cudaStream_t stream, size_t alignment = 256);
    ~ScopedGpuMemory();
    
    // Prevent copying, allow moving
    ScopedGpuMemory(const ScopedGpuMemory&) = delete;
    ScopedGpuMemory& operator=(const ScopedGpuMemory&) = delete;
    ScopedGpuMemory(ScopedGpuMemory&& other) noexcept;
    ScopedGpuMemory& operator=(ScopedGpuMemory&& other) noexcept;
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool isValid() const { return ptr_ != nullptr; }
    
    template<typename T>
    T* getAs() const { return static_cast<T*>(ptr_); }
    
private:
    void* ptr_;
    size_t size_;
    cudaStream_t stream_;
    bool use_stream_;
};

// Utility functions for memory operations
class MemoryUtils {
public:
    // Copy operations
    static bool copyHostToDevice(void* d_dst, const void* h_src, size_t size, cudaStream_t stream = 0);
    static bool copyDeviceToHost(void* h_dst, const void* d_src, size_t size, cudaStream_t stream = 0);
    static bool copyDeviceToDevice(void* d_dst, const void* d_src, size_t size, cudaStream_t stream = 0);
    
    // Memory set operations
    static bool setMemory(void* ptr, int value, size_t size, cudaStream_t stream = 0);
    static bool setMemoryAsync(void* ptr, int value, size_t size, cudaStream_t stream);
    
    // Memory prefetching (for unified memory)
    static bool prefetchToDevice(void* ptr, size_t size, int device, cudaStream_t stream = 0);
    static bool prefetchToHost(void* ptr, size_t size, cudaStream_t stream = 0);
    
    // Memory advice (for unified memory)
    static bool setMemoryAdvice(void* ptr, size_t size, cudaMemoryAdvise advice, int device);
    
    // Utility functions
    static size_t getAvailableMemory();
    static size_t getTotalMemory();
    static bool isPointerOnDevice(const void* ptr);
    static int getPointerDevice(const void* ptr);
};

// RAII wrapper for CUDA events
class CudaEvent {
public:
    CudaEvent(unsigned int flags = cudaEventDefault);
    ~CudaEvent();
    
    // Prevent copying, allow moving
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;
    
    cudaEvent_t get() const { return event_; }
    bool record(cudaStream_t stream = 0);
    bool synchronize();
    bool query() const;
    float elapsedTime(const CudaEvent& start) const;
    
private:
    cudaEvent_t event_;
};

// RAII wrapper for CUDA streams
class CudaStream {
public:
    CudaStream(unsigned int flags = cudaStreamDefault);
    ~CudaStream();
    
    // Prevent copying, allow moving
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    
    cudaStream_t get() const { return stream_; }
    bool synchronize();
    bool query() const;
    
private:
    cudaStream_t stream_;
};

} // namespace rtid