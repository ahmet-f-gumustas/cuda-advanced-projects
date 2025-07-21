#pragma once

#include <memory>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include "types.h"

namespace deep_engine {

class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;
    
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual DeviceType device_type() const = 0;
};

class CudaAllocator : public MemoryAllocator {
public:
    void* allocate(size_t bytes) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }
    
    void deallocate(void* ptr) override {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    
    DeviceType device_type() const override {
        return DeviceType::CUDA;
    }
};

class CudaPoolAllocator : public MemoryAllocator {
public:
    explicit CudaPoolAllocator(size_t initial_pool_size = 1024 * 1024 * 1024); // 1GB
    ~CudaPoolAllocator();
    
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    DeviceType device_type() const override { return DeviceType::CUDA; }
    
    // Pool management
    void reset();
    size_t used_memory() const { return used_memory_; }
    size_t total_memory() const { return total_memory_; }
    
private:
    struct Block {
        void* ptr;
        size_t size;
        bool free;
        Block* next;
        Block* prev;
    };
    
    void* pool_start_;
    size_t total_memory_;
    size_t used_memory_;
    Block* free_list_;
    Block* used_list_;
    std::mutex mutex_;
    
    void* find_free_block(size_t size);
    void coalesce_free_blocks();
};

class HostAllocator : public MemoryAllocator {
public:
    void* allocate(size_t bytes) override {
        return std::malloc(bytes);
    }
    
    void deallocate(void* ptr) override {
        std::free(ptr);
    }
    
    DeviceType device_type() const override {
        return DeviceType::CPU;
    }
};

class PinnedMemoryAllocator : public MemoryAllocator {
public:
    void* allocate(size_t bytes) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Pinned memory allocation failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }
    
    void deallocate(void* ptr) override {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
    
    DeviceType device_type() const override {
        return DeviceType::CPU;
    }
};

// Allocator factory
class AllocatorFactory {
public:
    static std::shared_ptr<MemoryAllocator> create(DeviceType device, 
                                                   bool use_pool = true) {
        switch (device) {
            case DeviceType::CUDA:
                if (use_pool) {
                    return std::make_shared<CudaPoolAllocator>();
                } else {
                    return std::make_shared<CudaAllocator>();
                }
            case DeviceType::CPU:
                return std::make_shared<HostAllocator>();
            default:
                throw std::runtime_error("Unsupported device type");
        }
    }
};

} // namespace deep_engine