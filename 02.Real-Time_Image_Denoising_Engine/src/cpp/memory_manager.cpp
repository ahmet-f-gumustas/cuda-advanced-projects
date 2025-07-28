#include "memory_manager.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace rtid {

// GpuMemoryPool Implementation
GpuMemoryPool& GpuMemoryPool::getInstance() {
    static GpuMemoryPool instance;
    return instance;
}

GpuMemoryPool::~GpuMemoryPool() {
    clear();
}

void* GpuMemoryPool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t aligned_size = alignSize(size, alignment);
    
    // Try to find a suitable free block
    MemoryBlock* block = findFreeBlock(aligned_size);
    
    if (block) {
        // Split block if it's significantly larger
        if (block->size > aligned_size + alignment) {
            splitBlock(block, aligned_size);
        }
        
        block->in_use = true;
        ptr_to_block_[block->ptr] = block;
        allocation_count_++;
        
        return block->ptr;
    }
    
    // No suitable block found, allocate new memory
    void* ptr = nullptr;
    cudaError_t result = cudaMalloc(&ptr, aligned_size);
    
    if (result != cudaSuccess) {
        std::cerr << "GPU memory allocation failed: " << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    
    // Create new block
    auto new_block = std::make_unique<MemoryBlock>(ptr, aligned_size);
    new_block->in_use = true;
    
    MemoryBlock* block_ptr = new_block.get();
    blocks_.push_back(std::move(new_block));
    ptr_to_block_[ptr] = block_ptr;
    
    total_allocated_ += aligned_size;
    allocation_count_++;
    
    return ptr;
}

bool GpuMemoryPool::deallocate(void* ptr) {
    if (!ptr) return true;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        std::cerr << "Warning: Attempting to deallocate unknown pointer" << std::endl;
        return false;
    }
    
    MemoryBlock* block = it->second;
    block->in_use = false;
    ptr_to_block_.erase(it);
    deallocation_count_++;
    
    // Try to merge with adjacent blocks
    mergeAdjacentBlocks();
    
    return true;
}

void* GpuMemoryPool::allocateAsync(size_t size, cudaStream_t stream, size_t alignment) {
    void* ptr = allocate(size, alignment);
    if (ptr) {
        auto it = ptr_to_block_.find(ptr);
        if (it != ptr_to_block_.end()) {
            it->second->stream = stream;
        }
    }
    return ptr;
}

bool GpuMemoryPool::deallocateAsync(void* ptr, cudaStream_t stream) {
    if (!ptr) return true;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        return false;
    }
    
    MemoryBlock* block = it->second;
    
    // If stream matches, deallocate immediately
    if (block->stream == stream || stream == 0) {
        return deallocate(ptr);
    }
    
    // Otherwise, mark for later deallocation when stream completes
    // For simplicity, we'll just deallocate immediately in this implementation
    return deallocate(ptr);
}

void GpuMemoryPool::reserve(size_t total_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = nullptr;
    cudaError_t result = cudaMalloc(&ptr, total_size);
    
    if (result != cudaSuccess) {
        std::cerr << "Failed to reserve GPU memory: " << cudaGetErrorString(result) << std::endl;
        return;
    }
    
    // Create a large free block
    auto block = std::make_unique<MemoryBlock>(ptr, total_size);
    block->in_use = false;
    
    blocks_.push_back(std::move(block));
    total_reserved_ += total_size;
}

void GpuMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        if (block->ptr) {
            cudaFree(block->ptr);
        }
    }
    
    blocks_.clear();
    ptr_to_block_.clear();
    total_allocated_ = 0;
    total_reserved_ = 0;
    allocation_count_ = 0;
    deallocation_count_ = 0;
}

size_t GpuMemoryPool::getTotalAllocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t GpuMemoryPool::getTotalAvailable() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t available = 0;
    for (const auto& block : blocks_) {
        if (!block->in_use) {
            available += block->size;
        }
    }
    return available;
}

size_t GpuMemoryPool::getFragmentation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t free_blocks = 0;
    size_t total_free = 0;
    
    for (const auto& block : blocks_) {
        if (!block->in_use) {
            free_blocks++;
            total_free += block->size;
        }
    }
    
    if (free_blocks <= 1 || total_free == 0) {
        return 0; // No fragmentation
    }
    
    // Fragmentation as percentage
    return (free_blocks * 100) / total_free;
}

void GpuMemoryPool::printMemoryStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "\n=== GPU Memory Pool Statistics ===" << std::endl;
    std::cout << "Total Reserved: " << std::fixed << std::setprecision(2) 
              << (total_reserved_ / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Total Allocated: " << std::fixed << std::setprecision(2)
              << (total_allocated_ / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Total Available: " << std::fixed << std::setprecision(2)
              << (getTotalAvailable() / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Allocations: " << allocation_count_ << std::endl;
    std::cout << "Deallocations: " << deallocation_count_ << std::endl;
    std::cout << "Active Blocks: " << (allocation_count_ - deallocation_count_) << std::endl;
    std::cout << "Fragmentation: " << getFragmentation() << "%" << std::endl;
    std::cout << "================================\n" << std::endl;
}

MemoryBlock* GpuMemoryPool::findFreeBlock(size_t size) {
    MemoryBlock* best_fit = nullptr;
    size_t best_size = SIZE_MAX;
    
    for (auto& block : blocks_) {
        if (!block->in_use && block->size >= size && block->size < best_size) {
            best_fit = block.get();
            best_size = block->size;
        }
    }
    
    return best_fit;
}

void GpuMemoryPool::splitBlock(MemoryBlock* block, size_t size) {
    if (block->size <= size) return;
    
    size_t remaining_size = block->size - size;
    void* new_ptr = static_cast<char*>(block->ptr) + size;
    
    // Create new block for remaining memory
    auto new_block = std::make_unique<MemoryBlock>(new_ptr, remaining_size);
    new_block->in_use = false;
    
    // Update original block size
    block->size = size;
    
    blocks_.push_back(std::move(new_block));
}

void GpuMemoryPool::mergeAdjacentBlocks() {
    // Sort blocks by address for easier merging
    std::sort(blocks_.begin(), blocks_.end(),
              [](const std::unique_ptr<MemoryBlock>& a, const std::unique_ptr<MemoryBlock>& b) {
                  return a->ptr < b->ptr;
              });
    
    // Merge adjacent free blocks
    for (size_t i = 0; i < blocks_.size() - 1; ++i) {
        auto& current = blocks_[i];
        auto& next = blocks_[i + 1];
        
        if (!current->in_use && !next->in_use) {
            char* current_end = static_cast<char*>(current->ptr) + current->size;
            if (current_end == next->ptr) {
                // Merge blocks
                current->size += next->size;
                blocks_.erase(blocks_.begin() + i + 1);
                --i; // Check current block again with new next
            }
        }
    }
}

size_t GpuMemoryPool::alignSize(size_t size, size_t alignment) const {
    return ((size + alignment - 1) / alignment) * alignment;
}

// ScopedGpuMemory Implementation
ScopedGpuMemory::ScopedGpuMemory(size_t size, size_t alignment)
    : ptr_(nullptr), size_(size), stream_(0), use_stream_(false) {
    ptr_ = GpuMemoryPool::getInstance().allocate(size, alignment);
}

ScopedGpuMemory::ScopedGpuMemory(size_t size, cudaStream_t stream, size_t alignment)
    : ptr_(nullptr), size_(size), stream_(stream), use_stream_(true) {
    ptr_ = GpuMemoryPool::getInstance().allocateAsync(size, stream, alignment);
}

ScopedGpuMemory::~ScopedGpuMemory() {
    if (ptr_) {
        if (use_stream_) {
            GpuMemoryPool::getInstance().deallocateAsync(ptr_, stream_);
        } else {
            GpuMemoryPool::getInstance().deallocate(ptr_);
        }
    }
}

ScopedGpuMemory::ScopedGpuMemory(ScopedGpuMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), stream_(other.stream_), use_stream_(other.use_stream_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

ScopedGpuMemory& ScopedGpuMemory::operator=(ScopedGpuMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_) {
            if (use_stream_) {
                GpuMemoryPool::getInstance().deallocateAsync(ptr_, stream_);
            } else {
                GpuMemoryPool::getInstance().deallocate(ptr_);
            }
        }
        
        ptr_ = other.ptr_;
        size_ = other.size_;
        stream_ = other.stream_;
        use_stream_ = other.use_stream_;
        
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

// MemoryUtils Implementation
bool MemoryUtils::copyHostToDevice(void* d_dst, const void* h_src, size_t size, cudaStream_t stream) {
    cudaError_t result;
    if (stream == 0) {
        result = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    } else {
        result = cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
    }
    return result == cudaSuccess;
}

bool MemoryUtils::copyDeviceToHost(void* h_dst, const void* d_src, size_t size, cudaStream_t stream) {
    cudaError_t result;
    if (stream == 0) {
        result = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    } else {
        result = cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);
    }
    return result == cudaSuccess;
}

bool MemoryUtils::copyDeviceToDevice(void* d_dst, const void* d_src, size_t size, cudaStream_t stream) {
    cudaError_t result;
    if (stream == 0) {
        result = cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
    } else {
        result = cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, stream);
    }
    return result == cudaSuccess;
}

bool MemoryUtils::setMemory(void* ptr, int value, size_t size, cudaStream_t stream) {
    cudaError_t result;
    if (stream == 0) {
        result = cudaMemset(ptr, value, size);
    } else {
        result = cudaMemsetAsync(ptr, value, size, stream);
    }
    return result == cudaSuccess;
}

bool MemoryUtils::setMemoryAsync(void* ptr, int value, size_t size, cudaStream_t stream) {
    cudaError_t result = cudaMemsetAsync(ptr, value, size, stream);
    return result == cudaSuccess;
}

size_t MemoryUtils::getAvailableMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t MemoryUtils::getTotalMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

bool MemoryUtils::isPointerOnDevice(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    
    if (result != cudaSuccess) {
        return false;
    }
    
#if CUDA_VERSION >= 10000
    return attributes.type == cudaMemoryTypeDevice;
#else
    return attributes.memoryType == cudaMemoryTypeDevice;
#endif
}

int MemoryUtils::getPointerDevice(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    
    if (result != cudaSuccess) {
        return -1;
    }
    
    return attributes.device;
}

// CudaEvent Implementation
CudaEvent::CudaEvent(unsigned int flags) : event_(0) {
    cudaEventCreate(&event_, flags);
}

CudaEvent::~CudaEvent() {
    if (event_) {
        cudaEventDestroy(event_);
    }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = 0;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        if (event_) {
            cudaEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = 0;
    }
    return *this;
}

bool CudaEvent::record(cudaStream_t stream) {
    return cudaEventRecord(event_, stream) == cudaSuccess;
}

bool CudaEvent::synchronize() {
    return cudaEventSynchronize(event_) == cudaSuccess;
}

bool CudaEvent::query() const {
    return cudaEventQuery(event_) == cudaSuccess;
}

float CudaEvent::elapsedTime(const CudaEvent& start) const {
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start.event_, event_);
    return milliseconds;
}

// CudaStream Implementation
CudaStream::CudaStream(unsigned int flags) : stream_(0) {
    cudaStreamCreate(&stream_);
}

CudaStream::~CudaStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = 0;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = 0;
    }
    return *this;
}

bool CudaStream::synchronize() {
    return cudaStreamSynchronize(stream_) == cudaSuccess;
}

bool CudaStream::query() const {
    return cudaStreamQuery(stream_) == cudaSuccess;
}

} // namespace rtid