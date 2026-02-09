#include "memory_manager.hpp"

namespace cuda_stabilizer {

MemoryManager::MemoryManager()
    : total_allocated_(0)
    , peak_allocated_(0)
{
}

MemoryManager::~MemoryManager() {
    freeAll();
}

void MemoryManager::free(void* ptr) {
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        if (!it->second.is_pinned) {
            cudaFree(ptr);
        }
        total_allocated_ -= it->second.size;
        allocations_.erase(it);
    }
}

void MemoryManager::freePinned(void* ptr) {
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        if (it->second.is_pinned) {
            cudaFreeHost(ptr);
        }
        total_allocated_ -= it->second.size;
        allocations_.erase(it);
    }
}

void MemoryManager::freeAll() {
    for (auto& pair : allocations_) {
        if (pair.second.is_pinned) {
            cudaFreeHost(pair.first);
        } else {
            cudaFree(pair.first);
        }
    }
    allocations_.clear();
    total_allocated_ = 0;
}

void MemoryManager::getDeviceMemoryInfo(size_t& free_bytes, size_t& total_bytes) const {
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

void MemoryManager::printStats() const {
    size_t free_bytes, total_bytes;
    getDeviceMemoryInfo(free_bytes, total_bytes);

    std::cout << "=== Memory Manager Statistics ===" << std::endl;
    std::cout << "Allocations: " << allocations_.size() << std::endl;
    std::cout << "Total allocated: " << (total_allocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Peak allocated: " << (peak_allocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Device memory free: " << (free_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Device memory total: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "=================================" << std::endl;

    if (!allocations_.empty()) {
        std::cout << "\nActive allocations:" << std::endl;
        for (const auto& pair : allocations_) {
            std::cout << "  " << pair.second.name
                      << " (" << (pair.second.size / 1024.0) << " KB)"
                      << (pair.second.is_pinned ? " [pinned]" : " [device]")
                      << std::endl;
        }
    }
}

} // namespace cuda_stabilizer
