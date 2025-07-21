#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cuda_runtime.h>
#include "../core/types.h"

namespace deep_engine {

// Memory block information
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool allocated;
    int device_id;
    
    MemoryBlock* next;
    MemoryBlock* prev;
};

// Memory pool for efficient allocation
class MemoryPool {
public:
    explicit MemoryPool(size_t initial_size = 1024 * 1024 * 1024,  // 1GB
                       int device_id = 0);
    ~MemoryPool();
    
    // Allocation and deallocation
    void* allocate(size_t size);
    void deallocate(void* ptr);
    
    // Pool management
    void reset();
    void defragment();
    size_t get_allocated_size() const;
    size_t get_total_size() const;
    size_t get_fragmentation() const;
    
    // Statistics
    struct Stats {
        size_t total_allocations;
        size_t total_deallocations;
        size_t current_allocations;
        size_t peak_usage;
        size_t fragmentation_count;
    };
    
    Stats get_stats() const { return stats_; }
    void print_stats() const;
    
private:
    int device_id_;
    void* pool_start_;
    size_t pool_size_;
    
    MemoryBlock* free_list_;
    MemoryBlock* allocated_list_;
    std::unordered_map<void*, MemoryBlock*> allocated_blocks_;
    
    mutable std::mutex mutex_;
    Stats stats_;
    
    MemoryBlock* find_free_block(size_t size);
    void split_block(MemoryBlock* block, size_t size);
    void coalesce_free_blocks();
    void grow_pool(size_t additional_size);
};

// Memory pool manager for multiple devices
class MemoryPoolManager {
public:
    static MemoryPoolManager& instance() {
        static MemoryPoolManager manager;
        return manager;
    }
    
    MemoryPool* get_pool(int device_id);
    void* allocate(size_t size, int device_id = -1);
    void deallocate(void* ptr, int device_id = -1);
    
    void reset_all();
    void print_all_stats() const;
    
private:
    MemoryPoolManager() = default;
    std::unordered_map<int, std::unique_ptr<MemoryPool>> pools_;
    mutable std::mutex mutex_;
};

// Workspace manager for temporary allocations
class WorkspaceManager {
public:
    explicit WorkspaceManager(size_t max_workspace_size = 512 * 1024 * 1024);  // 512MB
    ~WorkspaceManager();
    
    void* get_workspace(size_t size);
    void release_workspace(void* ptr);
    
private:
    struct Workspace {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Workspace> workspaces_;
    size_t max_workspace_size_;
    mutable std::mutex mutex_;
};

// Memory planner for static graph execution
class MemoryPlanner {
public:
    struct Allocation {
        size_t offset;
        size_t size;
        int lifetime_start;
        int lifetime_end;
    };
    
    void plan(const std::vector<std::pair<size_t, std::pair<int, int>>>& allocations);
    size_t get_total_memory_required() const { return total_memory_required_; }
    const std::vector<Allocation>& get_plan() const { return plan_; }
    
private:
    std::vector<Allocation> plan_;
    size_t total_memory_required_;
    
    void greedy_by_size();
    void greedy_by_breadth();
    bool intervals_overlap(const Allocation& a, const Allocation& b);
};

// Smart pointer with custom allocator
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    
    PoolAllocator() = default;
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>&) {}
    
    T* allocate(size_t n) {
        return static_cast<T*>(
            MemoryPoolManager::instance().allocate(n * sizeof(T))
        );
    }
    
    void deallocate(T* ptr, size_t) {
        MemoryPoolManager::instance().deallocate(ptr);
    }
};

template<typename T, typename U>
bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) {
    return true;
}

template<typename T, typename U>
bool operator!=(const PoolAllocator<T>&, const PoolAllocator<U>&) {
    return false;
}

} // namespace deep_engine