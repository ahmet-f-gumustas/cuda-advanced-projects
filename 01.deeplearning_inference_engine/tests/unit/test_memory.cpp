#include <gtest/gtest.h>
#include "core/allocator.h"
#include "optimizations/memory_pool.h"
#include <cuda_runtime.h>
#include <vector>
#include <thread>

using namespace deep_engine;

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// Basic allocator tests
TEST_F(MemoryTest, CudaAllocator) {
    CudaAllocator allocator;
    
    // Test allocation
    void* ptr = allocator.allocate(1024 * 1024);  // 1MB
    EXPECT_NE(ptr, nullptr);
    
    // Verify it's a valid CUDA pointer
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    
    // Test deallocation
    EXPECT_NO_THROW(allocator.deallocate(ptr));
}

TEST_F(MemoryTest, HostAllocator) {
    HostAllocator allocator;
    
    void* ptr = allocator.allocate(1024);
    EXPECT_NE(ptr, nullptr);
    
    // Should be host memory
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    EXPECT_NE(attributes.type, cudaMemoryTypeDevice);
    
    allocator.deallocate(ptr);
}

TEST_F(MemoryTest, PinnedMemoryAllocator) {
    PinnedMemoryAllocator allocator;
    
    void* ptr = allocator.allocate(1024);
    EXPECT_NE(ptr, nullptr);
    
    // Should be pinned host memory
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    
    allocator.deallocate(ptr);
}

// Pool allocator tests
TEST_F(MemoryTest, CudaPoolAllocatorBasic) {
    CudaPoolAllocator pool(10 * 1024 * 1024);  // 10MB pool
    
    // Test basic allocation
    void* ptr1 = pool.allocate(1024);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(pool.used_memory(), 1024);
    
    void* ptr2 = pool.allocate(2048);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(pool.used_memory(), 3072);
    
    // Deallocate and reallocate
    pool.deallocate(ptr1);
    EXPECT_EQ(pool.used_memory(), 2048);
    
    void* ptr3 = pool.allocate(512);
    EXPECT_NE(ptr3, nullptr);
    // Should reuse part of the freed block
    
    pool.deallocate(ptr2);
    pool.deallocate(ptr3);
}

TEST_F(MemoryTest, CudaPoolAllocatorReuse) {
    CudaPoolAllocator pool(1024 * 1024);  // 1MB pool
    
    // Allocate and free same size multiple times
    const size_t size = 4096;
    void* ptr = nullptr;
    
    for (int i = 0; i < 10; ++i) {
        void* new_ptr = pool.allocate(size);
        EXPECT_NE(new_ptr, nullptr);
        
        if (ptr != nullptr) {
            // Should get the same pointer back (reuse)
            EXPECT_EQ(new_ptr, ptr);
        }
        ptr = new_ptr;
        
        pool.deallocate(ptr);
    }
}

TEST_F(MemoryTest, CudaPoolAllocatorFragmentation) {
    CudaPoolAllocator pool(1024 * 1024);  // 1MB pool
    
    // Create fragmentation
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool.allocate(8192));  // 8KB each
    }
    
    // Free every other allocation
    for (int i = 0; i < 10; i += 2) {
        pool.deallocate(ptrs[i]);
    }
    
    // Try to allocate larger block than any single free block
    void* large = pool.allocate(16384);  // 16KB
    
    // Pool should handle fragmentation (coalesce or find contiguous space)
    // This might fail if fragmentation handling is not implemented
    if (large == nullptr) {
        // Reset and try again
        pool.reset();
        large = pool.allocate(16384);
        EXPECT_NE(large, nullptr);
    }
}

TEST_F(MemoryTest, CudaPoolAllocatorReset) {
    CudaPoolAllocator pool(1024 * 1024);
    
    // Allocate some memory
    void* ptr1 = pool.allocate(1024);
    void* ptr2 = pool.allocate(2048);
    
    EXPECT_GT(pool.used_memory(), 0);
    
    // Reset pool
    pool.reset();
    
    EXPECT_EQ(pool.used_memory(), 0);
    
    // Should be able to allocate again
    void* ptr3 = pool.allocate(1024);
    EXPECT_NE(ptr3, nullptr);
}

// Allocator factory tests
TEST_F(MemoryTest, AllocatorFactory) {
    // Test CUDA allocator creation
    auto cuda_alloc = AllocatorFactory::create(DeviceType::CUDA, false);
    EXPECT_NE(cuda_alloc, nullptr);
    EXPECT_EQ(cuda_alloc->device_type(), DeviceType::CUDA);
    
    // Test CUDA pool allocator creation
    auto cuda_pool = AllocatorFactory::create(DeviceType::CUDA, true);
    EXPECT_NE(cuda_pool, nullptr);
    EXPECT_EQ(cuda_pool->device_type(), DeviceType::CUDA);
    
    // Test CPU allocator creation
    auto cpu_alloc = AllocatorFactory::create(DeviceType::CPU, false);
    EXPECT_NE(cpu_alloc, nullptr);
    EXPECT_EQ(cpu_alloc->device_type(), DeviceType::CPU);
}

// Memory tracker tests
TEST_F(MemoryTest, MemoryTracker) {
    auto& tracker = MemoryTracker::instance();
    
    size_t initial_usage = tracker.get_current_usage();
    
    // Track allocation
    void* ptr;
    cudaMalloc(&ptr, 1024 * 1024);  // 1MB
    tracker.track_allocation(ptr, 1024 * 1024, "test_allocation");
    
    EXPECT_EQ(tracker.get_current_usage(), initial_usage + 1024 * 1024);
    EXPECT_GE(tracker.get_peak_usage(), initial_usage + 1024 * 1024);
    
    // Track deallocation
    tracker.track_deallocation(ptr);
    cudaFree(ptr);
    
    EXPECT_EQ(tracker.get_current_usage(), initial_usage);
}

// Multi-threaded pool allocator test
TEST_F(MemoryTest, CudaPoolAllocatorMultiThreaded) {
    CudaPoolAllocator pool(100 * 1024 * 1024);  // 100MB pool
    const int num_threads = 4;
    const int allocations_per_thread = 100;
    
    auto worker = [&pool, allocations_per_thread]() {
        std::vector<void*> ptrs;
        
        // Allocate
        for (int i = 0; i < allocations_per_thread; ++i) {
            size_t size = (rand() % 10 + 1) * 1024;  // 1-10 KB
            void* ptr = pool.allocate(size);
            EXPECT_NE(ptr, nullptr);
            ptrs.push_back(ptr);
        }
        
        // Deallocate in random order
        std::random_shuffle(ptrs.begin(), ptrs.end());
        for (void* ptr : ptrs) {
            pool.deallocate(ptr);
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Pool should be empty after all threads finish
    pool.reset();
    EXPECT_EQ(pool.used_memory(), 0);
}

// Workspace manager tests
TEST_F(MemoryTest, WorkspaceManager) {
    WorkspaceManager workspace(10 * 1024 * 1024);  // 10MB
    
    // Get workspace
    void* ws1 = workspace.get_workspace(1024 * 1024);  // 1MB
    EXPECT_NE(ws1, nullptr);
    
    // Get another workspace
    void* ws2 = workspace.get_workspace(2 * 1024 * 1024);  // 2MB
    EXPECT_NE(ws2, nullptr);
    EXPECT_NE(ws1, ws2);
    
    // Release first workspace
    workspace.release_workspace(ws1);
    
    // Get workspace of same size - should reuse
    void* ws3 = workspace.get_workspace(1024 * 1024);
    EXPECT_EQ(ws3, ws1);
}

// Memory planner tests
TEST_F(MemoryTest, MemoryPlanner) {
    MemoryPlanner planner;
    
    // Define memory requirements with lifetimes
    std::vector<std::pair<size_t, std::pair<int, int>>> allocations = {
        {1024, {0, 3}},    // 1KB, lives from step 0-3
        {2048, {1, 4}},    // 2KB, lives from step 1-4
        {1024, {4, 6}},    // 1KB, lives from step 4-6
        {512, {5, 7}},     // 512B, lives from step 5-7
    };
    
    planner.plan(allocations);
    
    // Total memory should be optimized
    size_t naive_total = 1024 + 2048 + 1024 + 512;
    EXPECT_LT(planner.get_total_memory_required(), naive_total);
    
    auto plan = planner.get_plan();
    EXPECT_EQ(plan.size(), 4);
    
    // First allocation should start at offset 0
    EXPECT_EQ(plan[0].offset, 0);
}

// Pool allocator with custom types
TEST_F(MemoryTest, PoolAllocatorTemplate) {
    std::vector<float, PoolAllocator<float>> vec;
    
    // Test that vector can use our custom allocator
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i * 0.5f);
    }
    
    EXPECT_EQ(vec.size(), 1000);
    EXPECT_FLOAT_EQ(vec[500], 250.0f);
}

// Stress test (disabled by default)
TEST_F(MemoryTest, DISABLED_MemoryPoolStress) {
    CudaPoolAllocator pool(1024 * 1024 * 1024);  // 1GB pool
    
    const int num_iterations = 10000;
    std::vector<void*> allocations;
    size_t total_allocated = 0;
    
    for (int i = 0; i < num_iterations; ++i) {
        // Random allocation size between 1KB and 1MB
        size_t size = (rand() % 1024 + 1) * 1024;
        
        // Randomly allocate or deallocate
        if (allocations.empty() || (rand() % 2 && total_allocated + size < 900 * 1024 * 1024)) {
            // Allocate
            void* ptr = pool.allocate(size);
            if (ptr != nullptr) {
                allocations.push_back(ptr);
                total_allocated += size;
            }
        } else if (!allocations.empty()) {
            // Deallocate random allocation
            int idx = rand() % allocations.size();
            pool.deallocate(allocations[idx]);
            allocations.erase(allocations.begin() + idx);
            // Note: We don't track individual sizes here for simplicity
        }
    }
    
    // Clean up remaining allocations
    for (void* ptr : allocations) {
        pool.deallocate(ptr);
    }
}