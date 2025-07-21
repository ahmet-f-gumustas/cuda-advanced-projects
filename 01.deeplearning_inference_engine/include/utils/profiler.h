#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <memory>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace deep_engine {

// CUDA event wrapper for timing
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();
    
    void start(cudaStream_t stream = 0);
    void stop(cudaStream_t stream = 0);
    float elapsed_ms() const;
    
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool started_;
    bool stopped_;
};

// Layer profiling information
struct LayerProfile {
    std::string name;
    std::string type;
    float forward_time_ms;
    float backward_time_ms;
    size_t memory_usage;
    size_t flops;
    int call_count;
    
    float avg_forward_time() const { 
        return call_count > 0 ? forward_time_ms / call_count : 0.0f; 
    }
    
    float avg_backward_time() const { 
        return call_count > 0 ? backward_time_ms / call_count : 0.0f; 
    }
};

// Profiler class
class Profiler {
public:
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }
    
    void enable(bool enable = true) { enabled_ = enable; }
    bool is_enabled() const { return enabled_; }
    
    // Layer profiling
    void start_layer(const std::string& name, const std::string& type,
                    cudaStream_t stream = 0);
    void end_layer(const std::string& name, cudaStream_t stream = 0);
    
    // Memory profiling
    void record_memory_allocation(const std::string& name, size_t bytes);
    void record_memory_deallocation(const std::string& name, size_t bytes);
    
    // FLOPS recording
    void record_flops(const std::string& name, size_t flops);
    
    // NVTX markers for NSight
    void push_range(const std::string& name, uint32_t color = 0xFF00FF00);
    void pop_range();
    
    // Results
    std::vector<LayerProfile> get_layer_profiles() const;
    void print_summary() const;
    void export_chrome_trace(const std::string& filename) const;
    void reset();
    
private:
    Profiler() : enabled_(false) {}
    
    bool enabled_;
    std::unordered_map<std::string, LayerProfile> profiles_;
    std::unordered_map<std::string, std::unique_ptr<CudaTimer>> active_timers_;
    std::unordered_map<std::string, size_t> memory_usage_;
    
    mutable std::mutex mutex_;
};

// RAII profiling scope
class ProfileScope {
public:
    ProfileScope(const std::string& name, const std::string& type = "",
                 cudaStream_t stream = 0)
        : name_(name), stream_(stream) {
        if (Profiler::instance().is_enabled()) {
            Profiler::instance().start_layer(name, type, stream);
        }
    }
    
    ~ProfileScope() {
        if (Profiler::instance().is_enabled()) {
            Profiler::instance().end_layer(name_, stream_);
        }
    }
    
private:
    std::string name_;
    cudaStream_t stream_;
};

// Memory usage tracker
class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }
    
    void track_allocation(void* ptr, size_t size, const std::string& tag = "");
    void track_deallocation(void* ptr);
    
    size_t get_current_usage() const { return current_usage_; }
    size_t get_peak_usage() const { return peak_usage_; }
    
    void print_summary() const;
    void print_allocations() const;
    
private:
    struct AllocationInfo {
        size_t size;
        std::string tag;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations_;
    size_t current_usage_ = 0;
    size_t peak_usage_ = 0;
    mutable std::mutex mutex_;
};

// Performance metrics
struct PerformanceMetrics {
    float throughput_samples_per_sec;
    float latency_ms;
    float gpu_utilization_percent;
    float memory_bandwidth_gbps;
    size_t memory_usage_mb;
    float power_usage_watts;
};

// Performance monitor
class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    void start();
    void stop();
    
    PerformanceMetrics get_metrics() const;
    void print_metrics() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Macros for easy profiling
#define PROFILE_LAYER(name, type) ProfileScope _prof_##__LINE__(name, type)
#define PROFILE_FUNCTION() ProfileScope _prof_##__LINE__(__FUNCTION__)

} // namespace deep_engine