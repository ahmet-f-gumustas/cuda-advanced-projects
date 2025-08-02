#pragma once
#include "cuda_utils.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>

class GPUTimer {
private:
    struct TimerEntry {
        CudaEvent start;
        CudaEvent end;
        std::vector<float> times;
    };
    
    std::unordered_map<std::string, TimerEntry> timers_;
    cudaStream_t stream_;
    
public:
    explicit GPUTimer(cudaStream_t stream = 0) : stream_(stream) {}
    
    void start(const std::string& name);
    void end(const std::string& name);
    
    float get_avg_time(const std::string& name) const;
    void print_summary() const;
    void reset();
};

class CPUTimer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
    std::unordered_map<std::string, TimePoint> start_times_;
    std::unordered_map<std::string, std::vector<float>> durations_;
    
public:
    void start(const std::string& name);
    void end(const std::string& name);
    
    float get_avg_time(const std::string& name) const;
    void print_summary() const;
};