#pragma once

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

class Timer {
public:
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<float, std::milli>;
    
    void start() {
        start_time_ = clock::now();
    }
    
    float stop() {
        auto end_time = clock::now();
        auto elapsed = std::chrono::duration_cast<duration>(end_time - start_time_);
        float ms = elapsed.count();
        measurements_.push_back(ms);
        return ms;
    }
    
    void reset() {
        measurements_.clear();
    }
    
    float get_mean() const {
        if (measurements_.empty()) return 0.0f;
        return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / measurements_.size();
    }
    
    float get_percentile(float p) const {
        if (measurements_.empty()) return 0.0f;
        
        std::vector<float> sorted = measurements_;
        std::sort(sorted.begin(), sorted.end());
        
        size_t idx = static_cast<size_t>(p * sorted.size());
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        
        return sorted[idx];
    }
    
private:
    clock::time_point start_time_;
    std::vector<float> measurements_;
};

// CUDA event tabanlı timer
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    
    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        measurements_.push_back(ms);
        return ms;
    }
    
    void reset() {
        measurements_.clear();
    }
    
    float get_mean() const {
        if (measurements_.empty()) return 0.0f;
        return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / measurements_.size();
    }
    
    float get_percentile(float p) const {
        if (measurements_.empty()) return 0.0f;
        
        std::vector<float> sorted = measurements_;
        std::sort(sorted.begin(), sorted.end());
        
        size_t idx = static_cast<size_t>(p * sorted.size());
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        
        return sorted[idx];
    }
    
private:
    cudaEvent_t start_, stop_;
    std::vector<float> measurements_;
};