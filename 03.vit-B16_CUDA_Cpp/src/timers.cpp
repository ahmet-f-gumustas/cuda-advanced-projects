#include "timers.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

// GPUTimer implementation
void GPUTimer::start(const std::string& name) {
    auto& entry = timers_[name];
    entry.start.record(stream_);
}

void GPUTimer::end(const std::string& name) {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
        throw std::runtime_error("Timer not started: " + name);
    }
    
    auto& entry = it->second;
    entry.end.record(stream_);
    
    // Synchronize and get elapsed time
    entry.end.synchronize();
    float elapsed = entry.end.elapsed_time(entry.start);
    entry.times.push_back(elapsed);
}

float GPUTimer::get_avg_time(const std::string& name) const {
    auto it = timers_.find(name);
    if (it == timers_.end() || it->second.times.empty()) {
        return 0.0f;
    }
    
    const auto& times = it->second.times;
    return std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
}

void GPUTimer::print_summary() const {
    std::cout << "\nGPU Timing Summary:\n";
    std::cout << std::setw(30) << "Operation" << std::setw(15) << "Avg Time (ms)" 
              << std::setw(15) << "Calls" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& [name, entry] : timers_) {
        if (!entry.times.empty()) {
            std::cout << std::setw(30) << name 
                     << std::setw(15) << std::fixed << std::setprecision(3) 
                     << get_avg_time(name)
                     << std::setw(15) << entry.times.size() << "\n";
        }
    }
}

void GPUTimer::reset() {
    timers_.clear();
}

// CPUTimer implementation
void CPUTimer::start(const std::string& name) {
    start_times_[name] = Clock::now();
}

void CPUTimer::end(const std::string& name) {
    auto it = start_times_.find(name);
    if (it == start_times_.end()) {
        throw std::runtime_error("Timer not started: " + name);
    }
    
    auto end_time = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                   (end_time - it->second);
    
    durations_[name].push_back(duration.count() / 1000.0f); // Convert to ms
    start_times_.erase(it);
}

float CPUTimer::get_avg_time(const std::string& name) const {
    auto it = durations_.find(name);
    if (it == durations_.end() || it->second.empty()) {
        return 0.0f;
    }
    
    const auto& times = it->second;
    return std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
}

void CPUTimer::print_summary() const {
    std::cout << "\nCPU Timing Summary:\n";
    std::cout << std::setw(30) << "Operation" << std::setw(15) << "Avg Time (ms)" 
              << std::setw(15) << "Calls" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& [name, times] : durations_) {
        if (!times.empty()) {
            std::cout << std::setw(30) << name 
                     << std::setw(15) << std::fixed << std::setprecision(3) 
                     << get_avg_time(name)
                     << std::setw(15) << times.size() << "\n";
        }
    }
}