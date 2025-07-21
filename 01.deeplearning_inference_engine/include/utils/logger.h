#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace deep_engine {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }
    
    void set_level(LogLevel level) { level_ = level; }
    LogLevel get_level() const { return level_; }
    
    void set_file(const std::string& filename);
    void enable_console(bool enable) { console_enabled_ = enable; }
    
    template<typename... Args>
    void log(LogLevel level, const std::string& format, Args... args) {
        if (level < level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
        ss << "[" << level_to_string(level) << "] ";
        
        // Format message
        size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
        std::unique_ptr<char[]> buf(new char[size]);
        snprintf(buf.get(), size, format.c_str(), args...);
        ss << std::string(buf.get(), buf.get() + size - 1);
        
        std::string message = ss.str();
        
        if (console_enabled_) {
            std::cout << message << std::endl;
        }
        
        if (file_stream_.is_open()) {
            file_stream_ << message << std::endl;
            file_stream_.flush();
        }
    }
    
    // Convenience methods
    template<typename... Args>
    void trace(const std::string& format, Args... args) {
        log(LogLevel::TRACE, format, args...);
    }
    
    template<typename... Args>
    void debug(const std::string& format, Args... args) {
        log(LogLevel::DEBUG, format, args...);
    }
    
    template<typename... Args>
    void info(const std::string& format, Args... args) {
        log(LogLevel::INFO, format, args...);
    }
    
    template<typename... Args>
    void warning(const std::string& format, Args... args) {
        log(LogLevel::WARNING, format, args...);
    }
    
    template<typename... Args>
    void error(const std::string& format, Args... args) {
        log(LogLevel::ERROR, format, args...);
    }
    
    template<typename... Args>
    void critical(const std::string& format, Args... args) {
        log(LogLevel::CRITICAL, format, args...);
    }
    
private:
    Logger() : level_(LogLevel::INFO), console_enabled_(true) {}
    
    LogLevel level_;
    bool console_enabled_;
    std::ofstream file_stream_;
    std::mutex mutex_;
    
    std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }
};

// Convenience macros
#define LOG_TRACE(...) deep_engine::Logger::instance().trace(__VA_ARGS__)
#define LOG_DEBUG(...) deep_engine::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...) deep_engine::Logger::instance().info(__VA_ARGS__)
#define LOG_WARNING(...) deep_engine::Logger::instance().warning(__VA_ARGS__)
#define LOG_ERROR(...) deep_engine::Logger::instance().error(__VA_ARGS__)
#define LOG_CRITICAL(...) deep_engine::Logger::instance().critical(__VA_ARGS__)

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            LOG_ERROR("CUDA error at %s:%d - %s", __FILE__, __LINE__, \
                     cudaGetErrorString(error)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// cuDNN error checking
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            LOG_ERROR("cuDNN error at %s:%d - %s", __FILE__, __LINE__, \
                     cudnnGetErrorString(status)); \
            throw std::runtime_error("cuDNN error"); \
        } \
    } while(0)

// cuBLAS error checking
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            LOG_ERROR("cuBLAS error at %s:%d - code %d", __FILE__, __LINE__, \
                     status); \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

} // namespace deep_engine