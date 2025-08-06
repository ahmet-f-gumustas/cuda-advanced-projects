#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }
    
    void set_level(LogLevel level) {
        level_ = level;
    }
    
    template<typename... Args>
    void debug(Args&&... args) {
        log(LogLevel::DEBUG, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void info(Args&&... args) {
        log(LogLevel::INFO, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void warning(Args&&... args) {
        log(LogLevel::WARNING, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void error(Args&&... args) {
        log(LogLevel::ERROR, std::forward<Args>(args)...);
    }
    
private:
    Logger() : level_(LogLevel::INFO) {}
    
    template<typename... Args>
    void log(LogLevel level, Args&&... args) {
        if (level < level_) return;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] ";
        ss << "[" << level_to_string(level) << "] ";
        
        ((ss << args << " "), ...);
        
        std::cout << ss.str() << std::endl;
    }
    
    const char* level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }
    
    LogLevel level_;
};

#define LOG_DEBUG(...) Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...) Logger::instance().info(__VA_ARGS__)
#define LOG_WARNING(...) Logger::instance().warning(__VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)