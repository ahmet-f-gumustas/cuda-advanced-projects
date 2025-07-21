#include <gtest/gtest.h>
#include "utils/profiler.h"
#include "utils/logger.h"
#include <thread>
#include <chrono>
#include <fstream>

using namespace deep_engine;

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset singleton states
        Profiler::instance().reset();
        Logger::instance().set_level(LogLevel::INFO);
    }
};

// Logger tests
TEST_F(UtilsTest, LoggerBasicLogging) {
    // Capture output by redirecting to file
    std::string log_file = "test_log.txt";
    Logger::instance().set_file(log_file);
    Logger::instance().enable_console(false);
    
    // Test different log levels
    LOG_TRACE("This is a trace message: %d", 1);
    LOG_DEBUG("This is a debug message: %s", "test");
    LOG_INFO("This is an info message: %f", 3.14);
    LOG_WARNING("This is a warning message");
    LOG_ERROR("This is an error message");
    LOG_CRITICAL("This is a critical message");
    
    // Read log file and verify content
    std::ifstream file(log_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    // Should contain INFO and above (not TRACE or DEBUG by default)
    EXPECT_TRUE(content.find("info message") != std::string::npos);
    EXPECT_TRUE(content.find("warning message") != std::string::npos);
    EXPECT_TRUE(content.find("error message") != std::string::npos);
    EXPECT_TRUE(content.find("critical message") != std::string::npos);
    EXPECT_TRUE(content.find("trace message") == std::string::npos);
    EXPECT_TRUE(content.find("debug message") == std::string::npos);
    
    // Clean up
    file.close();
    std::remove(log_file.c_str());
}

TEST_F(UtilsTest, LoggerLevelFiltering) {
    Logger::instance().set_level(LogLevel::WARNING);
    
    std::string log_file = "test_log_filter.txt";
    Logger::instance().set_file(log_file);
    Logger::instance().enable_console(false);
    
    LOG_INFO("This should not appear");
    LOG_WARNING("This should appear");
    LOG_ERROR("This should also appear");
    
    std::ifstream file(log_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    EXPECT_TRUE(content.find("should not appear") == std::string::npos);
    EXPECT_TRUE(content.find("should appear") != std::string::npos);
    EXPECT_TRUE(content.find("should also appear") != std::string::npos);
    
    file.close();
    std::remove(log_file.c_str());
}

// Profiler tests
TEST_F(UtilsTest, ProfilerBasic) {
    Profiler::instance().enable(true);
    
    // Record some timings
    Profiler::instance().start_layer("conv1", "Conv2d");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Profiler::instance().end_layer("conv1");
    
    Profiler::instance().start_layer("relu1", "ReLU");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    Profiler::instance().end_layer("relu1");
    
    // Get profiles
    auto profiles = Profiler::instance().get_layer_profiles();
    
    EXPECT_EQ(profiles.size(), 2);
    
    // Find conv1 profile
    auto conv_profile = std::find_if(profiles.begin(), profiles.end(),
        [](const LayerProfile& p) { return p.name == "conv1"; });
    
    EXPECT_NE(conv_profile, profiles.end());
    EXPECT_EQ(conv_profile->type, "Conv2d");
    EXPECT_GT(conv_profile->forward_time_ms, 0.0f);
}

TEST_F(UtilsTest, ProfilerScope) {
    Profiler::instance().enable(true);
    
    {
        ProfileScope scope("test_operation", "TestType");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    auto profiles = Profiler::instance().get_layer_profiles();
    EXPECT_EQ(profiles.size(), 1);
    EXPECT_EQ(profiles[0].name, "test_operation");
}

TEST_F(UtilsTest, ProfilerDisabled) {
    Profiler::instance().enable(false);
    
    Profiler::instance().start_layer("conv1", "Conv2d");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Profiler::instance().end_layer("conv1");
    
    auto profiles = Profiler::instance().get_layer_profiles();
    EXPECT_EQ(profiles.size(), 0);  // Should not record when disabled
}

TEST_F(UtilsTest, ProfilerMemoryTracking) {
    Profiler::instance().enable(true);
    
    Profiler::instance().record_memory_allocation("tensor1", 1024 * 1024);  // 1MB
    Profiler::instance().record_memory_allocation("tensor2", 2 * 1024 * 1024);  // 2MB
    Profiler::instance().record_memory_deallocation("tensor1", 1024 * 1024);
    
    // This test mainly verifies the API works without crashing
    EXPECT_TRUE(true);
}

TEST_F(UtilsTest, ProfilerFLOPSRecording) {
    Profiler::instance().enable(true);
    
    Profiler::instance().record_flops("conv1", 1000000);  // 1M FLOPS
    Profiler::instance().record_flops("fc1", 2000000);    // 2M FLOPS
    
    // This test mainly verifies the API works without crashing
    EXPECT_TRUE(true);
}

// CudaTimer tests
TEST_F(UtilsTest, CudaTimer) {
    CudaTimer timer;
    
    timer.start();
    // Do some GPU work
    cudaDeviceSynchronize();
    timer.stop();
    
    float elapsed = timer.elapsed_ms();
    EXPECT_GE(elapsed, 0.0f);  // Should be non-negative
}

// Memory tracker tests
TEST_F(UtilsTest, MemoryTrackerBasic) {
    auto& tracker = MemoryTracker::instance();
    
    size_t initial = tracker.get_current_usage();
    
    // Track some allocations
    void* fake_ptr1 = reinterpret_cast<void*>(0x1000);
    void* fake_ptr2 = reinterpret_cast<void*>(0x2000);
    
    tracker.track_allocation(fake_ptr1, 1024, "test1");
    EXPECT_EQ(tracker.get_current_usage(), initial + 1024);
    
    tracker.track_allocation(fake_ptr2, 2048, "test2");
    EXPECT_EQ(tracker.get_current_usage(), initial + 3072);
    EXPECT_GE(tracker.get_peak_usage(), initial + 3072);
    
    tracker.track_deallocation(fake_ptr1);
    EXPECT_EQ(tracker.get_current_usage(), initial + 2048);
    
    tracker.track_deallocation(fake_ptr2);
    EXPECT_EQ(tracker.get_current_usage(), initial);
}

// Performance monitor test (if implemented)
TEST_F(UtilsTest, DISABLED_PerformanceMonitor) {
    PerformanceMonitor monitor;
    
    monitor.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    monitor.stop();
    
    auto metrics = monitor.get_metrics();
    
    // Basic sanity checks
    EXPECT_GE(metrics.gpu_utilization_percent, 0.0f);
    EXPECT_LE(metrics.gpu_utilization_percent, 100.0f);
    EXPECT_GE(metrics.memory_usage_mb, 0);
}

// Multi-threaded logger test
TEST_F(UtilsTest, LoggerMultiThreaded) {
    std::string log_file = "test_log_mt.txt";
    Logger::instance().set_file(log_file);
    Logger::instance().enable_console(false);
    
    const int num_threads = 4;
    const int logs_per_thread = 100;
    
    auto worker = [logs_per_thread](int thread_id) {
        for (int i = 0; i < logs_per_thread; ++i) {
            LOG_INFO("Thread %d, iteration %d", thread_id, i);
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Count lines in log file
    std::ifstream file(log_file);
    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        line_count++;
    }
    
    EXPECT_EQ(line_count, num_threads * logs_per_thread);
    
    file.close();
    std::remove(log_file.c_str());
}

// Chrome trace export test
TEST_F(UtilsTest, ProfilerChromeTrace) {
    Profiler::instance().enable(true);
    
    // Record some nested operations
    Profiler::instance().start_layer("forward", "Model");
    
    Profiler::instance().start_layer("conv1", "Conv2d");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    Profiler::instance().end_layer("conv1");
    
    Profiler::instance().start_layer("relu1", "ReLU");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Profiler::instance().end_layer("relu1");
    
    Profiler::instance().end_layer("forward");
    
    // Export to Chrome trace format
    std::string trace_file = "test_trace.json";
    Profiler::instance().export_chrome_trace(trace_file);
    
    // Verify file exists and has content
    std::ifstream file(trace_file);
    EXPECT_TRUE(file.good());
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    // Should contain JSON structure
    EXPECT_TRUE(content.find("{") != std::string::npos);
    EXPECT_TRUE(content.find("}") != std::string::npos);
    
    file.close();
    std::remove(trace_file.c_str());
}

// NVTX markers test
TEST_F(UtilsTest, ProfilerNVTX) {
    Profiler::instance().enable(true);
    
    // Test NVTX push/pop
    Profiler::instance().push_range("test_range", 0xFF00FF00);  // Green
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Profiler::instance().pop_range();
    
    // This mainly tests that NVTX calls don't crash
    EXPECT_TRUE(true);
}