#pragma once

#include "cuda_denoiser.h"
#include "memory_manager.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

namespace rtid {

struct ProcessingStats {
    double avg_fps = 0.0;
    double min_fps = 0.0;
    double max_fps = 0.0;
    double avg_latency_ms = 0.0;
    size_t frames_processed = 0;
    size_t frames_dropped = 0;
    double gpu_utilization = 0.0;
    size_t memory_usage_mb = 0;
};

struct FrameData {
    cv::Mat image;
    std::chrono::high_resolution_clock::time_point timestamp;
    int frame_id;
    DenoiseAlgorithm algorithm;
    DenoiseParams params;
    std::promise<cv::Mat> result_promise;
    
    FrameData() = default;
    FrameData(const cv::Mat& img, int id, DenoiseAlgorithm algo, const DenoiseParams& p) 
        : image(img), timestamp(std::chrono::high_resolution_clock::now()), 
          frame_id(id), algorithm(algo), params(p) {}
};

class ImageProcessor {
public:
    ImageProcessor(int max_queue_size = 10);
    ~ImageProcessor();
    
    // Lifecycle management
    bool initialize(int max_width = 1920, int max_height = 1080);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Single frame processing
    bool processFrame(const cv::Mat& input, cv::Mat& output, 
                     DenoiseAlgorithm algo, const DenoiseParams& params);
    
    // Async processing
    std::future<cv::Mat> processFrameAsync(const cv::Mat& input, 
                                          DenoiseAlgorithm algo, 
                                          const DenoiseParams& params);
    
    // Real-time processing pipeline
    bool startRealTimeProcessing();
    void stopRealTimeProcessing();
    bool isProcessing() const { return processing_active_; }
    
    // Frame queue management
    bool enqueueFrame(const cv::Mat& frame, DenoiseAlgorithm algo, const DenoiseParams& params);
    bool dequeueResult(cv::Mat& result, int timeout_ms = 100);
    void clearQueue();
    
    // Camera integration
    bool startCameraCapture(int camera_id = 0);
    void stopCameraCapture();
    bool isCameraActive() const { return camera_active_; }
    
    // Performance monitoring
    ProcessingStats getStats() const;
    void resetStats();
    void enableProfiling(bool enable) { profiling_enabled_ = enable; }
    
    // Configuration
    void setMaxQueueSize(int size);
    void setThreadCount(int count);
    void setDenoiseParams(const DenoiseParams& params) { default_params_ = params; }
    void setDenoiseAlgorithm(DenoiseAlgorithm algo) { default_algorithm_ = algo; }
    
    // Callbacks
    using FrameCallback = std::function<void(const cv::Mat& frame, double processing_time)>;
    void setFrameCallback(FrameCallback callback) { frame_callback_ = callback; }
    
private:
    // Processing thread functions
    void processingWorker();
    void cameraWorker();
    void updateStats(double processing_time);
    
    // Internal utilities
    bool validateFrame(const cv::Mat& frame) const;
    void cleanupResources();
    
    // Core components
    std::unique_ptr<CudaDenoiser> denoiser_;
    cv::VideoCapture camera_;
    
    // Threading
    std::vector<std::thread> worker_threads_;
    std::thread camera_thread_;
    std::atomic<bool> processing_active_;
    std::atomic<bool> camera_active_;
    std::atomic<bool> shutdown_requested_;
    
    // Frame queue
    std::queue<FrameData> input_queue_;
    std::queue<cv::Mat> output_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::condition_variable result_condition_;
    
    // Configuration
    int max_queue_size_;
    int thread_count_;
    int max_width_;
    int max_height_;
    DenoiseParams default_params_;
    DenoiseAlgorithm default_algorithm_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    ProcessingStats stats_;
    std::vector<double> frame_times_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    bool profiling_enabled_;
    
    // State
    std::atomic<bool> initialized_;
    std::atomic<int> frame_counter_;
    
    // Callback
    FrameCallback frame_callback_;
};

// Utility class for benchmarking different algorithms
class DenoiseBenchmark {
public:
    struct BenchmarkResult {
        DenoiseAlgorithm algorithm;
        DenoiseParams params;
        double avg_time_ms;
        double min_time_ms;
        double max_time_ms;
        double std_dev_ms;
        double psnr;
        double ssim;
        size_t memory_usage_mb;
    };
    
    DenoiseBenchmark();
    ~DenoiseBenchmark();
    
    // Run benchmarks
    std::vector<BenchmarkResult> benchmarkAlgorithms(const cv::Mat& test_image, 
                                                     const cv::Mat& reference_clean = cv::Mat(),
                                                     int iterations = 100);
    
    BenchmarkResult benchmarkSingleAlgorithm(const cv::Mat& test_image,
                                            DenoiseAlgorithm algo,
                                            const DenoiseParams& params,
                                            const cv::Mat& reference_clean = cv::Mat(),
                                            int iterations = 100);
    
    // Parameter optimization
    DenoiseParams optimizeParameters(const cv::Mat& test_image,
                                   DenoiseAlgorithm algo,
                                   const cv::Mat& reference_clean,
                                   int max_iterations = 50);
    
    // Quality metrics
    static double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
    static double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2);
    static double calculateMSE(const cv::Mat& img1, const cv::Mat& img2);
    
private:
    std::unique_ptr<CudaDenoiser> denoiser_;
    std::unique_ptr<ImageProcessor> processor_;
};

} // namespace rtid