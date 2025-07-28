#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace rtid {

enum class DenoiseAlgorithm {
    BILATERAL,
    NON_LOCAL_MEANS,
    GAUSSIAN,
    ADAPTIVE_BILATERAL
};

struct DenoiseParams {
    float sigma_color = 50.0f;
    float sigma_space = 50.0f;
    float h = 10.0f;  // NLM filtering strength
    int template_window_size = 7;
    int search_window_size = 21;
    float gaussian_sigma = 1.0f;
    int kernel_size = 5;
};

class CudaDenoiser {
public:
    CudaDenoiser(int max_width = 1920, int max_height = 1080);
    ~CudaDenoiser();

    // Main denoising functions
    bool denoise(const cv::Mat& input, cv::Mat& output, DenoiseAlgorithm algo, const DenoiseParams& params);
    bool denoiseAsync(const cv::Mat& input, cv::Mat& output, DenoiseAlgorithm algo, const DenoiseParams& params, cudaStream_t stream = 0);
    
    // Batch processing
    bool denoiseBatch(const std::vector<cv::Mat>& inputs, std::vector<cv::Mat>& outputs, 
                      DenoiseAlgorithm algo, const DenoiseParams& params);
    
    // Performance monitoring
    double getLastProcessingTime() const { return last_processing_time_ms_; }
    void resetPerformanceCounters();
    void printPerformanceStats() const;
    
    // Memory management
    bool allocateBuffers(int width, int height);
    void freeBuffers();
    size_t getMemoryUsage() const;
    
    // Algorithm-specific functions
    bool bilateralFilter(float* d_input, float* d_output, int width, int height, const DenoiseParams& params, cudaStream_t stream = 0);
    bool nonLocalMeans(float* d_input, float* d_output, int width, int height, const DenoiseParams& params, cudaStream_t stream = 0);
    bool gaussianFilter(float* d_input, float* d_output, int width, int height, const DenoiseParams& params, cudaStream_t stream = 0);
    bool adaptiveBilateral(float* d_input, float* d_output, int width, int height, const DenoiseParams& params, cudaStream_t stream = 0);

private:
    // Device memory pointers
    float* d_input_buffer_;
    float* d_output_buffer_;
    float* d_temp_buffer_;
    unsigned char* d_input_uchar_;
    unsigned char* d_output_uchar_;
    
    // Memory management
    size_t buffer_size_;
    int allocated_width_;
    int allocated_height_;
    
    // CUDA streams for async processing
    cudaStream_t stream1_;
    cudaStream_t stream2_;
    
    // Performance tracking
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    double last_processing_time_ms_;
    std::vector<double> processing_times_;
    
    // Utility functions
    bool uploadImage(const cv::Mat& image, float* d_buffer);
    bool downloadImage(float* d_buffer, cv::Mat& image, int width, int height);
    bool checkCudaError(const char* operation) const;
    void initializeCudaEvents();
    void cleanupCudaEvents();
};

// Utility functions for external use
bool initializeCuda();
void cleanupCuda();
int getOptimalBlockSize(int problem_size);
dim3 getOptimalGridSize(int width, int height, int block_size);

} // namespace rtid