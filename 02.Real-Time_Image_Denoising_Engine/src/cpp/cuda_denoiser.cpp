#include "cuda_denoiser.h"
#include "memory_manager.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>

namespace rtid {

// External function declarations from CUDA kernels
extern "C" {
    bool launchBilateralFilter(float* d_input, float* d_output, int width, int height,
                              float sigma_color, float sigma_space, int kernel_radius,
                              cudaStream_t stream);
    bool launchBilateralFilterShared(float* d_input, float* d_output, int width, int height,
                                    float sigma_color, float sigma_space, int kernel_radius,
                                    cudaStream_t stream);
    bool launchAdaptiveBilateralFilter(float* d_input, float* d_output, int width, int height,
                                      float sigma_color, float sigma_space, int kernel_radius,
                                      cudaStream_t stream);
    bool launchNonLocalMeans(float* d_input, float* d_output, int width, int height,
                            float h_param, int template_radius, int search_radius,
                            cudaStream_t stream);
    bool launchFastNonLocalMeans(float* d_input, float* d_output, int width, int height,
                                float h_param, int template_radius, int search_radius,
                                cudaStream_t stream);
    bool launchGaussianFilter(float* d_input, float* d_output, int width, int height,
                             float sigma, int radius, cudaStream_t stream);
    bool launchSeparableGaussianFilter(float* d_input, float* d_temp, float* d_output,
                                      int width, int height, float sigma, int radius,
                                      cudaStream_t stream);
}

CudaDenoiser::CudaDenoiser(int max_width, int max_height)
    : d_input_buffer_(nullptr)
    , d_output_buffer_(nullptr)
    , d_temp_buffer_(nullptr)
    , d_input_uchar_(nullptr)
    , d_output_uchar_(nullptr)
    , buffer_size_(0)
    , allocated_width_(0)
    , allocated_height_(0)
    , stream1_(0)
    , stream2_(0)
    , start_event_(0)
    , stop_event_(0)
    , last_processing_time_ms_(0.0)
{
    initializeCudaEvents();
    
    // Create CUDA streams
    cudaStreamCreate(&stream1_);
    cudaStreamCreate(&stream2_);
    
    // Pre-allocate buffers if dimensions are provided
    if (max_width > 0 && max_height > 0) {
        allocateBuffers(max_width, max_height);
    }
}

CudaDenoiser::~CudaDenoiser() {
    freeBuffers();
    cleanupCudaEvents();
    
    if (stream1_) {
        cudaStreamDestroy(stream1_);
    }
    if (stream2_) {
        cudaStreamDestroy(stream2_);
    }
}

bool CudaDenoiser::denoise(const cv::Mat& input, cv::Mat& output, 
                          DenoiseAlgorithm algo, const DenoiseParams& params) {
    return denoiseAsync(input, output, algo, params, 0);
}

bool CudaDenoiser::denoiseAsync(const cv::Mat& input, cv::Mat& output,
                               DenoiseAlgorithm algo, const DenoiseParams& params,
                               cudaStream_t stream) {
    if (input.empty()) {
        std::cerr << "Error: Input image is empty" << std::endl;
        return false;
    }
    
    int width = input.cols;
    int height = input.rows;
    
    // Ensure buffers are allocated
    if (!allocateBuffers(width, height)) {
        std::cerr << "Error: Failed to allocate GPU buffers" << std::endl;
        return false;
    }
    
    // Start timing
    cudaEventRecord(start_event_, stream);
    
    // Upload image to GPU
    if (!uploadImage(input, d_input_buffer_)) {
        std::cerr << "Error: Failed to upload image to GPU" << std::endl;
        return false;
    }
    
    bool success = false;
    
    // Apply selected algorithm
    switch (algo) {
        case DenoiseAlgorithm::BILATERAL:
            success = bilateralFilter(d_input_buffer_, d_output_buffer_, width, height, params, stream);
            break;
        case DenoiseAlgorithm::NON_LOCAL_MEANS:
            success = nonLocalMeans(d_input_buffer_, d_output_buffer_, width, height, params, stream);
            break;
        case DenoiseAlgorithm::GAUSSIAN:
            success = gaussianFilter(d_input_buffer_, d_output_buffer_, width, height, params, stream);
            break;
        case DenoiseAlgorithm::ADAPTIVE_BILATERAL:
            success = adaptiveBilateral(d_input_buffer_, d_output_buffer_, width, height, params, stream);
            break;
        default:
            std::cerr << "Error: Unknown denoising algorithm" << std::endl;
            return false;
    }
    
    if (!success) {
        std::cerr << "Error: Denoising algorithm failed" << std::endl;
        return false;
    }
    
    // Download result from GPU
    if (!downloadImage(d_output_buffer_, output, width, height)) {
        std::cerr << "Error: Failed to download image from GPU" << std::endl;
        return false;
    }
    
    // End timing
    cudaEventRecord(stop_event_, stream);
    cudaEventSynchronize(stop_event_);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    last_processing_time_ms_ = static_cast<double>(elapsed_ms);
    processing_times_.push_back(last_processing_time_ms_);
    
    // Keep only last 100 measurements for statistics
    if (processing_times_.size() > 100) {
        processing_times_.erase(processing_times_.begin());
    }
    
    return true;
}

bool CudaDenoiser::denoiseBatch(const std::vector<cv::Mat>& inputs, 
                               std::vector<cv::Mat>& outputs,
                               DenoiseAlgorithm algo, const DenoiseParams& params) {
    if (inputs.empty()) {
        return false;
    }
    
    outputs.resize(inputs.size());
    
    // Process each image
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!denoiseAsync(inputs[i], outputs[i], algo, params, stream1_)) {
            std::cerr << "Error: Failed to process image " << i << std::endl;
            return false;
        }
    }
    
    // Synchronize to ensure all processing is complete
    cudaStreamSynchronize(stream1_);
    
    return true;
}

bool CudaDenoiser::allocateBuffers(int width, int height) {
    // Check if we need to reallocate
    if (width <= allocated_width_ && height <= allocated_height_ && d_input_buffer_) {
        return true; // Already allocated with sufficient size
    }
    
    // Free existing buffers
    freeBuffers();
    
    size_t float_buffer_size = width * height * sizeof(float);
    size_t uchar_buffer_size = width * height * sizeof(unsigned char);
    
    // Allocate float buffers
    auto& memory_pool = GpuMemoryPool::getInstance();
    
    d_input_buffer_ = static_cast<float*>(memory_pool.allocate(float_buffer_size));
    d_output_buffer_ = static_cast<float*>(memory_pool.allocate(float_buffer_size));
    d_temp_buffer_ = static_cast<float*>(memory_pool.allocate(float_buffer_size));
    
    // Allocate unsigned char buffers for format conversion
    d_input_uchar_ = static_cast<unsigned char*>(memory_pool.allocate(uchar_buffer_size));
    d_output_uchar_ = static_cast<unsigned char*>(memory_pool.allocate(uchar_buffer_size));
    
    if (!d_input_buffer_ || !d_output_buffer_ || !d_temp_buffer_ || 
        !d_input_uchar_ || !d_output_uchar_) {
        std::cerr << "Error: Failed to allocate GPU memory" << std::endl;
        freeBuffers();
        return false;
    }
    
    allocated_width_ = width;
    allocated_height_ = height;
    buffer_size_ = float_buffer_size;
    
    return true;
}

void CudaDenoiser::freeBuffers() {
    auto& memory_pool = GpuMemoryPool::getInstance();
    
    if (d_input_buffer_) {
        memory_pool.deallocate(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    if (d_output_buffer_) {
        memory_pool.deallocate(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
    if (d_temp_buffer_) {
        memory_pool.deallocate(d_temp_buffer_);
        d_temp_buffer_ = nullptr;
    }
    if (d_input_uchar_) {
        memory_pool.deallocate(d_input_uchar_);
        d_input_uchar_ = nullptr;
    }
    if (d_output_uchar_) {
        memory_pool.deallocate(d_output_uchar_);
        d_output_uchar_ = nullptr;
    }
    
    allocated_width_ = 0;
    allocated_height_ = 0;
    buffer_size_ = 0;
}

size_t CudaDenoiser::getMemoryUsage() const {
    return buffer_size_ * 5; // 5 buffers total
}

void CudaDenoiser::resetPerformanceCounters() {
    processing_times_.clear();
    last_processing_time_ms_ = 0.0;
}

void CudaDenoiser::printPerformanceStats() const {
    if (processing_times_.empty()) {
        std::cout << "No performance data available" << std::endl;
        return;
    }
    
    double avg_time = std::accumulate(processing_times_.begin(), processing_times_.end(), 0.0) 
                     / processing_times_.size();
    double min_time = *std::min_element(processing_times_.begin(), processing_times_.end());
    double max_time = *std::max_element(processing_times_.begin(), processing_times_.end());
    
    std::cout << "Performance Statistics:" << std::endl;
    std::cout << "  Frames processed: " << processing_times_.size() << std::endl;
    std::cout << "  Average time: " << avg_time << " ms" << std::endl;
    std::cout << "  Min time: " << min_time << " ms" << std::endl;
    std::cout << "  Max time: " << max_time << " ms" << std::endl;
    std::cout << "  Average FPS: " << (1000.0 / avg_time) << std::endl;
    std::cout << "  Memory usage: " << (getMemoryUsage() / 1024 / 1024) << " MB" << std::endl;
}

bool CudaDenoiser::bilateralFilter(float* d_input, float* d_output, int width, int height,
                                  const DenoiseParams& params, cudaStream_t stream) {
    int kernel_radius = params.kernel_size / 2;
    return launchBilateralFilterShared(d_input, d_output, width, height,
                                      params.sigma_color, params.sigma_space, 
                                      kernel_radius, stream);
}

bool CudaDenoiser::nonLocalMeans(float* d_input, float* d_output, int width, int height,
                                const DenoiseParams& params, cudaStream_t stream) {
    int template_radius = params.template_window_size / 2;
    int search_radius = params.search_window_size / 2;
    
    // Use fast NLM for real-time performance
    return launchFastNonLocalMeans(d_input, d_output, width, height,
                                  params.h, template_radius, search_radius, stream);
}

bool CudaDenoiser::gaussianFilter(float* d_input, float* d_output, int width, int height,
                                 const DenoiseParams& params, cudaStream_t stream) {
    int radius = params.kernel_size / 2;
    
    // Use separable Gaussian for better performance
    return launchSeparableGaussianFilter(d_input, d_temp_buffer_, d_output,
                                        width, height, params.gaussian_sigma, 
                                        radius, stream);
}

bool CudaDenoiser::adaptiveBilateral(float* d_input, float* d_output, int width, int height,
                                    const DenoiseParams& params, cudaStream_t stream) {
    int kernel_radius = params.kernel_size / 2;
    return launchAdaptiveBilateralFilter(d_input, d_output, width, height,
                                       params.sigma_color, params.sigma_space,
                                       kernel_radius, stream);
}

bool CudaDenoiser::uploadImage(const cv::Mat& image, float* d_buffer) {
    cv::Mat float_image;
    
    // Convert to float if necessary
    if (image.type() != CV_32F) {
        image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    } else {
        float_image = image;
    }
    
    // Convert to grayscale if necessary
    if (float_image.channels() > 1) {
        cv::cvtColor(float_image, float_image, cv::COLOR_BGR2GRAY);
    }
    
    // Copy to GPU
    cudaError_t result = cudaMemcpy(d_buffer, float_image.ptr<float>(), 
                                   float_image.total() * sizeof(float), 
                                   cudaMemcpyHostToDevice);
    
    return checkCudaError("Image upload");
}

bool CudaDenoiser::downloadImage(float* d_buffer, cv::Mat& image, int width, int height) {
    // Create output image
    image.create(height, width, CV_32F);
    
    // Copy from GPU
    cudaError_t result = cudaMemcpy(image.ptr<float>(), d_buffer,
                                   width * height * sizeof(float),
                                   cudaMemcpyDeviceToHost);
    
    if (!checkCudaError("Image download")) {
        return false;
    }
    
    // Convert back to 8-bit if needed
    cv::Mat output_8bit;
    image.convertTo(output_8bit, CV_8U, 255.0);
    image = output_8bit;
    
    return true;
}

bool CudaDenoiser::checkCudaError(const char* operation) const {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

void CudaDenoiser::initializeCudaEvents() {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

void CudaDenoiser::cleanupCudaEvents() {
    if (start_event_) {
        cudaEventDestroy(start_event_);
        start_event_ = 0;
    }
    if (stop_event_) {
        cudaEventDestroy(stop_event_);
        stop_event_ = 0;
    }
}

// Utility functions implementation
bool initializeCuda() {
    // Get device count
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    
    // Set device
    cudaSetDevice(0);
    
    // Initialize memory pool
    auto& memory_pool = GpuMemoryPool::getInstance();
    size_t total_mem, free_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Reserve 70% of available memory for the pool
    size_t pool_size = static_cast<size_t>(free_mem * 0.7);
    memory_pool.reserve(pool_size);
    
    std::cout << "GPU Memory Pool initialized with " << (pool_size / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

void cleanupCuda() {
    // Clear memory pool
    auto& memory_pool = GpuMemoryPool::getInstance();
    memory_pool.clear();
    
    // Reset CUDA device
    cudaDeviceReset();
    
    std::cout << "CUDA cleanup completed" << std::endl;
}

int getOptimalBlockSize(int problem_size) {
    // Common block sizes for different problem sizes
    if (problem_size <= 32) return 32;
    if (problem_size <= 64) return 64;
    if (problem_size <= 128) return 128;
    if (problem_size <= 256) return 256;
    return 512;
}

dim3 getOptimalGridSize(int width, int height, int block_size) {
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);
    
    // Ensure grid dimensions don't exceed device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    grid_dim.x = std::min(grid_dim.x, static_cast<unsigned int>(prop.maxGridSize[0]));
    grid_dim.y = std::min(grid_dim.y, static_cast<unsigned int>(prop.maxGridSize[1]));
    
    return grid_dim;
}