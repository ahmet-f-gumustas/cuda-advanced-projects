#include "common/cuda_utils.h"
#include "common/logger.h"
#include "common/timer.h"
#include <cudnn.h>
#include <curand.h>
#include <memory>
#include <vector>
#include <algorithm>

// cuDNN MBConv (Mobile Inverted Bottleneck) demo implementasyonu
class CudnnMBConvBenchmark {
public:
    CudnnMBConvBenchmark(int batch_size, int height, int width)
        : batch_size_(batch_size), height_(height), width_(width) {
        
        CUDNN_CHECK(cudnnCreate(&cudnn_));
        
        // Tensor descriptors
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&depthwise_output_desc_));
        
        // Filter descriptors
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&depthwise_filter_desc_));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&pointwise_filter_desc_));
        
        // Convolution descriptors
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&depthwise_conv_desc_));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&pointwise_conv_desc_));
        
        // Activation descriptor
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc_));
    }
    
    ~CudnnMBConvBenchmark() {
        // Cleanup
        cudnnDestroyTensorDescriptor(input_desc_);
        cudnnDestroyTensorDescriptor(output_desc_);
        cudnnDestroyTensorDescriptor(depthwise_output_desc_);
        cudnnDestroyFilterDescriptor(depthwise_filter_desc_);
        cudnnDestroyFilterDescriptor(pointwise_filter_desc_);
        cudnnDestroyConvolutionDescriptor(depthwise_conv_desc_);
        cudnnDestroyConvolutionDescriptor(pointwise_conv_desc_);
        cudnnDestroyActivationDescriptor(activation_desc_);
        cudnnDestroy(cudnn_);
        
        // Free memory
        if (d_input_) cudaFree(d_input_);
        if (d_depthwise_output_) cudaFree(d_depthwise_output_);
        if (d_output_) cudaFree(d_output_);
        if (d_depthwise_filter_) cudaFree(d_depthwise_filter_);
        if (d_pointwise_filter_) cudaFree(d_pointwise_filter_);
        if (d_workspace_) cudaFree(d_workspace_);
    }
    
    void setup_mbconv(int in_channels, int out_channels, int expand_ratio = 6) {
        int expanded_channels = in_channels * expand_ratio;
        
        // Input tensor (NCHW)
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size_, in_channels, height_, width_));
        
        // Depthwise output tensor
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            depthwise_output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size_, expanded_channels, height_, width_));
        
        // Final output tensor
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size_, out_channels, height_, width_));
        
        // Depthwise filter (groups = channels)
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            depthwise_filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            expanded_channels, 1, 3, 3));  // 3x3 depthwise
        
        // Pointwise filter
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            pointwise_filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            out_channels, expanded_channels, 1, 1));  // 1x1 pointwise
        
        // Depthwise convolution (groups = channels)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            depthwise_conv_desc_, 1, 1, 1, 1, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(depthwise_conv_desc_, expanded_channels));
        
        // Pointwise convolution
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            pointwise_conv_desc_, 0, 0, 1, 1, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        
        // ReLU6 activation
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            activation_desc_, CUDNN_ACTIVATION_CLIPPED_RELU,
            CUDNN_NOT_PROPAGATE_NAN, 6.0));  // ReLU6
        
        // Allocate memory
        size_t input_size = batch_size_ * in_channels * height_ * width_ * sizeof(float);
        size_t depthwise_output_size = batch_size_ * expanded_channels * height_ * width_ * sizeof(float);
        size_t output_size = batch_size_ * out_channels * height_ * width_ * sizeof(float);
        size_t depthwise_filter_size = expanded_channels * 1 * 3 * 3 * sizeof(float);
        size_t pointwise_filter_size = out_channels * expanded_channels * 1 * 1 * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_input_, input_size));
        CUDA_CHECK(cudaMalloc(&d_depthwise_output_, depthwise_output_size));
        CUDA_CHECK(cudaMalloc(&d_output_, output_size));
        CUDA_CHECK(cudaMalloc(&d_depthwise_filter_, depthwise_filter_size));
        CUDA_CHECK(cudaMalloc(&d_pointwise_filter_, pointwise_filter_size));
        
        // Initialize with random values
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, (float*)d_input_, input_size / sizeof(float));
        curandGenerateUniform(gen, (float*)d_depthwise_filter_, depthwise_filter_size / sizeof(float));
        curandGenerateUniform(gen, (float*)d_pointwise_filter_, pointwise_filter_size / sizeof(float));
        curandDestroyGenerator(gen);
        
        // Find best algorithms
        find_best_algorithms();
    }
    
    void benchmark(int warmup_runs = 10, int timing_runs = 100) {
        const float alpha = 1.0f, beta = 0.0f;
        
        // Warmup
        for (int i = 0; i < warmup_runs; ++i) {
            // Depthwise conv + ReLU6
            CUDNN_CHECK(cudnnConvolutionForward(
                cudnn_, &alpha, input_desc_, d_input_,
                depthwise_filter_desc_, d_depthwise_filter_,
                depthwise_conv_desc_, depthwise_algo_, d_workspace_, workspace_size_,
                &beta, depthwise_output_desc_, d_depthwise_output_));
            
            CUDNN_CHECK(cudnnActivationForward(
                cudnn_, activation_desc_, &alpha,
                depthwise_output_desc_, d_depthwise_output_,
                &beta, depthwise_output_desc_, d_depthwise_output_));
            
            // Pointwise conv
            CUDNN_CHECK(cudnnConvolutionForward(
                cudnn_, &alpha, depthwise_output_desc_, d_depthwise_output_,
                pointwise_filter_desc_, d_pointwise_filter_,
                pointwise_conv_desc_, pointwise_algo_, d_workspace_, workspace_size_,
                &beta, output_desc_, d_output_));
        }
        
        // Timing
        CudaTimer timer;
        for (int i = 0; i < timing_runs; ++i) {
            timer.start();
            
            // MBConv block
            CUDNN_CHECK(cudnnConvolutionForward(
                cudnn_, &alpha, input_desc_, d_input_,
                depthwise_filter_desc_, d_depthwise_filter_,
                depthwise_conv_desc_, depthwise_algo_, d_workspace_, workspace_size_,
                &beta, depthwise_output_desc_, d_depthwise_output_));
            
            CUDNN_CHECK(cudnnActivationForward(
                cudnn_, activation_desc_, &alpha,
                depthwise_output_desc_, d_depthwise_output_,
                &beta, depthwise_output_desc_, d_depthwise_output_));
            
            CUDNN_CHECK(cudnnConvolutionForward(
                cudnn_, &alpha, depthwise_output_desc_, d_depthwise_output_,
                pointwise_filter_desc_, d_pointwise_filter_,
                pointwise_conv_desc_, pointwise_algo_, d_workspace_, workspace_size_,
                &beta, output_desc_, d_output_));
            
            timer.stop();
        }
        
        LOG_INFO("MBConv benchmark results:");
        LOG_INFO("  Mean latency:", timer.get_mean(), "ms");
        LOG_INFO("  P50 latency:", timer.get_percentile(0.5), "ms");
        LOG_INFO("  P95 latency:", timer.get_percentile(0.95), "ms");
    }
    
private:
    void find_best_algorithms() {
        const float alpha = 1.0f, beta = 0.0f;
        
        // Find best depthwise algorithm
        int requested_algo_count, returned_algo_count;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_, &requested_algo_count));
        
        std::vector<cudnnConvolutionFwdAlgoPerf_t> depthwise_results(requested_algo_count);
        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
            cudnn_, input_desc_, depthwise_filter_desc_, depthwise_conv_desc_,
            depthwise_output_desc_, requested_algo_count, &returned_algo_count,
            depthwise_results.data()));
        
        depthwise_algo_ = depthwise_results[0].algo;
        
        // Find best pointwise algorithm
        std::vector<cudnnConvolutionFwdAlgoPerf_t> pointwise_results(requested_algo_count);
        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
            cudnn_, depthwise_output_desc_, pointwise_filter_desc_, pointwise_conv_desc_,
            output_desc_, requested_algo_count, &returned_algo_count,
            pointwise_results.data()));
        
        pointwise_algo_ = pointwise_results[0].algo;
        
        // Get workspace size
        size_t depthwise_workspace, pointwise_workspace;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_, input_desc_, depthwise_filter_desc_, depthwise_conv_desc_,
            depthwise_output_desc_, depthwise_algo_, &depthwise_workspace));
        
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_, depthwise_output_desc_, pointwise_filter_desc_, pointwise_conv_desc_,
            output_desc_, pointwise_algo_, &pointwise_workspace));
        
        workspace_size_ = std::max(depthwise_workspace, pointwise_workspace);
        if (workspace_size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_size_));
        }
    }
    
    cudnnHandle_t cudnn_;
    
    // Tensor descriptors
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnTensorDescriptor_t depthwise_output_desc_;
    
    // Filter descriptors
    cudnnFilterDescriptor_t depthwise_filter_desc_;
    cudnnFilterDescriptor_t pointwise_filter_desc_;
    
    // Convolution descriptors
    cudnnConvolutionDescriptor_t depthwise_conv_desc_;
    cudnnConvolutionDescriptor_t pointwise_conv_desc_;
    
    // Activation descriptor
    cudnnActivationDescriptor_t activation_desc_;
    
    // Algorithms
    cudnnConvolutionFwdAlgo_t depthwise_algo_;
    cudnnConvolutionFwdAlgo_t pointwise_algo_;
    
    // Dimensions
    int batch_size_;
    int height_;
    int width_;
    
    // Device memory
    void* d_input_ = nullptr;
    void* d_depthwise_output_ = nullptr;
    void* d_output_ = nullptr;
    void* d_depthwise_filter_ = nullptr;
    void* d_pointwise_filter_ = nullptr;
    void* d_workspace_ = nullptr;
    size_t workspace_size_ = 0;
};

// Export function for benchmark app
extern "C" void run_cudnn_mbconv_benchmark(int batch_size, int height, int width,
                                          int in_channels, int out_channels,
                                          int warmup, int runs) {
    CudnnMBConvBenchmark benchmark(batch_size, height, width);
    benchmark.setup_mbconv(in_channels, out_channels);
    benchmark.benchmark(warmup, runs);
}