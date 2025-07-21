#include <benchmark/benchmark.h>
#include "kernels/conv_kernels.cuh"
#include "kernels/activation_kernels.cuh"
#include "kernels/gemm_kernels.cuh"
#include "kernels/reduction_kernels.cuh"
#include <cuda_runtime.h>
#include <vector>

using namespace deep_engine::kernels;

class KernelBenchmark : public benchmark::Fixture {
public:
    cudaStream_t stream_;
    
    void SetUp(const ::benchmark::State& state) override {
        cudaSetDevice(0);
        cudaStreamCreate(&stream_);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
};

// Activation kernel benchmarks
BENCHMARK_DEFINE_F(KernelBenchmark, ReLUKernel)(benchmark::State& state) {
    int size = state.range(0);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // Initialize with random data
    std::vector<float> h_data(size);
    for (int i = 0; i < size; ++i) {
        h_data[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
    cudaMemcpy(d_input, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    for (auto _ : state) {
        launch_relu(d_input, d_output, size, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, ReLUKernel)
    ->Range(1024, 100*1024*1024)  // 1K to 100M elements
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(KernelBenchmark, GELUKernel)(benchmark::State& state) {
    int size = state.range(0);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    for (auto _ : state) {
        launch_gelu(d_input, d_output, size, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, GELUKernel)
    ->Range(1024, 10*1024*1024)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(KernelBenchmark, SoftmaxKernel)(benchmark::State& state) {
    int batch_size = state.range(0);
    int num_classes = state.range(1);
    int total_size = batch_size * num_classes;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    
    for (auto _ : state) {
        launch_softmax(d_input, d_output, batch_size, num_classes, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, SoftmaxKernel)
    ->Args({1, 1000})      // ImageNet classes
    ->Args({32, 1000})
    ->Args({128, 1000})
    ->Args({1, 50000})     // Large vocabulary
    ->Args({32, 50000})
    ->Unit(benchmark::kMicrosecond);

// Im2col kernel benchmarks
BENCHMARK_DEFINE_F(KernelBenchmark, Im2ColKernel)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int height = state.range(2);
    int width = state.range(2);  // Square images
    int kernel_size = 3;
    int pad = 1;
    int stride = 1;
    int dilation = 1;
    
    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    int output_w = (width + 2 * pad - kernel_size) / stride + 1;
    
    int input_size = batch_size * channels * height * width;
    int output_size = batch_size * channels * kernel_size * kernel_size * output_h * output_w;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    for (auto _ : state) {
        launch_im2col(d_input, d_output, batch_size, channels, height, width,
                      kernel_size, kernel_size, pad, pad, stride, stride,
                      dilation, dilation, output_h, output_w, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetBytesProcessed(state.iterations() * 
        (input_size + output_size) * sizeof(float));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, Im2ColKernel)
    ->Args({1, 64, 56})    // Typical conv layer
    ->Args({8, 64, 56})
    ->Args({32, 64, 56})
    ->Args({1, 256, 14})   // Deeper layer
    ->Args({8, 256, 14})
    ->Unit(benchmark::kMicrosecond);

// Depthwise convolution kernel benchmark
BENCHMARK_DEFINE_F(KernelBenchmark, DepthwiseConv2dKernel)(benchmark::State& state) {
    int batch = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    int kernel_h = 3, kernel_w = 3;
    int pad_h = 1, pad_w = 1;
    int stride_h = 1, stride_w = 1;
    int output_h = size, output_w = size;  // Same padding
    
    int input_size = batch * channels * size * size;
    int filter_size = channels * kernel_h * kernel_w;
    int output_size = batch * channels * output_h * output_w;
    
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_filter, filter_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    for (auto _ : state) {
        launch_depthwise_conv2d(d_input, d_filter, nullptr, d_output,
                               batch, channels, size, size,
                               kernel_h, kernel_w, pad_h, pad_w,
                               stride_h, stride_w, output_h, output_w,
                               false, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetItemsProcessed(state.iterations() * batch);
    
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, DepthwiseConv2dKernel)
    ->Args({1, 32, 112})   // MobileNet sizes
    ->Args({8, 32, 112})
    ->Args({1, 64, 56})
    ->Args({8, 64, 56})
    ->Unit(benchmark::kMicrosecond);

// GEMM kernel benchmarks
BENCHMARK_DEFINE_F(KernelBenchmark, GEMMKernel)(benchmark::State& state) {
    int M = state.range(0);
    int N = state.range(1);
    int K = state.range(2);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    CublasGemmWrapper gemm_wrapper;
    float alpha = 1.0f, beta = 0.0f;
    
    for (auto _ : state) {
        gemm_wrapper.gemm(d_A, d_B, d_C, M, N, K, alpha, beta, 
                         false, false, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    // FLOPS: 2 * M * N * K (multiply-add)
    state.SetItemsProcessed(state.iterations() * 2 * M * N * K);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

BENCHMARK_REGISTER_F(KernelBenchmark, GEMMKernel)
    ->Args({128, 128, 128})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({1024, 768, 768})    // Transformer dimensions
    ->Args({4096, 4096, 1024})  // Large FC layers
    ->Unit(benchmark::kMicrosecond);

// Fused bias ReLU kernel benchmark
BENCHMARK_DEFINE_F(KernelBenchmark, BiasReLUKernel)(benchmark::State& state) {
    int batch = state.range(0);
    int channels = state.range(1);
    int spatial_size = state.range(2) * state.range(2);
    int total_size = batch * channels * spatial_size;
    
    float *d_input, *d_bias, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_bias, channels * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    
    for (auto _ : state) {
        launch_bias_relu(d_input, d_bias, d_output, batch, channels, 
                        spatial_size, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetBytesProcessed(state.iterations() * 
        (total_size * 2 + channels) * sizeof(float));
    
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, BiasReLUKernel)
    ->Args({1, 64, 56})
    ->Args({8, 64, 56})
    ->Args({32, 64, 56})
    ->Args({1, 256, 14})
    ->Args({8, 256, 14})
    ->Unit(benchmark::kMicrosecond);

// Reduction kernel benchmarks
BENCHMARK_DEFINE_F(KernelBenchmark, ReduceSumKernel)(benchmark::State& state) {
    int size = state.range(0);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));  // Single output value
    
    for (auto _ : state) {
        launch_reduce_sum(d_input, d_output, size, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, ReduceSumKernel)
    ->Range(1024, 100*1024*1024)
    ->Unit(benchmark::kMicrosecond);

// INT8 quantized kernel benchmarks
BENCHMARK_DEFINE_F(KernelBenchmark, QuantizedConv2dKernel)(benchmark::State& state) {
    int batch = 1;
    int in_channels = state.range(0);
    int out_channels = state.range(1);
    int size = state.range(2);
    int kernel_size = 3;
    int pad = 1;
    int stride = 1;
    
    int input_size = batch * in_channels * size * size;
    int filter_size = out_channels * in_channels * kernel_size * kernel_size;
    int output_size = batch * out_channels * size * size;
    
    int8_t *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, output_size);
    
    float input_scale = 0.1f, filter_scale = 0.05f, output_scale = 0.15f;
    
    for (auto _ : state) {
        quantized_conv2d_kernel<<<256, 256, 0, stream_>>>(
            d_input, d_filter, nullptr, d_output,
            input_scale, filter_scale, output_scale,
            batch, in_channels, out_channels, size, size,
            kernel_size, kernel_size, pad, pad, stride, stride,
            size, size, false);
        cudaStreamSynchronize(stream_);
    }
    
    state.SetItemsProcessed(state.iterations() * batch);
    
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}

BENCHMARK_REGISTER_F(KernelBenchmark, QuantizedConv2dKernel)
    ->Args({64, 64, 56})
    ->Args({128, 128, 28})
    ->Args({256, 256, 14})
    ->Unit(benchmark::kMicrosecond);

// Main function
BENCHMARK_MAIN();