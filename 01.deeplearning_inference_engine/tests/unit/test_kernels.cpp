#include <gtest/gtest.h>
#include "kernels/conv_kernels.cuh"
#include "kernels/activation_kernels.cuh"
#include "kernels/gemm_kernels.cuh"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>

using namespace deep_engine;
using namespace deep_engine::kernels;

class KernelTest : public ::testing::Test {
protected:
    cudaStream_t stream_;
    
    void SetUp() override {
        cudaSetDevice(0);
        cudaStreamCreate(&stream_);
    }
    
    void TearDown() override {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
};

// Activation kernel tests
TEST_F(KernelTest, ReLUKernel) {
    const int size = 1000;
    std::vector<float> host_input(size);
    
    // Generate test data with positive and negative values
    for (int i = 0; i < size; ++i) {
        host_input[i] = i - size/2.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, host_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_relu(d_input, d_output, size, stream_);
    
    // Copy result back
    std::vector<float> host_output(size);
    cudaMemcpy(host_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < size; ++i) {
        float expected = std::max(0.0f, host_input[i]);
        EXPECT_FLOAT_EQ(host_output[i], expected);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(KernelTest, LeakyReLUKernel) {
    const int size = 1000;
    const float negative_slope = 0.1f;
    
    std::vector<float> host_input(size);
    for (int i = 0; i < size; ++i) {
        host_input[i] = i - size/2.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, host_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_leaky_relu(d_input, d_output, negative_slope, size, stream_);
    
    std::vector<float> host_output(size);
    cudaMemcpy(host_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i) {
        float expected = host_input[i] > 0 ? host_input[i] : host_input[i] * negative_slope;
        EXPECT_FLOAT_EQ(host_output[i], expected);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(KernelTest, SigmoidKernel) {
    const int size = 100;
    std::vector<float> test_values = {-5.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, test_values.size() * sizeof(float));
    cudaMalloc(&d_output, test_values.size() * sizeof(float));
    
    cudaMemcpy(d_input, test_values.data(), test_values.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_sigmoid(d_input, d_output, test_values.size(), stream_);
    
    std::vector<float> host_output(test_values.size());
    cudaMemcpy(host_output.data(), d_output, test_values.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < test_values.size(); ++i) {
        float expected = 1.0f / (1.0f + std::exp(-test_values[i]));
        EXPECT_NEAR(host_output[i], expected, 1e-5f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(KernelTest, SoftmaxKernel) {
    const int batch_size = 2;
    const int classes = 5;
    
    std::vector<float> host_input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // Batch 0
        5.0f, 4.0f, 3.0f, 2.0f, 1.0f   // Batch 1
    };
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * classes * sizeof(float));
    cudaMalloc(&d_output, batch_size * classes * sizeof(float));
    
    cudaMemcpy(d_input, host_input.data(), batch_size * classes * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_softmax(d_input, d_output, batch_size, classes, stream_);
    
    std::vector<float> host_output(batch_size * classes);
    cudaMemcpy(host_output.data(), d_output, batch_size * classes * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify each batch sums to 1
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int c = 0; c < classes; ++c) {
            sum += host_output[b * classes + c];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
    
    // Verify relative ordering is preserved
    EXPECT_LT(host_output[0], host_output[1]);  // exp(1) < exp(2)
    EXPECT_GT(host_output[5], host_output[6]);  // exp(5) > exp(4)
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Im2col kernel test
TEST_F(KernelTest, Im2ColKernel) {
    const int batch_size = 1;
    const int channels = 2;
    const int height = 4;
    const int width = 4;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int pad_h = 1;
    const int pad_w = 1;
    const int stride_h = 1;
    const int stride_w = 1;
    const int dilation_h = 1;
    const int dilation_w = 1;
    
    const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Create simple test pattern
    std::vector<float> host_input(batch_size * channels * height * width);
    for (int i = 0; i < host_input.size(); ++i) {
        host_input[i] = i;
    }
    
    float *d_input, *d_output;
    const int output_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;
    
    cudaMalloc(&d_input, host_input.size() * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cudaMemcpy(d_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_im2col(d_input, d_output, batch_size, channels, height, width,
                  kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                  dilation_h, dilation_w, output_h, output_w, stream_);
    
    std::vector<float> host_output(output_size);
    cudaMemcpy(host_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Basic sanity check - output should have expected size
    EXPECT_EQ(host_output.size(), output_size);
    
    // Check that padding produces zeros
    // First element should be 0 (top-left padding)
    EXPECT_FLOAT_EQ(host_output[0], 0.0f);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Bias ReLU fusion kernel test
TEST_F(KernelTest, BiasReLUKernel) {
    const int batch = 2;
    const int channels = 3;
    const int spatial_size = 4;
    const int total_size = batch * channels * spatial_size;
    
    std::vector<float> host_input(total_size);
    std::vector<float> host_bias(channels);
    
    // Initialize input and bias
    for (int i = 0; i < total_size; ++i) {
        host_input[i] = i - total_size/2.0f;  // Mix of positive and negative
    }
    for (int i = 0; i < channels; ++i) {
        host_bias[i] = i * 0.5f;
    }
    
    float *d_input, *d_bias, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_bias, channels * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    
    cudaMemcpy(d_input, host_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, host_bias.data(), channels * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_bias_relu(d_input, d_bias, d_output, batch, channels, spatial_size, stream_);
    
    std::vector<float> host_output(total_size);
    cudaMemcpy(host_output.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify bias addition and ReLU
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int s = 0; s < spatial_size; ++s) {
                int idx = b * channels * spatial_size + c * spatial_size + s;
                float expected = std::max(0.0f, host_input[idx] + host_bias[c]);
                EXPECT_FLOAT_EQ(host_output[idx], expected);
            }
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}

// GEMM kernel test
TEST_F(KernelTest, GEMMWrapper) {
    CublasGemmWrapper gemm_wrapper;
    
    const int M = 128;
    const int N = 256;
    const int K = 512;
    
    // Create test matrices
    std::vector<float> host_A(M * K);
    std::vector<float> host_B(K * N);
    std::vector<float> host_C(M * N, 0.0f);
    
    // Initialize with simple values
    for (int i = 0; i < M * K; ++i) {
        host_A[i] = (i % 10) * 0.1f;
    }
    for (int i = 0; i < K * N; ++i) {
        host_B[i] = (i % 5) * 0.2f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, host_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, host_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform GEMM: C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;
    gemm_wrapper.gemm(d_A, d_B, d_C, M, N, K, alpha, beta, false, false, stream_);
    
    std::vector<float> host_result(M * N);
    cudaMemcpy(host_result.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Basic check - result should not be all zeros
    float sum = 0.0f;
    for (float val : host_result) {
        sum += std::abs(val);
    }
    EXPECT_GT(sum, 0.0f);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Performance test (disabled by default)
TEST_F(KernelTest, DISABLED_KernelPerformance) {
    const int size = 10 * 1024 * 1024;  // 10M elements
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_relu(d_input, d_output, size, stream_);
    }
    cudaStreamSynchronize(stream_);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    
    for (int i = 0; i < iterations; ++i) {
        launch_relu(d_input, d_output, size, stream_);
    }
    cudaStreamSynchronize(stream_);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float bandwidth = (size * sizeof(float) * 2 * iterations) / (duration.count() / 1e6) / 1e9;  // GB/s
    std::cout << "ReLU kernel bandwidth: " << bandwidth << " GB/s" << std::endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
}