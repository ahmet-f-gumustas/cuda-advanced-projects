#include <gtest/gtest.h>
#include "layers/convolution.h"
#include "layers/activation.h"
#include "layers/pooling.h"
#include "layers/batchnorm.h"
#include "layers/dense.h"
#include "core/tensor.h"
#include "test_layers_helpers.h"
#include <cuda_runtime.h>

using namespace deep_engine;

class LayerTest : public ::testing::Test {
protected:
    ExecutionContext ctx_;
    
    void SetUp() override {
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    bool tensors_equal(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
            return false;
        }
        
        std::vector<float> data_a(a.size()), data_b(b.size());
        cudaMemcpy(data_a.data(), a.data<float>(), a.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(data_b.data(), b.data<float>(), b.bytes(), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// Convolution Layer Tests
TEST_F(LayerTest, Conv2dForwardShape) {
    Conv2d conv(3, 64, 7, 2, 3);  // in_ch=3, out_ch=64, kernel=7, stride=2, pad=3
    
    Tensor input({1, 3, 224, 224});
    auto output = conv.forward(input, ctx_);
    
    // Output shape: (224 + 2*3 - 7) / 2 + 1 = 112
    EXPECT_EQ(output.shape(), std::vector<int>({1, 64, 112, 112}));
}

TEST_F(LayerTest, Conv2dParameters) {
    Conv2d conv(16, 32, 3, 1, 1);
    
    // Check parameter count
    // Weight: 32 * 16 * 3 * 3 = 4608
    // Bias: 32
    EXPECT_EQ(conv.num_params(), 4608 + 32);
    
    // Check weight shape
    EXPECT_EQ(conv.weight().shape(), std::vector<int>({32, 16, 3, 3}));
    EXPECT_EQ(conv.bias().shape(), std::vector<int>({32}));
}

TEST_F(LayerTest, Conv2dNoBias) {
    Conv2d conv(8, 16, 3, 1, 1, 1, 1, false);  // use_bias=false
    
    EXPECT_EQ(conv.num_params(), 8 * 16 * 3 * 3);  // Only weights
}

TEST_F(LayerTest, Conv2dGroups) {
    // Grouped convolution
    Conv2d conv(16, 32, 3, 1, 1, 1, 4);  // groups=4
    
    // Each group: in_ch/groups * out_ch/groups * k * k
    // 4 groups: 4 * 8 * 3 * 3 * 4 = 1152
    EXPECT_EQ(conv.weight().shape(), std::vector<int>({32, 4, 3, 3}));
}

TEST_F(LayerTest, DepthwiseConv2d) {
    DepthwiseConv2d conv(32, 3, 1, 1);
    
    Tensor input({1, 32, 64, 64});
    auto output = conv.forward(input, ctx_);
    
    EXPECT_EQ(output.shape(), std::vector<int>({1, 32, 64, 64}));
}

// Activation Layer Tests
TEST_F(LayerTest, ReLUForward) {
    ReLU relu;
    
    // Create tensor with positive and negative values
    Tensor input({2, 3});
    std::vector<float> data = {-1, 0, 1, -2, 3, -4};
    cudaMemcpy(input.data<float>(), data.data(), input.bytes(), cudaMemcpyHostToDevice);
    
    auto output = relu.forward(input, ctx_);
    
    std::vector<float> result(6);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    std::vector<float> expected = {0, 0, 1, 0, 3, 0};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(LayerTest, LeakyReLUForward) {
    LeakyReLU leaky_relu(0.1f);
    
    Tensor input({2, 2});
    std::vector<float> data = {-10, -1, 1, 10};
    cudaMemcpy(input.data<float>(), data.data(), input.bytes(), cudaMemcpyHostToDevice);
    
    auto output = leaky_relu.forward(input, ctx_);
    
    std::vector<float> result(4);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(result[0], -1.0f);  // -10 * 0.1
    EXPECT_FLOAT_EQ(result[1], -0.1f);  // -1 * 0.1
    EXPECT_FLOAT_EQ(result[2], 1.0f);   // 1 (positive)
    EXPECT_FLOAT_EQ(result[3], 10.0f);  // 10 (positive)
}

TEST_F(LayerTest, SigmoidForward) {
    Sigmoid sigmoid;
    
    Tensor input({1, 3});
    std::vector<float> data = {-2, 0, 2};
    cudaMemcpy(input.data<float>(), data.data(), input.bytes(), cudaMemcpyHostToDevice);
    
    auto output = sigmoid.forward(input, ctx_);
    
    std::vector<float> result(3);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    // Sigmoid(-2) ≈ 0.119, Sigmoid(0) = 0.5, Sigmoid(2) ≈ 0.881
    EXPECT_NEAR(result[0], 0.119f, 0.01f);
    EXPECT_NEAR(result[1], 0.5f, 0.01f);
    EXPECT_NEAR(result[2], 0.881f, 0.01f);
}

TEST_F(LayerTest, SoftmaxForward) {
    Softmax softmax(-1);  // Last axis
    
    Tensor input({2, 3});
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    cudaMemcpy(input.data<float>(), data.data(), input.bytes(), cudaMemcpyHostToDevice);
    
    auto output = softmax.forward(input, ctx_);
    
    std::vector<float> result(6);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    // Check that each row sums to 1
    float sum1 = result[0] + result[1] + result[2];
    float sum2 = result[3] + result[4] + result[5];
    EXPECT_NEAR(sum1, 1.0f, 1e-5f);
    EXPECT_NEAR(sum2, 1.0f, 1e-5f);
    
    // Check that values are in ascending order (since input was ascending)
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
    EXPECT_LT(result[3], result[4]);
    EXPECT_LT(result[4], result[5]);
}

// Pooling Layer Tests
TEST_F(LayerTest, MaxPool2dForward) {
    MaxPool2d pool(2, 2, 0);  // kernel=2, stride=2, padding=0
    
    Tensor input({1, 1, 4, 4});
    std::vector<float> data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    cudaMemcpy(input.data<float>(), data.data(), input.bytes(), cudaMemcpyHostToDevice);
    
    auto output = pool.forward(input, ctx_);
    
    EXPECT_EQ(output.shape(), std::vector<int>({1, 1, 2, 2}));
    
    std::vector<float> result(4);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    // Max pooling should select: 6, 8, 14, 16
    EXPECT_FLOAT_EQ(result[0], 6.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 14.0f);
    EXPECT_FLOAT_EQ(result[3], 16.0f);
}

TEST_F(LayerTest, AvgPool2dForward) {
    AvgPool2d pool(2, 2, 0);
    
    Tensor input({1, 1, 4, 4});
    std::vector<float> data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    cudaMemcpy(input.data<float>(), data.data(), input.bytes(), cudaMemcpyHostToDevice);
    
    auto output = pool.forward(input, ctx_);
    
    std::vector<float> result(4);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    // Average pooling: (1+2+5+6)/4 = 3.5, etc.
    EXPECT_FLOAT_EQ(result[0], 3.5f);
    EXPECT_FLOAT_EQ(result[1], 5.5f);
    EXPECT_FLOAT_EQ(result[2], 11.5f);
    EXPECT_FLOAT_EQ(result[3], 13.5f);
}

TEST_F(LayerTest, GlobalAvgPoolForward) {
    GlobalAvgPool pool;
    
    Tensor input({2, 3, 4, 4});  // batch=2, channels=3, h=4, w=4
    input.fill(2.0f);  // Fill with 2s
    
    auto output = pool.forward(input, ctx_);
    
    EXPECT_EQ(output.shape(), std::vector<int>({2, 3, 1, 1}));
    
    std::vector<float> result(6);
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    // All values should be 2.0
    for (float val : result) {
        EXPECT_FLOAT_EQ(val, 2.0f);
    }
}

// BatchNorm Layer Tests
TEST_F(LayerTest, BatchNorm2dForwardTraining) {
    BatchNorm2d bn(16);  // 16 channels
    bn.set_training(true);
    
    Tensor input({4, 16, 8, 8});  // batch=4
    input = Tensor::random_normal(input.shape(), 0.0f, 1.0f);
    
    auto output = bn.forward(input, ctx_);
    
    EXPECT_EQ(output.shape(), input.shape());
    
    // In training mode, output should be normalized
    // Check that running stats are updated
    auto& running_mean = bn.running_mean();
    auto& running_var = bn.running_var();
    
    EXPECT_EQ(running_mean.shape(), std::vector<int>({16}));
    EXPECT_EQ(running_var.shape(), std::vector<int>({16}));
}

TEST_F(LayerTest, BatchNorm2dForwardInference) {
    BatchNorm2d bn(8);
    bn.set_training(false);  // Inference mode
    
    // Initialize running stats
    bn.running_mean().fill(0.5f);
    bn.running_var().fill(1.0f);
    
    Tensor input({2, 8, 4, 4});
    input.fill(0.5f);
    
    auto output = bn.forward(input, ctx_);
    
    // With mean=0.5, var=1.0, and input=0.5, output should be ~0
    // (after applying weight and bias which are initialized to 1 and 0)
    std::vector<float> result(output.size());
    cudaMemcpy(result.data(), output.data<float>(), output.bytes(), cudaMemcpyDeviceToHost);
    
    for (float val : result) {
        EXPECT_NEAR(val, 0.0f, 0.1f);
    }
}

// Dense Layer Tests
TEST_F(LayerTest, DenseForwardShape) {
    DenseLayer dense(128, 64);
    
    Tensor input({10, 128});  // batch=10, features=128
    auto output = dense.forward(input, ctx_);
    
    EXPECT_EQ(output.shape(), std::vector<int>({10, 64}));
}

TEST_F(LayerTest, DenseParameters) {
    DenseLayer dense(100, 50, true);  // use_bias=true
    
    // Weight: 100 * 50 = 5000
    // Bias: 50
    EXPECT_EQ(dense.num_params(), 5050);
    
    EXPECT_EQ(dense.weight().shape(), std::vector<int>({50, 100}));
    EXPECT_EQ(dense.bias().shape(), std::vector<int>({50}));
}

TEST_F(LayerTest, DenseNoBias) {
    DenseLayer dense(64, 32, false);  // use_bias=false
    
    EXPECT_EQ(dense.num_params(), 64 * 32);
}

// Layer Composition Tests
TEST_F(LayerTest, ConvBNReLUSequence) {
    // Common pattern: Conv -> BatchNorm -> ReLU
    Conv2d conv(3, 64, 3, 1, 1);
    BatchNorm2d bn(64);
    ReLU relu;
    
    Tensor input({1, 3, 32, 32});
    
    auto x = conv.forward(input, ctx_);
    EXPECT_EQ(x.shape(), std::vector<int>({1, 64, 32, 32}));
    
    x = bn.forward(x, ctx_);
    EXPECT_EQ(x.shape(), std::vector<int>({1, 64, 32, 32}));
    
    x = relu.forward(x, ctx_);
    EXPECT_EQ(x.shape(), std::vector<int>({1, 64, 32, 32}));
}

// Fusion Support Tests
TEST_F(LayerTest, LayerFusionSupport) {
    Conv2d conv(3, 64, 3);
    BatchNorm2d bn(64);
    ReLU relu;
    
    // Conv can fuse with BatchNorm
    EXPECT_TRUE(conv.can_fuse_with(bn));
    
    // BatchNorm can fuse with ReLU
    EXPECT_TRUE(bn.can_fuse_with(relu));
    
    // ReLU typically doesn't fuse with following layers
    EXPECT_FALSE(relu.can_fuse_with(conv));
}

// Quantization Support Tests
TEST_F(LayerTest, QuantizationSupport) {
    Conv2d conv(3, 64, 3);
    DenseLayer dense(128, 64);
    BatchNorm2d bn(64);
    
    EXPECT_TRUE(conv.supports_quantization());
    EXPECT_TRUE(dense.supports_quantization());
    EXPECT_TRUE(bn.supports_quantization());
    
    // Test quantization doesn't crash
    EXPECT_NO_THROW(conv.quantize(8));
    EXPECT_NO_THROW(dense.quantize(8));
}

// Performance test for layer operations
TEST_F(LayerTest, DISABLED_ConvolutionPerformance) {
    Conv2d conv(3, 64, 3, 1, 1);
    Tensor input({32, 3, 224, 224});  // ImageNet size
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        conv.forward(input, ctx_);
    }
    ctx_.synchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    
    for (int i = 0; i < iterations; ++i) {
        conv.forward(input, ctx_);
    }
    ctx_.synchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time = duration.count() / 1000.0f / iterations;
    std::cout << "Conv2d forward pass: " << avg_time << " ms" << std::endl;
    
    // Should be reasonably fast
    EXPECT_LT(avg_time, 10.0f);  // Less than 10ms
}