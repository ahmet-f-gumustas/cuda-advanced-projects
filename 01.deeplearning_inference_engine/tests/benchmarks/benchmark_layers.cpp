#include <benchmark/benchmark.h>
#include "layers/convolution.h"
#include "layers/activation.h"
#include "layers/pooling.h"
#include "layers/batchnorm.h"
#include "layers/dense.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

using namespace deep_engine;

// Helper to ensure CUDA is initialized
class CudaFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        cudaSetDevice(0);
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    void TearDown(const ::benchmark::State& state) override {
        cudaDeviceSynchronize();
    }
    
protected:
    std::unique_ptr<ExecutionContext> ctx_;
};

// Convolution benchmarks
BENCHMARK_DEFINE_F(CudaFixture, Conv2d_3x3)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    Conv2d conv(channels, channels, 3, 1, 1);
    Tensor input({batch_size, channels, size, size});
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        conv.forward(input, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto output = conv.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * input.bytes() * 2);  // Input + output
}

BENCHMARK_REGISTER_F(CudaFixture, Conv2d_3x3)
    ->Args({1, 64, 224})    // ResNet first layer size
    ->Args({8, 64, 224})
    ->Args({32, 64, 224})
    ->Args({1, 128, 112})   // Deeper layers
    ->Args({8, 128, 112})
    ->Args({32, 128, 112})
    ->Args({1, 256, 56})
    ->Args({8, 256, 56})
    ->Args({32, 256, 56})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(CudaFixture, Conv2d_1x1)(benchmark::State& state) {
    int batch_size = state.range(0);
    int in_channels = state.range(1);
    int out_channels = state.range(2);
    int size = state.range(3);
    
    Conv2d conv(in_channels, out_channels, 1, 1, 0);
    Tensor input({batch_size, in_channels, size, size});
    
    for (auto _ : state) {
        auto output = conv.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(CudaFixture, Conv2d_1x1)
    ->Args({1, 256, 512, 56})   // Channel expansion
    ->Args({8, 256, 512, 56})
    ->Args({1, 512, 256, 56})   // Channel reduction
    ->Args({8, 512, 256, 56})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(CudaFixture, DepthwiseConv2d)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    DepthwiseConv2d conv(channels, 3, 1, 1);
    Tensor input({batch_size, channels, size, size});
    
    for (auto _ : state) {
        auto output = conv.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(CudaFixture, DepthwiseConv2d)
    ->Args({1, 32, 112})    // MobileNet sizes
    ->Args({8, 32, 112})
    ->Args({1, 64, 56})
    ->Args({8, 64, 56})
    ->Unit(benchmark::kMicrosecond);

// Activation benchmarks
BENCHMARK_DEFINE_F(CudaFixture, ReLU)(benchmark::State& state) {
    int size = state.range(0);
    
    ReLU relu;
    Tensor input({size});
    
    for (auto _ : state) {
        auto output = relu.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * input.bytes() * 2);
}

BENCHMARK_REGISTER_F(CudaFixture, ReLU)
    ->Range(1024, 1024*1024*10)  // 1K to 10M elements
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(CudaFixture, GELU)(benchmark::State& state) {
    int size = state.range(0);
    
    GELU gelu;
    Tensor input({size});
    
    for (auto _ : state) {
        auto output = gelu.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * input.bytes() * 2);
}

BENCHMARK_REGISTER_F(CudaFixture, GELU)
    ->Range(1024, 1024*1024*10)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(CudaFixture, Softmax)(benchmark::State& state) {
    int batch_size = state.range(0);
    int classes = state.range(1);
    
    Softmax softmax(-1);
    Tensor input({batch_size, classes});
    
    for (auto _ : state) {
        auto output = softmax.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(CudaFixture, Softmax)
    ->Args({1, 1000})     // ImageNet
    ->Args({32, 1000})
    ->Args({128, 1000})
    ->Args({1, 10000})    // Large vocabulary
    ->Args({32, 10000})
    ->Unit(benchmark::kMicrosecond);

// Pooling benchmarks
BENCHMARK_DEFINE_F(CudaFixture, MaxPool2d)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    MaxPool2d pool(2, 2);
    Tensor input({batch_size, channels, size, size});
    
    for (auto _ : state) {
        auto output = pool.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(CudaFixture, MaxPool2d)
    ->Args({1, 64, 224})
    ->Args({8, 64, 224})
    ->Args({32, 64, 224})
    ->Args({1, 256, 56})
    ->Args({8, 256, 56})
    ->Unit(benchmark::kMicrosecond);

// BatchNorm benchmarks
BENCHMARK_DEFINE_F(CudaFixture, BatchNorm2d)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    BatchNorm2d bn(channels);
    bn.set_training(false);  // Inference mode
    Tensor input({batch_size, channels, size, size});
    
    for (auto _ : state) {
        auto output = bn.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(CudaFixture, BatchNorm2d)
    ->Args({1, 64, 224})
    ->Args({8, 64, 224})
    ->Args({32, 64, 224})
    ->Args({1, 256, 56})
    ->Args({8, 256, 56})
    ->Unit(benchmark::kMicrosecond);

// Dense layer benchmarks
BENCHMARK_DEFINE_F(CudaFixture, Dense)(benchmark::State& state) {
    int batch_size = state.range(0);
    int in_features = state.range(1);
    int out_features = state.range(2);
    
    DenseLayer dense(in_features, out_features);
    Tensor input({batch_size, in_features});
    
    for (auto _ : state) {
        auto output = dense.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    // FLOPS: batch_size * in_features * out_features * 2 (multiply-add)
    state.SetItemsProcessed(state.iterations() * batch_size * in_features * out_features * 2);
}

BENCHMARK_REGISTER_F(CudaFixture, Dense)
    ->Args({1, 1024, 1024})
    ->Args({32, 1024, 1024})
    ->Args({128, 1024, 1024})
    ->Args({1, 4096, 4096})    // Large layers
    ->Args({32, 4096, 4096})
    ->Args({1, 768, 3072})     // Transformer FFN
    ->Args({32, 768, 3072})
    ->Unit(benchmark::kMicrosecond);

// Fused operations benchmarks
BENCHMARK_DEFINE_F(CudaFixture, ConvBNReLU_Separate)(benchmark::State& state) {
    int batch_size = state.range(0);
    
    Conv2d conv(64, 128, 3, 1, 1);
    BatchNorm2d bn(128);
    ReLU relu;
    bn.set_training(false);
    
    Tensor input({batch_size, 64, 56, 56});
    
    for (auto _ : state) {
        auto x = conv.forward(input, *ctx_);
        x = bn.forward(x, *ctx_);
        x = relu.forward(x, *ctx_);
        ctx_->synchronize();
    }
}

BENCHMARK_DEFINE_F(CudaFixture, ConvBNReLU_Fused)(benchmark::State& state) {
    int batch_size = state.range(0);
    
    auto conv = std::make_shared<Conv2d>(64, 128, 3, 1, 1);
    auto bn = std::make_shared<BatchNorm2d>(128);
    auto relu = std::make_shared<ReLU>();
    bn->set_training(false);
    
    ConvBNReLU fused(conv, bn, relu);
    Tensor input({batch_size, 64, 56, 56});
    
    for (auto _ : state) {
        auto output = fused.forward(input, *ctx_);
        ctx_->synchronize();
    }
}

BENCHMARK_REGISTER_F(CudaFixture, ConvBNReLU_Separate)
    ->Arg(1)->Arg(8)->Arg(32)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(CudaFixture, ConvBNReLU_Fused)
    ->Arg(1)->Arg(8)->Arg(32)
    ->Unit(benchmark::kMicrosecond);

// Main function
BENCHMARK_MAIN();