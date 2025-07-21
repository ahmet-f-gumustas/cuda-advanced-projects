#include <benchmark/benchmark.h>
#include "core/graph.h"
#include "core/tensor.h"
#include "layers/convolution.h"
#include "layers/activation.h"
#include "layers/pooling.h"
#include "layers/batchnorm.h"
#include "layers/dense.h"
#include "../unit/test_layers_helpers.h"
#include <memory>
#include <cuda_runtime.h>

using namespace deep_engine;

class ModelBenchmark : public benchmark::Fixture {
protected:
    std::unique_ptr<ExecutionContext> ctx_;
    
    void SetUp(const ::benchmark::State& state) override {
        cudaSetDevice(0);
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    void TearDown(const ::benchmark::State& state) override {
        ctx_->synchronize();
    }
    
    // Helper to build ResNet block
    std::unique_ptr<ComputationGraph> build_resnet_block(int in_channels, int out_channels, 
                                                         int stride = 1) {
        auto graph = std::make_unique<ComputationGraph>();
        
        // Main path
        auto conv1 = std::make_shared<Conv2d>(in_channels, out_channels, 3, stride, 1);
        auto bn1 = std::make_shared<BatchNorm2d>(out_channels);
        auto relu1 = std::make_shared<ReLU>();
        auto conv2 = std::make_shared<Conv2d>(out_channels, out_channels, 3, 1, 1);
        auto bn2 = std::make_shared<BatchNorm2d>(out_channels);
        
        auto conv1_id = graph->add_node("conv1", conv1);
        auto bn1_id = graph->add_node("bn1", bn1);
        auto relu1_id = graph->add_node("relu1", relu1);
        auto conv2_id = graph->add_node("conv2", conv2);
        auto bn2_id = graph->add_node("bn2", bn2);
        
        graph->add_edge(conv1_id, bn1_id);
        graph->add_edge(bn1_id, relu1_id);
        graph->add_edge(relu1_id, conv2_id);
        graph->add_edge(conv2_id, bn2_id);
        
        graph->mark_input(conv1_id);
        graph->mark_output(bn2_id);
        
        return graph;
    }
    
    // Helper to build MobileNet block
    std::unique_ptr<ComputationGraph> build_mobilenet_block(int in_channels, int out_channels) {
        auto graph = std::make_unique<ComputationGraph>();
        
        // Depthwise separable convolution
        auto dw_conv = std::make_shared<DepthwiseConv2d>(in_channels, 3, 1, 1);
        auto dw_bn = std::make_shared<BatchNorm2d>(in_channels);
        auto dw_relu = std::make_shared<ReLU>();
        auto pw_conv = std::make_shared<Conv2d>(in_channels, out_channels, 1, 1, 0);
        auto pw_bn = std::make_shared<BatchNorm2d>(out_channels);
        auto pw_relu = std::make_shared<ReLU>();
        
        auto dw_conv_id = graph->add_node("dw_conv", dw_conv);
        auto dw_bn_id = graph->add_node("dw_bn", dw_bn);
        auto dw_relu_id = graph->add_node("dw_relu", dw_relu);
        auto pw_conv_id = graph->add_node("pw_conv", pw_conv);
        auto pw_bn_id = graph->add_node("pw_bn", pw_bn);
        auto pw_relu_id = graph->add_node("pw_relu", pw_relu);
        
        graph->add_edge(dw_conv_id, dw_bn_id);
        graph->add_edge(dw_bn_id, dw_relu_id);
        graph->add_edge(dw_relu_id, pw_conv_id);
        graph->add_edge(pw_conv_id, pw_bn_id);
        graph->add_edge(pw_bn_id, pw_relu_id);
        
        graph->mark_input(dw_conv_id);
        graph->mark_output(pw_relu_id);
        
        return graph;
    }
};

// Single layer benchmarks for baseline
BENCHMARK_DEFINE_F(ModelBenchmark, SingleConv2d)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    Conv2d conv(channels, channels * 2, 3, 2, 1);  // Stride 2 to reduce size
    Tensor input({batch_size, channels, size, size});
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        conv.forward(input, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto output = conv.forward(input, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(ModelBenchmark, SingleConv2d)
    ->Args({1, 3, 224})     // Single image, RGB
    ->Args({8, 3, 224})     // Small batch
    ->Args({32, 3, 224})    // Typical batch
    ->Args({1, 64, 56})     // Deeper layer
    ->Args({8, 64, 56})
    ->Unit(benchmark::kMillisecond);

// ResNet block benchmark
BENCHMARK_DEFINE_F(ModelBenchmark, ResNetBlock)(benchmark::State& state) {
    int batch_size = state.range(0);
    int channels = state.range(1);
    int size = state.range(2);
    
    auto graph = build_resnet_block(channels, channels);
    graph->finalize();
    
    Tensor input({batch_size, channels, size, size});
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        graph->forward({input}, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto outputs = graph->forward({input}, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(ModelBenchmark, ResNetBlock)
    ->Args({1, 64, 56})
    ->Args({8, 64, 56})
    ->Args({32, 64, 56})
    ->Args({1, 256, 14})
    ->Args({8, 256, 14})
    ->Unit(benchmark::kMillisecond);

// MobileNet block benchmark
BENCHMARK_DEFINE_F(ModelBenchmark, MobileNetBlock)(benchmark::State& state) {
    int batch_size = state.range(0);
    int in_channels = state.range(1);
    int out_channels = state.range(2);
    int size = state.range(3);
    
    auto graph = build_mobilenet_block(in_channels, out_channels);
    graph->finalize();
    
    Tensor input({batch_size, in_channels, size, size});
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        graph->forward({input}, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto outputs = graph->forward({input}, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(ModelBenchmark, MobileNetBlock)
    ->Args({1, 32, 64, 112})
    ->Args({8, 32, 64, 112})
    ->Args({1, 128, 256, 28})
    ->Args({8, 128, 256, 28})
    ->Unit(benchmark::kMillisecond);

// Full model simulation - simplified VGG-like
BENCHMARK_DEFINE_F(ModelBenchmark, VGGLikeModel)(benchmark::State& state) {
    int batch_size = state.range(0);
    
    // Build simplified VGG-like model
    auto graph = std::make_unique<ComputationGraph>();
    
    // Conv block 1
    auto conv1_1 = std::make_shared<Conv2d>(3, 64, 3, 1, 1);
    auto relu1_1 = std::make_shared<ReLU>();
    auto conv1_2 = std::make_shared<Conv2d>(64, 64, 3, 1, 1);
    auto relu1_2 = std::make_shared<ReLU>();
    auto pool1 = std::make_shared<MaxPool2d>(2, 2);
    
    // Conv block 2
    auto conv2_1 = std::make_shared<Conv2d>(64, 128, 3, 1, 1);
    auto relu2_1 = std::make_shared<ReLU>();
    auto conv2_2 = std::make_shared<Conv2d>(128, 128, 3, 1, 1);
    auto relu2_2 = std::make_shared<ReLU>();
    auto pool2 = std::make_shared<MaxPool2d>(2, 2);
    
    // Add nodes
    auto conv1_1_id = graph->add_node("conv1_1", conv1_1);
    auto relu1_1_id = graph->add_node("relu1_1", relu1_1);
    auto conv1_2_id = graph->add_node("conv1_2", conv1_2);
    auto relu1_2_id = graph->add_node("relu1_2", relu1_2);
    auto pool1_id = graph->add_node("pool1", pool1);
    
    auto conv2_1_id = graph->add_node("conv2_1", conv2_1);
    auto relu2_1_id = graph->add_node("relu2_1", relu2_1);
    auto conv2_2_id = graph->add_node("conv2_2", conv2_2);
    auto relu2_2_id = graph->add_node("relu2_2", relu2_2);
    auto pool2_id = graph->add_node("pool2", pool2);
    
    // Connect nodes
    graph->add_edge(conv1_1_id, relu1_1_id);
    graph->add_edge(relu1_1_id, conv1_2_id);
    graph->add_edge(conv1_2_id, relu1_2_id);
    graph->add_edge(relu1_2_id, pool1_id);
    
    graph->add_edge(pool1_id, conv2_1_id);
    graph->add_edge(conv2_1_id, relu2_1_id);
    graph->add_edge(relu2_1_id, conv2_2_id);
    graph->add_edge(conv2_2_id, relu2_2_id);
    graph->add_edge(relu2_2_id, pool2_id);
    
    graph->mark_input(conv1_1_id);
    graph->mark_output(pool2_id);
    
    graph->finalize();
    
    Tensor input({batch_size, 3, 224, 224});
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        graph->forward({input}, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto outputs = graph->forward({input}, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(ModelBenchmark, VGGLikeModel)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Unit(benchmark::kMillisecond);

// Optimized vs Unoptimized graph comparison
BENCHMARK_DEFINE_F(ModelBenchmark, GraphOptimizationComparison)(benchmark::State& state) {
    bool apply_optimization = state.range(0);
    int batch_size = 8;
    
    // Build graph with fusible patterns
    auto graph = std::make_unique<ComputationGraph>();
    
    auto conv = std::make_shared<Conv2d>(64, 128, 3, 1, 1);
    auto bn = std::make_shared<BatchNorm2d>(128);
    auto relu = std::make_shared<ReLU>();
    
    auto conv_id = graph->add_node("conv", conv);
    auto bn_id = graph->add_node("bn", bn);
    auto relu_id = graph->add_node("relu", relu);
    
    graph->add_edge(conv_id, bn_id);
    graph->add_edge(bn_id, relu_id);
    
    graph->mark_input(conv_id);
    graph->mark_output(relu_id);
    
    if (apply_optimization) {
        LayerFusionOptimizer fusion_opt;
        graph->optimize(fusion_opt);
    }
    
    graph->finalize();
    
    Tensor input({batch_size, 64, 56, 56});
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        graph->forward({input}, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto outputs = graph->forward({input}, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(ModelBenchmark, GraphOptimizationComparison)
    ->Arg(0)  // Without optimization
    ->Arg(1)  // With optimization
    ->Unit(benchmark::kMillisecond);

// Memory-bound vs Compute-bound benchmark
BENCHMARK_DEFINE_F(ModelBenchmark, MemoryVsComputeBound)(benchmark::State& state) {
    int workload_type = state.range(0);  // 0 = memory-bound, 1 = compute-bound
    int batch_size = 32;
    
    auto graph = std::make_unique<ComputationGraph>();
    
    if (workload_type == 0) {
        // Memory-bound: many pointwise operations
        auto input_id = graph->add_node("input", std::make_shared<IdentityLayer>());
        auto prev_id = input_id;
        
        for (int i = 0; i < 10; ++i) {
            auto relu = std::make_shared<ReLU>();
            auto relu_id = graph->add_node("relu_" + std::to_string(i), relu);
            graph->add_edge(prev_id, relu_id);
            prev_id = relu_id;
        }
        
        graph->mark_input(input_id);
        graph->mark_output(prev_id);
    } else {
        // Compute-bound: heavy convolutions
        auto conv1 = std::make_shared<Conv2d>(512, 512, 3, 1, 1);
        auto conv2 = std::make_shared<Conv2d>(512, 512, 3, 1, 1);
        
        auto conv1_id = graph->add_node("conv1", conv1);
        auto conv2_id = graph->add_node("conv2", conv2);
        
        graph->add_edge(conv1_id, conv2_id);
        graph->mark_input(conv1_id);
        graph->mark_output(conv2_id);
    }
    
    graph->finalize();
    
    Tensor input = (workload_type == 0) ? 
        Tensor({batch_size, 512, 14, 14}) : 
        Tensor({batch_size, 512, 7, 7});
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        graph->forward({input}, *ctx_);
    }
    ctx_->synchronize();
    
    for (auto _ : state) {
        auto outputs = graph->forward({input}, *ctx_);
        ctx_->synchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK_REGISTER_F(ModelBenchmark, MemoryVsComputeBound)
    ->Arg(0)  // Memory-bound
    ->Arg(1)  // Compute-bound
    ->Unit(benchmark::kMillisecond);

// Dynamic batching simulation
BENCHMARK_DEFINE_F(ModelBenchmark, DynamicBatching)(benchmark::State& state) {
    int max_batch_size = state.range(0);
    
    // Simple model
    Conv2d conv(3, 64, 7, 2, 3);
    ReLU relu;
    MaxPool2d pool(3, 2, 1);
    
    // Simulate varying batch sizes
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    int current_idx = 0;
    
    for (auto _ : state) {
        int batch_size = batch_sizes[current_idx % batch_sizes.size()];
        if (batch_size > max_batch_size) batch_size = max_batch_size;
        
        Tensor input({batch_size, 3, 224, 224});
        
        auto x = conv.forward(input, *ctx_);
        x = relu.forward(x, *ctx_);
        x = pool.forward(x, *ctx_);
        
        ctx_->synchronize();
        
        state.SetItemsProcessed(batch_size);
        current_idx++;
    }
}

BENCHMARK_REGISTER_F(ModelBenchmark, DynamicBatching)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Unit(benchmark::kMillisecond);

// Main function
BENCHMARK_MAIN();