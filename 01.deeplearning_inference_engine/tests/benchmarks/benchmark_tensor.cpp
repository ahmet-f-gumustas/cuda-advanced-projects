#include <benchmark/benchmark.h>
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>

using namespace deep_engine;

// Helper fixture for CUDA initialization
class TensorBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        cudaSetDevice(0);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        cudaDeviceSynchronize();
    }
};

// Tensor creation benchmarks
BENCHMARK_DEFINE_F(TensorBenchmark, TensorCreation)(benchmark::State& state) {
    std::vector<int> shape = {
        static_cast<int>(state.range(0)),
        static_cast<int>(state.range(1)), 
        static_cast<int>(state.range(2)),
        static_cast<int>(state.range(3))
    };
    
    for (auto _ : state) {
        Tensor tensor(shape);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * 
        state.range(0) * state.range(1) * state.range(2) * state.range(3) * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorCreation)
    ->Args({1, 3, 224, 224})      // Typical image size
    ->Args({32, 3, 224, 224})     // Batch of images
    ->Args({1, 512, 7, 7})        // Feature maps
    ->Args({32, 512, 7, 7})       // Batch of feature maps
    ->Args({128, 768})            // Transformer hidden states
    ->Args({32, 1024, 1024})      // Large matrices
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorZeros)(benchmark::State& state) {
    int size = state.range(0);
    std::vector<int> shape = {size};
    
    for (auto _ : state) {
        auto tensor = Tensor::zeros(shape);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorZeros)
    ->Range(1024, 1024*1024*10)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorOnes)(benchmark::State& state) {
    int size = state.range(0);
    std::vector<int> shape = {size};
    
    for (auto _ : state) {
        auto tensor = Tensor::ones(shape);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorOnes)
    ->Range(1024, 1024*1024*10)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorRandomUniform)(benchmark::State& state) {
    int size = state.range(0);
    std::vector<int> shape = {size};
    
    for (auto _ : state) {
        auto tensor = Tensor::random_uniform(shape, 0.0f, 1.0f);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorRandomUniform)
    ->Range(1024, 1024*1024*10)
    ->Unit(benchmark::kMicrosecond);

// Tensor operations benchmarks
BENCHMARK_DEFINE_F(TensorBenchmark, TensorView)(benchmark::State& state) {
    int size = state.range(0);
    Tensor tensor({size});
    
    for (auto _ : state) {
        auto viewed = tensor.view({size/2, 2});
        benchmark::DoNotOptimize(viewed);
    }
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorView)
    ->Range(1024, 1024*1024)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorTranspose)(benchmark::State& state) {
    int rows = state.range(0);
    int cols = state.range(1);
    Tensor tensor({rows, cols});
    
    for (auto _ : state) {
        auto transposed = tensor.transpose({1, 0});
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(float) * 2);
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorTranspose)
    ->Args({128, 128})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({1024, 768})   // Non-square
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorSlice)(benchmark::State& state) {
    int size = state.range(0);
    int slice_size = state.range(1);
    Tensor tensor({size});
    
    for (auto _ : state) {
        auto sliced = tensor.slice(0, 0, slice_size);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * slice_size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorSlice)
    ->Args({10000, 1000})
    ->Args({10000, 5000})
    ->Args({100000, 10000})
    ->Args({1000000, 100000})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorClone)(benchmark::State& state) {
    int size = state.range(0);
    Tensor tensor({size});
    
    for (auto _ : state) {
        auto cloned = tensor.clone();
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorClone)
    ->Range(1024, 1024*1024*10)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorCat)(benchmark::State& state) {
    int num_tensors = state.range(0);
    int tensor_size = state.range(1);
    
    std::vector<Tensor> tensors;
    for (int i = 0; i < num_tensors; ++i) {
        tensors.push_back(Tensor::ones({tensor_size}));
    }
    
    for (auto _ : state) {
        auto result = cat(tensors, 0);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * num_tensors * tensor_size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorCat)
    ->Args({2, 10000})
    ->Args({4, 10000})
    ->Args({8, 10000})
    ->Args({16, 10000})
    ->Args({2, 100000})
    ->Args({4, 100000})
    ->Unit(benchmark::kMicrosecond);

// Memory copy benchmarks
BENCHMARK_DEFINE_F(TensorBenchmark, TensorCopyFrom)(benchmark::State& state) {
    int size = state.range(0);
    Tensor src({size});
    Tensor dst({size});
    
    for (auto _ : state) {
        dst.copy_from(src.data(), src.bytes());
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorCopyFrom)
    ->Range(1024, 1024*1024*100)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(TensorBenchmark, TensorFill)(benchmark::State& state) {
    int size = state.range(0);
    Tensor tensor({size});
    
    for (auto _ : state) {
        tensor.fill(3.14f);
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorFill)
    ->Range(1024, 1024*1024*100)
    ->Unit(benchmark::kMicrosecond);

// Type conversion benchmarks
BENCHMARK_DEFINE_F(TensorBenchmark, TensorQuantizeINT8)(benchmark::State& state) {
    int size = state.range(0);
    Tensor tensor = Tensor::random_uniform({size}, -1.0f, 1.0f);
    
    for (auto _ : state) {
        auto quantized = tensor.quantize_int8();
        cudaDeviceSynchronize();
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK_REGISTER_F(TensorBenchmark, TensorQuantizeINT8)
    ->Range(1024, 1024*1024*10)
    ->Unit(benchmark::kMicrosecond);

// Main function
BENCHMARK_MAIN();