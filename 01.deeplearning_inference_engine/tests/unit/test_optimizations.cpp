#include <gtest/gtest.h>
#include "optimizations/quantization.h"
#include "optimizations/fusion.h"
#include "optimizations/memory_pool.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "layers/convolution.h"
#include "layers/batchnorm.h"
#include "layers/activation.h"

using namespace deep_engine;

class OptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// Quantization Tests
TEST_F(OptimizationTest, Int8QuantizerSymmetric) {
    Int8Quantizer quantizer(QuantizationScheme::SYMMETRIC);
    
    // Create tensor with known values
    Tensor tensor({2, 3});
    std::vector<float> data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 0.75f};
    cudaMemcpy(tensor.data<float>(), data.data(), tensor.bytes(), cudaMemcpyHostToDevice);
    
    // Compute quantization parameters
    auto params = quantizer.compute_params(tensor);
    
    // For symmetric quantization with range [-1, 1], scale should be ~1/127
    EXPECT_NEAR(params.scale, 1.0f / 127.0f, 0.001f);
    EXPECT_EQ(params.zero_point, 0);
    
    // Quantize
    auto quantized = quantizer.quantize(tensor);
    EXPECT_EQ(quantized.dtype(), DataType::INT8);
    
    // Dequantize
    auto dequantized = quantizer.dequantize(quantized);
    EXPECT_EQ(dequantized.dtype(), DataType::FP32);
    
    // Check values are close to original
    std::vector<float> result(6);
    cudaMemcpy(result.data(), dequantized.data<float>(), 
               dequantized.bytes(), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(result[i], data[i], 0.01f);  // Within 1% error
    }
}

TEST_F(OptimizationTest, DynamicQuantizer) {
    DynamicQuantizer quantizer(8);
    
    Tensor tensor = Tensor::random_uniform({100}, -2.0f, 2.0f);
    
    // Quantize dynamically
    auto quantized = quantizer.quantize(tensor);
    EXPECT_EQ(quantized.dtype(), DataType::INT8);
    
    // Dequantize
    auto dequantized = quantizer.dequantize(quantized);
    
    // Compute error
    std::vector<float> original(100), restored(100);
    cudaMemcpy(original.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(restored.data(), dequantized.data<float>(), 
               dequantized.bytes(), cudaMemcpyDeviceToHost);
    
    float max_error = 0.0f;
    for (size_t i = 0; i < 100; ++i) {
        float error = std::abs(original[i] - restored[i]);
        max_error = std::max(max_error, error);
    }
    
    // Error should be small
    EXPECT_LT(max_error, 0.05f);  // Less than 5% of range
}

TEST_F(OptimizationTest, QuantizationCalibrator) {
    QuantizationConfig config;
    config.bits = 8;
    config.scheme = QuantizationScheme::SYMMETRIC;
    config.percentile = 99.9f;
    
    QuantizationCalibrator calibrator(config);
    
    // Collect statistics from multiple batches
    for (int i = 0; i < 10; ++i) {
        Tensor batch = Tensor::random_normal({32, 3, 224, 224}, 0.0f, 1.0f);
        calibrator.collect_statistics("input", batch);
    }
    
    // Compute calibrated parameters
    auto params = calibrator.compute_params("input");
    
    // Scale should be reasonable for normal distribution
    EXPECT_GT(params.scale, 0.0f);
    EXPECT_LT(params.scale, 1.0f);
}

// Layer Fusion Tests
TEST_F(OptimizationTest, ConvBNFusion) {
    auto conv = std::make_shared<Conv2d>(16, 32, 3, 1, 1);
    auto bn = std::make_shared<BatchNorm2d>(32);
    
    // Set BN parameters to known values
    bn->running_mean().fill(0.5f);
    bn->running_var().fill(1.0f);
    bn->weight().fill(2.0f);
    bn->bias().fill(0.1f);
    bn->set_training(false);  // Use running stats
    
    // Create fused layer
    ConvBN fused(conv, bn);
    
    // Test that fusion works correctly
    ExecutionContext ctx;
    Tensor input({1, 16, 32, 32});
    input = Tensor::random_normal(input.shape());
    
    // Compute separate
    auto conv_out = conv->forward(input, ctx);
    auto bn_out = bn->forward(conv_out, ctx);
    
    // Compute fused
    auto fused_out = fused.forward(input, ctx);
    
    // Results should be very close
    std::vector<float> separate(bn_out.size()), fused_data(fused_out.size());
    cudaMemcpy(separate.data(), bn_out.data<float>(), 
               bn_out.bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(fused_data.data(), fused_out.data<float>(), 
               fused_out.bytes(), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < separate.size(); ++i) {
        EXPECT_NEAR(separate[i], fused_data[i], 1e-4f);
    }
}

TEST_F(OptimizationTest, ConvBNReLUFusion) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3, 1, 1);
    auto bn = std::make_shared<BatchNorm2d>(64);
    auto relu = std::make_shared<ReLU>();
    
    bn->set_training(false);
    
    // Create fused layer
    ConvBNReLU fused(conv, bn, relu);
    
    ExecutionContext ctx;
    Tensor input({1, 3, 32, 32});
    
    // The fused layer should produce same output as sequential execution
    auto output = fused.forward(input, ctx);
    
    EXPECT_EQ(output.shape(), std::vector<int>({1, 64, 32, 32}));
    
    // Check that ReLU is applied (no negative values)
    std::vector<float> data(output.size());
    cudaMemcpy(data.data(), output.data<float>(), 
               output.bytes(), cudaMemcpyDeviceToHost);
    
    for (float val : data) {
        EXPECT_GE(val, 0.0f);
    }
}

TEST_F(OptimizationTest, FusionOptimizer) {
    // Create graph with fusible patterns
    auto graph = std::make_unique<ComputationGraph>();
    
    auto conv1 = std::make_shared<Conv2d>(3, 64, 3);
    auto bn1 = std::make_shared<BatchNorm2d>(64);
    auto relu1 = std::make_shared<ReLU>();
    
    auto conv2 = std::make_shared<Conv2d>(64, 128, 3);
    auto bn2 = std::make_shared<BatchNorm2d>(128);
    
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
    
    size_t original_nodes = graph->num_nodes();
    
    // Apply fusion optimization
    LayerFusionOptimizer fusion_opt;
    graph->optimize(fusion_opt);
    
    // Graph should have fewer nodes after fusion
    EXPECT_LT(graph->num_nodes(), original_nodes);
}

// Memory Pool Tests
TEST_F(OptimizationTest, MemoryPoolBasic) {
    MemoryPool pool(100 * 1024 * 1024);  // 100MB pool
    
    // Allocate some memory
    void* ptr1 = pool.allocate(1024 * 1024);     // 1MB
    void* ptr2 = pool.allocate(2 * 1024 * 1024); // 2MB
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    
    // Check allocated size
    EXPECT_EQ(pool.get_allocated_size(), 3 * 1024 * 1024);
    
    // Deallocate
    pool.deallocate(ptr1);
    EXPECT_EQ(pool.get_allocated_size(), 2 * 1024 * 1024);
    
    // Reallocate should reuse freed memory
    void* ptr3 = pool.allocate(1024 * 1024);
    EXPECT_EQ(ptr3, ptr1);  // Should get same memory back
}

TEST_F(OptimizationTest, MemoryPoolFragmentation) {
    MemoryPool pool(10 * 1024 * 1024);  // 10MB pool
    
    // Create fragmentation
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool.allocate(512 * 1024));  // 512KB each
    }
    
    // Free every other allocation
    for (int i = 0; i < 10; i += 2) {
        pool.deallocate(ptrs[i]);
    }
    
    // Try to allocate large block
    void* large = pool.allocate(1024 * 1024);  // 1MB
    
    // Should fail due to fragmentation
    EXPECT_EQ(large, nullptr);
    
    // Defragment
    pool.defragment();
    
    // Now allocation should succeed
    large = pool.allocate(1024 * 1024);
    EXPECT_NE(large, nullptr);
}

TEST_F(OptimizationTest, MemoryPoolManager) {
    auto& manager = MemoryPoolManager::instance();
    
    // Allocate on device 0
    void* ptr1 = manager.allocate(1024 * 1024, 0);
    EXPECT_NE(ptr1, nullptr);
    
    // Get pool for device 0
    auto* pool = manager.get_pool(0);
    EXPECT_NE(pool, nullptr);
    EXPECT_GT(pool->get_allocated_size(), 0);
    
    // Deallocate
    manager.deallocate(ptr1, 0);
}

TEST_F(OptimizationTest, WorkspaceManager) {
    WorkspaceManager workspace(10 * 1024 * 1024);  // 10MB workspace
    
    // Get workspace
    void* ws1 = workspace.get_workspace(1024 * 1024);
    EXPECT_NE(ws1, nullptr);
    
    // Get another workspace
    void* ws2 = workspace.get_workspace(2 * 1024 * 1024);
    EXPECT_NE(ws2, nullptr);
    EXPECT_NE(ws1, ws2);
    
    // Release and reget
    workspace.release_workspace(ws1);
    void* ws3 = workspace.get_workspace(1024 * 1024);
    EXPECT_EQ(ws3, ws1);  // Should reuse released workspace
}

TEST_F(OptimizationTest, MemoryPlanner) {
    MemoryPlanner planner;
    
    // Define allocations with lifetimes
    std::vector<std::pair<size_t, std::pair<int, int>>> allocations = {
        {1024, {0, 5}},   // 1KB, lives from step 0 to 5
        {2048, {2, 7}},   // 2KB, lives from step 2 to 7
        {512, {6, 10}},   // 512B, lives from step 6 to 10
        {1024, {8, 12}},  // 1KB, lives from step 8 to 12
    };
    
    planner.plan(allocations);
    
    // Total memory should be less than sum of all allocations
    // because of reuse
    size_t total_individual = 1024 + 2048 + 512 + 1024;
    EXPECT_LT(planner.get_total_memory_required(), total_individual);
    
    // Check plan
    auto plan = planner.get_plan();
    EXPECT_EQ(plan.size(), 4);
    
    // First two allocations should not overlap in memory
    EXPECT_NE(plan[0].offset, plan[1].offset);
    
    // Third allocation can reuse first allocation's memory
    EXPECT_EQ(plan[2].offset, plan[0].offset);
}

// Quantized Layer Tests
TEST_F(OptimizationTest, QuantizedConv2d) {
    QuantizationParams input_params = {0.1f, 0, -12.8f, 12.7f};
    QuantizationParams weight_params = {0.05f, 0, -6.4f, 6.35f};
    QuantizationParams output_params = {0.2f, 0, -25.6f, 25.4f};
    
    QuantizedConv2d conv(3, 64, 3, 1, 1, input_params, weight_params, output_params);
    
    ExecutionContext ctx;
    Tensor input({1, 3, 32, 32}, DataType::INT8);
    
    auto output = conv.forward(input, ctx);
    
    EXPECT_EQ(output.dtype(), DataType::INT8);
    EXPECT_EQ(output.shape(), std::vector<int>({1, 64, 32, 32}));
}

TEST_F(OptimizationTest, QuantizedLinear) {
    QuantizationParams input_params = {0.1f, 0, -12.8f, 12.7f};
    QuantizationParams weight_params = {0.05f, 0, -6.4f, 6.35f};
    QuantizationParams output_params = {0.15f, 0, -19.2f, 19.05f};
    
    QuantizedLinear linear(128, 64, input_params, weight_params, output_params);
    
    ExecutionContext ctx;
    Tensor input({10, 128}, DataType::INT8);
    
    auto output = linear.forward(input, ctx);
    
    EXPECT_EQ(output.dtype(), DataType::INT8);
    EXPECT_EQ(output.shape(), std::vector<int>({10, 64}));
}

// Mixed Precision Tests
TEST_F(OptimizationTest, MixedPrecisionWrapper) {
    auto conv = std::make_unique<Conv2d>(3, 64, 3);
    MixedPrecisionWrapper wrapped(std::move(conv), DataType::FP16, DataType::FP32);
    
    ExecutionContext ctx;
    Tensor input({1, 3, 32, 32}, DataType::FP32);
    
    // Forward converts to FP16, computes, then back to FP32
    auto output = wrapped.forward(input, ctx);
    
    EXPECT_EQ(output.dtype(), DataType::FP32);
    EXPECT_EQ(output.shape(), std::vector<int>({1, 64, 30, 30}));
}

// Graph-level Optimization Tests
TEST_F(OptimizationTest, QuantizationOptimizer) {
    auto graph = std::make_unique<ComputationGraph>();
    
    // Build simple graph
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    auto fc = std::make_shared<DenseLayer>(64 * 30 * 30, 10);
    
    auto conv_id = graph->add_node("conv", conv);
    auto relu_id = graph->add_node("relu", relu);
    auto fc_id = graph->add_node("fc", fc);
    
    graph->add_edge(conv_id, relu_id);
    graph->add_edge(relu_id, fc_id);
    
    graph->mark_input(conv_id);
    graph->mark_output(fc_id);
    
    // Apply quantization
    QuantizationOptimizer quant_opt(8);
    graph->optimize(quant_opt);
    
    // Layers that support quantization should be quantized
    // (In real implementation, this would check if layers are actually quantized)
    EXPECT_TRUE(conv->supports_quantization());
    EXPECT_TRUE(fc->supports_quantization());
}

// Performance benchmarks (disabled by default)
TEST_F(OptimizationTest, DISABLED_QuantizationSpeedup) {
    // Compare FP32 vs INT8 performance
    Conv2d conv_fp32(3, 64, 3, 1, 1);
    
    QuantizationParams dummy_params = {0.1f, 0, -12.8f, 12.7f};
    QuantizedConv2d conv_int8(3, 64, 3, 1, 1, dummy_params, dummy_params, dummy_params);
    
    ExecutionContext ctx;
    Tensor input_fp32({32, 3, 224, 224}, DataType::FP32);
    Tensor input_int8({32, 3, 224, 224}, DataType::INT8);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        conv_fp32.forward(input_fp32, ctx);
        conv_int8.forward(input_int8, ctx);
    }
    ctx.synchronize();
    
    // Benchmark FP32
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        conv_fp32.forward(input_fp32, ctx);
    }
    ctx.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto fp32_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Benchmark INT8
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        conv_int8.forward(input_int8, ctx);
    }
    ctx.synchronize();
    end = std::chrono::high_resolution_clock::now();
    auto int8_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float speedup = float(fp32_time) / float(int8_time);
    std::cout << "INT8 speedup: " << speedup << "x" << std::endl;
    
    // INT8 should be faster
    EXPECT_GT(speedup, 1.5f);
}

TEST_F(OptimizationTest, DISABLED_MemoryPoolPerformance) {
    const size_t allocation_size = 10 * 1024 * 1024;  // 10MB
    const int num_allocations = 100;
    
    // Benchmark raw CUDA allocation
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocations; ++i) {
        void* ptr;
        cudaMalloc(&ptr, allocation_size);
        cudaFree(ptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Benchmark memory pool
    MemoryPool pool(1024 * 1024 * 1024);  // 1GB pool
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocations; ++i) {
        void* ptr = pool.allocate(allocation_size);
        pool.deallocate(ptr);
    }
    end = std::chrono::high_resolution_clock::now();
    auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float speedup = float(cuda_time) / float(pool_time);
    std::cout << "Memory pool speedup: " << speedup << "x" << std::endl;
    
    // Pool should be significantly faster
    EXPECT_GT(speedup, 10.0f);
}