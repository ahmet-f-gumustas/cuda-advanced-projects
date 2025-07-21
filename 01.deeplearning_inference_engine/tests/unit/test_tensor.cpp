#include <gtest/gtest.h>
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

using namespace deep_engine;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is initialized
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        // Clean up CUDA resources
        cudaDeviceSynchronize();
    }
};

TEST_F(TensorTest, ConstructorDefault) {
    Tensor tensor;
    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.ndim(), 0);
    EXPECT_TRUE(tensor.shape().empty());
}

TEST_F(TensorTest, ConstructorWithShape) {
    std::vector<int> shape = {2, 3, 4, 5};
    Tensor tensor(shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 4);
    EXPECT_EQ(tensor.size(), 2 * 3 * 4 * 5);
    EXPECT_EQ(tensor.dtype(), DataType::FP32);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(TensorTest, ConstructorWithShapeAndType) {
    std::vector<int> shape = {10, 20};
    Tensor tensor(shape, DataType::INT8);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 200);
    EXPECT_EQ(tensor.dtype(), DataType::INT8);
    EXPECT_EQ(tensor.bytes(), 200); // INT8 = 1 byte per element
}

TEST_F(TensorTest, FactoryZeros) {
    auto tensor = Tensor::zeros({3, 4, 5});
    
    EXPECT_EQ(tensor.shape(), std::vector<int>({3, 4, 5}));
    EXPECT_EQ(tensor.size(), 60);
    
    // Copy to host and verify all zeros
    std::vector<float> host_data(tensor.size());
    cudaMemcpy(host_data.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    
    for (float val : host_data) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST_F(TensorTest, FactoryOnes) {
    auto tensor = Tensor::ones({2, 3});
    
    EXPECT_EQ(tensor.size(), 6);
    
    // Copy to host and verify all ones
    std::vector<float> host_data(tensor.size());
    cudaMemcpy(host_data.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    
    for (float val : host_data) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(TensorTest, FactoryRandomUniform) {
    auto tensor = Tensor::random_uniform({100}, 0.0f, 1.0f);
    
    // Copy to host
    std::vector<float> host_data(tensor.size());
    cudaMemcpy(host_data.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    
    // Check all values are in range [0, 1]
    for (float val : host_data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
    
    // Check that values are not all the same (random)
    float first_val = host_data[0];
    bool all_same = true;
    for (float val : host_data) {
        if (val != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST_F(TensorTest, FactoryRandomNormal) {
    auto tensor = Tensor::random_normal({1000}, 0.0f, 1.0f);
    
    // Copy to host
    std::vector<float> host_data(tensor.size());
    cudaMemcpy(host_data.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    
    // Calculate mean and std
    float sum = std::accumulate(host_data.begin(), host_data.end(), 0.0f);
    float mean = sum / host_data.size();
    
    float sq_sum = 0.0f;
    for (float val : host_data) {
        sq_sum += (val - mean) * (val - mean);
    }
    float std = std::sqrt(sq_sum / host_data.size());
    
    // Check mean is close to 0 and std is close to 1
    EXPECT_NEAR(mean, 0.0f, 0.1f);
    EXPECT_NEAR(std, 1.0f, 0.1f);
}

TEST_F(TensorTest, View) {
    auto tensor = Tensor::ones({2, 3, 4});
    
    // Valid view
    auto viewed = tensor.view({6, 4});
    EXPECT_EQ(viewed.shape(), std::vector<int>({6, 4}));
    EXPECT_EQ(viewed.size(), tensor.size());
    EXPECT_EQ(viewed.data(), tensor.data()); // Same underlying data
    
    // Invalid view (size mismatch)
    EXPECT_THROW(tensor.view({5, 5}), std::runtime_error);
}

TEST_F(TensorTest, Reshape) {
    auto tensor = Tensor::ones({4, 5});
    auto reshaped = tensor.reshape({2, 10});
    
    EXPECT_EQ(reshaped.shape(), std::vector<int>({2, 10}));
    EXPECT_EQ(reshaped.size(), tensor.size());
}

TEST_F(TensorTest, Transpose2D) {
    // Create a 2x3 tensor with known values
    Tensor tensor({2, 3});
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    cudaMemcpy(tensor.data<float>(), data.data(), 
               tensor.bytes(), cudaMemcpyHostToDevice);
    
    // Transpose
    auto transposed = tensor.transpose({1, 0});
    
    EXPECT_EQ(transposed.shape(), std::vector<int>({3, 2}));
    
    // Verify transposed values
    std::vector<float> result(6);
    cudaMemcpy(result.data(), transposed.data<float>(), 
               transposed.bytes(), cudaMemcpyDeviceToHost);
    
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(TensorTest, Slice) {
    auto tensor = Tensor::ones({10, 20, 30});
    
    // Slice along axis 0
    auto sliced = tensor.slice(0, 2, 5);
    EXPECT_EQ(sliced.shape(), std::vector<int>({3, 20, 30}));
    
    // Slice along axis 1
    auto sliced2 = tensor.slice(1, 5, 15);
    EXPECT_EQ(sliced2.shape(), std::vector<int>({10, 10, 30}));
}

TEST_F(TensorTest, Squeeze) {
    auto tensor = Tensor::ones({1, 3, 1, 4, 1});
    
    // Squeeze all dimensions
    auto squeezed = tensor.squeeze();
    EXPECT_EQ(squeezed.shape(), std::vector<int>({3, 4}));
    
    // Squeeze specific dimension
    auto squeezed2 = tensor.squeeze(0);
    EXPECT_EQ(squeezed2.shape(), std::vector<int>({3, 1, 4, 1}));
}

TEST_F(TensorTest, Unsqueeze) {
    auto tensor = Tensor::ones({3, 4});
    
    // Add dimension at position 0
    auto unsqueezed = tensor.unsqueeze(0);
    EXPECT_EQ(unsqueezed.shape(), std::vector<int>({1, 3, 4}));
    
    // Add dimension at position 1
    auto unsqueezed2 = tensor.unsqueeze(1);
    EXPECT_EQ(unsqueezed2.shape(), std::vector<int>({3, 1, 4}));
    
    // Add dimension at the end
    auto unsqueezed3 = tensor.unsqueeze(2);
    EXPECT_EQ(unsqueezed3.shape(), std::vector<int>({3, 4, 1}));
}

TEST_F(TensorTest, Clone) {
    auto tensor = Tensor::random_uniform({5, 5});
    auto cloned = tensor.clone();
    
    // Different memory locations
    EXPECT_NE(cloned.data(), tensor.data());
    
    // Same shape and size
    EXPECT_EQ(cloned.shape(), tensor.shape());
    EXPECT_EQ(cloned.size(), tensor.size());
    
    // Same values
    std::vector<float> original(25), cloned_data(25);
    cudaMemcpy(original.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(cloned_data.data(), cloned.data<float>(), 
               cloned.bytes(), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(original[i], cloned_data[i]);
    }
}

TEST_F(TensorTest, MoveConstructor) {
    auto tensor = Tensor::ones({3, 3});
    void* original_data = tensor.data();
    
    Tensor moved(std::move(tensor));
    
    // Moved tensor has the data
    EXPECT_EQ(moved.data(), original_data);
    EXPECT_EQ(moved.shape(), std::vector<int>({3, 3}));
    
    // Original tensor is empty
    EXPECT_EQ(tensor.data(), nullptr);
}

TEST_F(TensorTest, MoveAssignment) {
    auto tensor1 = Tensor::ones({2, 2});
    auto tensor2 = Tensor::zeros({3, 3});
    
    void* original_data = tensor1.data();
    
    tensor2 = std::move(tensor1);
    
    // tensor2 now has tensor1's data
    EXPECT_EQ(tensor2.data(), original_data);
    EXPECT_EQ(tensor2.shape(), std::vector<int>({2, 2}));
    
    // tensor1 is empty
    EXPECT_EQ(tensor1.data(), nullptr);
}

TEST_F(TensorTest, CopyConstructor) {
    auto tensor = Tensor::random_uniform({4, 4});
    Tensor copied(tensor);
    
    // Different memory locations
    EXPECT_NE(copied.data(), tensor.data());
    
    // Same properties
    EXPECT_EQ(copied.shape(), tensor.shape());
    EXPECT_EQ(copied.dtype(), tensor.dtype());
}

TEST_F(TensorTest, CopyAssignment) {
    auto tensor1 = Tensor::ones({2, 3});
    auto tensor2 = Tensor::zeros({4, 5});
    
    tensor2 = tensor1;
    
    // Different memory locations
    EXPECT_NE(tensor2.data(), tensor1.data());
    
    // Same shape after assignment
    EXPECT_EQ(tensor2.shape(), tensor1.shape());
}

TEST_F(TensorTest, Fill) {
    auto tensor = Tensor::zeros({10, 10});
    tensor.fill(3.14f);
    
    std::vector<float> data(100);
    cudaMemcpy(data.data(), tensor.data<float>(), 
               tensor.bytes(), cudaMemcpyDeviceToHost);
    
    for (float val : data) {
        EXPECT_FLOAT_EQ(val, 3.14f);
    }
}

TEST_F(TensorTest, CatAxis0) {
    auto t1 = Tensor::ones({2, 3});
    auto t2 = Tensor::ones({3, 3}) * 2.0f;  // Filled with 2s
    auto t3 = Tensor::ones({1, 3}) * 3.0f;  // Filled with 3s
    
    auto result = cat({t1, t2, t3}, 0);
    
    EXPECT_EQ(result.shape(), std::vector<int>({6, 3}));
}

TEST_F(TensorTest, CatInvalidShapes) {
    auto t1 = Tensor::ones({2, 3});
    auto t2 = Tensor::ones({2, 4}); // Different size on non-cat axis
    
    EXPECT_THROW(cat({t1, t2}, 0), std::runtime_error);
}

// Test quantization when implemented
TEST_F(TensorTest, QuantizeInt8) {
    auto tensor = Tensor::random_uniform({100}, -1.0f, 1.0f);
    
    // For now, just test that the method exists
    // When implemented, add proper quantization tests
    EXPECT_NO_THROW(tensor.quantize_int8());
}

// Performance test (disabled by default)
TEST_F(TensorTest, DISABLED_LargeTensorCreation) {
    // Test creating large tensors
    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor large({1024, 1024, 100}); // 100M elements
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Large tensor creation took: " << duration.count() << "ms" << std::endl;
    EXPECT_LT(duration.count(), 1000); // Should take less than 1 second
}