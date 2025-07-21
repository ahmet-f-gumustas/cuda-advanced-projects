#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "types.h"
#include "allocator.h"

namespace deep_engine {

class Tensor {
private:
    std::vector<int> shape_;
    size_t size_;
    DataType dtype_;
    std::shared_ptr<MemoryAllocator> allocator_;
    void* data_;
    bool owns_data_;
    
public:
    // Constructors
    Tensor() : size_(0), dtype_(DataType::FP32), data_(nullptr), owns_data_(false) {}
    
    explicit Tensor(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    
    Tensor(const std::vector<int>& shape, void* data, DataType dtype = DataType::FP32);
    
    // Move constructor
    Tensor(Tensor&& other) noexcept;
    
    // Copy constructor
    Tensor(const Tensor& other);
    
    // Destructor
    ~Tensor();
    
    // Assignment operators
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    
    // Factory methods
    static Tensor zeros(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    static Tensor ones(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    static Tensor random_uniform(const std::vector<int>& shape, float min = 0.0f, float max = 1.0f);
    static Tensor random_normal(const std::vector<int>& shape, float mean = 0.0f, float std = 1.0f);
    
    // Shape operations
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor transpose(const std::vector<int>& axes) const;
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    
    // Slice operations
    Tensor slice(int axis, int start, int end) const;
    Tensor slice(const std::vector<std::pair<int, int>>& ranges) const;
    
    // Type conversion
    Tensor to(DataType new_dtype) const;
    Tensor to_device(int device_id) const;
    Tensor to_host() const;
    
    // Quantization
    Tensor quantize_int8(float scale = 0.0f, int zero_point = 0) const;
    std::pair<Tensor, float> dynamic_quantize_int8() const;
    
    // Accessors
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    template<typename T>
    T* data() { return static_cast<T*>(data_); }
    
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_); }
    
    const std::vector<int>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * dtype_size(dtype_); }
    DataType dtype() const { return dtype_; }
    int ndim() const { return shape_.size(); }
    
    // Element access (for debugging, host only)
    template<typename T>
    T get(const std::vector<int>& indices) const;
    
    template<typename T>
    void set(const std::vector<int>& indices, T value);
    
    // Utility functions
    void fill(float value);
    void copy_from(const void* src, size_t bytes);
    void copy_to(void* dst, size_t bytes) const;
    Tensor clone() const;
    
    // Debug
    void print(const std::string& name = "") const;
    std::string to_string() const;
    
private:
    void allocate();
    void deallocate();
    size_t compute_size(const std::vector<int>& shape) const;
    size_t dtype_size(DataType dtype) const;
};

// Tensor operations
Tensor cat(const std::vector<Tensor>& tensors, int axis = 0);
std::vector<Tensor> split(const Tensor& tensor, int chunks, int axis = 0);
Tensor stack(const std::vector<Tensor>& tensors, int axis = 0);

} // namespace deep_engine