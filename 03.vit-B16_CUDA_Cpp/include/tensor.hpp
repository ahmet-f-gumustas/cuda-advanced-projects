#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <stdexcept>

class Tensor {
public:
    enum class DataType { FP32, FP16 };
    
private:
    void* data_ = nullptr;
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t size_ = 0;
    DataType dtype_;
    bool owns_memory_ = true;
    
public:
    Tensor() = default;
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    Tensor(void* data, const std::vector<int>& shape, DataType dtype);
    ~Tensor();
    
    // Move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Delete copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Accessors
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    template<typename T>
    T* data_ptr() { return static_cast<T*>(data_); }
    
    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data_); }
    
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    size_t size() const { return size_; }
    DataType dtype() const { return dtype_; }
    size_t element_size() const;
    size_t nbytes() const { return size_ * element_size(); }
    int ndim() const { return shape_.size(); }
    
    // Operations
    void reshape(const std::vector<int>& new_shape);
    Tensor view(const std::vector<int>& new_shape) const;
    void fill(float value);
    void copy_from_host(const void* host_data);
    void copy_to_host(void* host_data) const;
    
    // Static factory methods
    static Tensor zeros(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    static Tensor ones(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    static Tensor randn(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    
private:
    void compute_strides();
    void allocate();
    void deallocate();
};