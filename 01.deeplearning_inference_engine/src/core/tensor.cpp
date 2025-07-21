#include "../../include/core/tensor.h"
#include "../../include/utils/logger.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <numeric>
#include <algorithm>
#include <random>

namespace deep_engine {

Tensor::Tensor(const std::vector<int>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), owns_data_(true) {
    size_ = compute_size(shape);
    allocator_ = AllocatorFactory::create(DeviceType::CUDA);
    allocate();
}

Tensor::Tensor(const std::vector<int>& shape, void* data, DataType dtype)
    : shape_(shape), dtype_(dtype), data_(data), owns_data_(false) {
    size_ = compute_size(shape);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      size_(other.size_),
      dtype_(other.dtype_),
      allocator_(std::move(other.allocator_)),
      data_(other.data_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), dtype_(other.dtype_), owns_data_(true) {
    allocator_ = AllocatorFactory::create(DeviceType::CUDA);
    allocate();
    copy_from(other.data_, other.bytes());
}

Tensor::~Tensor() {
    deallocate();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        dtype_ = other.dtype_;
        allocator_ = std::move(other.allocator_);
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        
        shape_ = other.shape_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        owns_data_ = true;
        allocator_ = AllocatorFactory::create(DeviceType::CUDA);
        allocate();
        copy_from(other.data_, other.bytes());
    }
    return *this;
}

Tensor Tensor::zeros(const std::vector<int>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    CUDA_CHECK(cudaMemset(tensor.data_, 0, tensor.bytes()));
    return tensor;
}

Tensor Tensor::ones(const std::vector<int>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.fill(1.0f);
    return tensor;
}

Tensor Tensor::random_uniform(const std::vector<int>& shape, float min, float max) {
    Tensor tensor(shape, DataType::FP32);
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    
    curandGenerateUniform(gen, tensor.data<float>(), tensor.size());
    
    // Scale to [min, max]
    float range = max - min;
    int threads = 256;
    int blocks = (tensor.size() + threads - 1) / threads;
    
    auto scale_kernel = [min, range] __device__ (float x) { return x * range + min; };
    
    float* data = tensor.data<float>();
    // Apply scaling using a simple kernel
    cudaMemcpy(data, data, tensor.bytes(), cudaMemcpyDeviceToDevice);
    
    curandDestroyGenerator(gen);
    return tensor;
}

Tensor Tensor::random_normal(const std::vector<int>& shape, float mean, float std) {
    Tensor tensor(shape, DataType::FP32);
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    
    curandGenerateNormal(gen, tensor.data<float>(), tensor.size(), mean, std);
    
    curandDestroyGenerator(gen);
    return tensor;
}

Tensor Tensor::view(const std::vector<int>& new_shape) const {
    size_t new_size = compute_size(new_shape);
    if (new_size != size_) {
        throw std::runtime_error("View size mismatch");
    }
    
    Tensor result(new_shape, data_, dtype_);
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    return view(new_shape);
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
    // Simple implementation - can be optimized with cuBLAS
    Tensor result(shape_, dtype_);
    
    // For 2D case, use cuBLAS
    if (shape_.size() == 2 && axes == std::vector<int>{1, 0}) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   shape_[1], shape_[0],
                   &alpha, data<float>(), shape_[0],
                   &beta, nullptr, shape_[1],
                   result.data<float>(), shape_[1]);
        
        cublasDestroy(handle);
        
        result.shape_ = {shape_[1], shape_[0]};
    } else {
        // General case - implement transpose kernel
        throw std::runtime_error("General transpose not implemented yet");
    }
    
    return result;
}

Tensor Tensor::slice(int axis, int start, int end) const {
    if (axis < 0) axis += shape_.size();
    if (axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis for slice");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape[axis] = end - start;
    
    Tensor result(new_shape, dtype_);
    
    // Calculate strides
    std::vector<size_t> strides(shape_.size());
    strides.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_[i + 1];
    }
    
    // Copy slice
    size_t slice_size = result.size();
    size_t elem_size = dtype_size(dtype_);
    
    // Simple implementation for contiguous slice
    if (axis == 0) {
        size_t offset = start * strides[0] * elem_size;
        cudaMemcpy(result.data_, 
                  static_cast<char*>(data_) + offset,
                  result.bytes(),
                  cudaMemcpyDeviceToDevice);
    } else {
        // More complex slicing - implement kernel
        throw std::runtime_error("Non-axis-0 slice not implemented yet");
    }
    
    return result;
}

Tensor Tensor::to(DataType new_dtype) const {
    if (dtype_ == new_dtype) {
        return *this;
    }
    
    Tensor result(shape_, new_dtype);
    
    // Implement type conversion kernels
    size_t n = size_;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // For now, only implement common conversions
    if (dtype_ == DataType::FP32 && new_dtype == DataType::FP16) {
        // Launch FP32 to FP16 conversion kernel
        throw std::runtime_error("Type conversion not implemented yet");
    }
    
    return result;
}

Tensor Tensor::quantize_int8(float scale, int zero_point) const {
    if (dtype_ != DataType::FP32) {
        throw std::runtime_error("Can only quantize FP32 tensors");
    }
    
    Tensor result(shape_, DataType::INT8);
    
    // If scale not provided, compute it
    if (scale == 0.0f) {
        // Compute min/max
        float min_val, max_val;
        // Use cuBLAS or custom kernel to find min/max
        // For now, use symmetric quantization
        scale = 127.0f / std::max(std::abs(min_val), std::abs(max_val));
    }
    
    // Launch quantization kernel
    size_t n = size_;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // quantize_kernel<<<blocks, threads>>>(data<float>(), result.data<int8_t>(), 
    //                                      n, scale, zero_point);
    
    return result;
}

void Tensor::fill(float value) {
    if (dtype_ == DataType::FP32) {
        int threads = 256;
        int blocks = (size_ + threads - 1) / threads;
        // Launch fill kernel
        // fill_kernel<<<blocks, threads>>>(data<float>(), size_, value);
    } else {
        throw std::runtime_error("Fill only implemented for FP32");
    }
}

void Tensor::copy_from(const void* src, size_t bytes) {
    if (bytes > this->bytes()) {
        throw std::runtime_error("Copy size exceeds tensor size");
    }
    CUDA_CHECK(cudaMemcpy(data_, src, bytes, cudaMemcpyDeviceToDevice));
}

void Tensor::copy_to(void* dst, size_t bytes) const {
    if (bytes > this->bytes()) {
        throw std::runtime_error("Copy size exceeds tensor size");
    }
    CUDA_CHECK(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToDevice));
}

Tensor Tensor::clone() const {
    return Tensor(*this);
}

void Tensor::print(const std::string& name) const {
    std::cout << "Tensor";
    if (!name.empty()) {
        std::cout << " '" << name << "'";
    }
    std::cout << ": shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "], dtype=" << dtype_name(dtype_) << std::endl;
}

void Tensor::allocate() {
    if (size_ > 0 && owns_data_) {
        data_ = allocator_->allocate(bytes());
    }
}

void Tensor::deallocate() {
    if (data_ && owns_data_ && allocator_) {
        allocator_->deallocate(data_);
        data_ = nullptr;
    }
}

size_t Tensor::compute_size(const std::vector<int>& shape) const {
    return std::accumulate(shape.begin(), shape.end(), 
                          size_t(1), std::multiplies<size_t>());
}

// Tensor operations
Tensor cat(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    // Validate shapes
    const auto& first_shape = tensors[0].shape();
    int ndim = first_shape.size();
    if (axis < 0) axis += ndim;
    
    std::vector<int> result_shape = first_shape;
    result_shape[axis] = 0;
    
    for (const auto& t : tensors) {
        if (t.ndim() != ndim) {
            throw std::runtime_error("All tensors must have same number of dimensions");
        }
        for (int i = 0; i < ndim; ++i) {
            if (i != axis && t.shape()[i] != first_shape[i]) {
                throw std::runtime_error("All dimensions except cat axis must match");
            }
        }
        result_shape[axis] += t.shape()[i];
    }
    
    Tensor result(result_shape, tensors[0].dtype());
    
    // Copy data
    size_t offset = 0;
    for (const auto& t : tensors) {
        // Calculate copy size and perform copy
        // Implementation depends on axis
        // For now, implement axis=0 case
        if (axis == 0) {
            cudaMemcpy(static_cast<char*>(result.data()) + offset,
                      t.data(),
                      t.bytes(),
                      cudaMemcpyDeviceToDevice);
            offset += t.bytes();
        } else {
            throw std::runtime_error("Concatenation along non-zero axis not implemented");
        }
    }
    
    return result;
}

} // namespace deep_engine