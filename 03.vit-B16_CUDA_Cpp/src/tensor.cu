#include "tensor.hpp"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <numeric>
#include <algorithm>

// Kernels
__global__ void fill_kernel_fp32(float* data, float value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void fill_kernel_fp16(__half* data, float value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = __float2half(value);
    }
}

__global__ void randn_kernel_fp32(float* data, size_t size, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_normal(&state);
    }
}

__global__ void randn_kernel_fp16(__half* data, size_t size, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = __float2half(curand_normal(&state));
    }
}

// Tensor implementation
Tensor::Tensor(const std::vector<int>& shape, DataType dtype) 
    : shape_(shape), dtype_(dtype) {
    compute_strides();
    allocate();
}

Tensor::Tensor(void* data, const std::vector<int>& shape, DataType dtype)
    : data_(data), shape_(shape), dtype_(dtype), owns_memory_(false) {
    compute_strides();
}

Tensor::~Tensor() {
    deallocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), 
      strides_(std::move(other.strides_)), size_(other.size_),
      dtype_(other.dtype_), owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.owns_memory_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        dtype_ = other.dtype_;
        owns_memory_ = other.owns_memory_;
        
        other.data_ = nullptr;
        other.owns_memory_ = false;
    }
    return *this;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_ = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = size_;
        size_ *= shape_[i];
    }
}

void Tensor::allocate() {
    if (size_ > 0) {
        data_ = aligned_alloc_cuda(nbytes());
        owns_memory_ = true;
    }
}

void Tensor::deallocate() {
    if (owns_memory_ && data_) {
        aligned_free_cuda(data_);
        data_ = nullptr;
    }
}

size_t Tensor::element_size() const {
    switch (dtype_) {
        case DataType::FP32: return sizeof(float);
        case DataType::FP16: return sizeof(__half);
        default: throw std::runtime_error("Unknown data type");
    }
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    size_t new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != size_) {
        throw std::runtime_error("Cannot reshape tensor: size mismatch");
    }
    shape_ = new_shape;
    compute_strides();
}

Tensor Tensor::view(const std::vector<int>& new_shape) const {
    Tensor result(data_, new_shape, dtype_);
    return result;
}

void Tensor::fill(float value) {
    const int threads = 256;
    const int blocks = (size_ + threads - 1) / threads;
    
    if (dtype_ == DataType::FP32) {
        fill_kernel_fp32<<<blocks, threads>>>(data_ptr<float>(), value, size_);
    } else {
        fill_kernel_fp16<<<blocks, threads>>>(data_ptr<__half>(), value, size_);
    }
    CUDA_CHECK(cudaGetLastError());
}

void Tensor::copy_from_host(const void* host_data) {
    CUDA_CHECK(cudaMemcpy(data_, host_data, nbytes(), cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(void* host_data) const {
    CUDA_CHECK(cudaMemcpy(host_data, data_, nbytes(), cudaMemcpyDeviceToHost));
}

Tensor Tensor::zeros(const std::vector<int>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.fill(0.0f);
    return tensor;
}

Tensor Tensor::ones(const std::vector<int>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.fill(1.0f);
    return tensor;
}

Tensor Tensor::randn(const std::vector<int>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    
    const int threads = 256;
    const int blocks = (tensor.size() + threads - 1) / threads;
    unsigned long long seed = 42;
    
    if (dtype == DataType::FP32) {
        randn_kernel_fp32<<<blocks, threads>>>(tensor.data_ptr<float>(), tensor.size(), seed);
    } else {
        randn_kernel_fp16<<<blocks, threads>>>(tensor.data_ptr<__half>(), tensor.size(), seed);
    }
    CUDA_CHECK(cudaGetLastError());
    
    return tensor;
}