#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>
#include <vector>
#include <cassert>
#include <initializer_list>

namespace deep_engine {

enum class DataType {
    FP32,
    FP16,
    INT8,
    INT32
};

enum class TensorFormat {
    NCHW,
    NHWC,
    NC,
    NCHW_VECT_C  // Tensor Core optimized format
};

// Memory layout descriptor
struct TensorDescriptor {
    std::vector<int> dims;
    DataType dtype;
    TensorFormat format;
    size_t stride[4];  // Strides for each dimension
    
    size_t total_elements() const {
        size_t total = 1;
        for (auto d : dims) total *= d;
        return total;
    }
    
    size_t bytes() const {
        size_t elem_size = 4; // default FP32
        switch (dtype) {
            case DataType::FP16: elem_size = 2; break;
            case DataType::INT8: elem_size = 1; break;
            case DataType::INT32: elem_size = 4; break;
            default: break;
        }
        return total_elements() * elem_size;
    }
};

// Forward declarations
class MemoryPool;
class CudnnHandle;

class Tensor {
private:
    TensorDescriptor desc_;
    void* device_ptr_;
    void* host_ptr_;
    bool owns_memory_;
    std::shared_ptr<MemoryPool> memory_pool_;
    cudnnTensorDescriptor_t cudnn_desc_;
    
    // Private constructor for zero-copy view creation
    Tensor(void* ptr, const TensorDescriptor& desc, bool owns = false)
        : device_ptr_(ptr), desc_(desc), owns_memory_(owns), host_ptr_(nullptr) {
        create_cudnn_descriptor();
    }
    
    void create_cudnn_descriptor();
    void destroy_cudnn_descriptor();
    
public:
    Tensor() : device_ptr_(nullptr), host_ptr_(nullptr), owns_memory_(false), cudnn_desc_(nullptr) {}
    
    // Standard constructor with shape
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::FP32, 
           TensorFormat format = TensorFormat::NCHW);
    
    // Constructor with memory pool
    Tensor(const std::vector<int>& shape, std::shared_ptr<MemoryPool> pool,
           DataType dtype = DataType::FP32, TensorFormat format = TensorFormat::NCHW);
    
    // Move semantics - kripto paralarda olduğu gibi ownership transfer önemli
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Copy'yi engelleyip explicit clone fonksiyonu kullanmak daha güvenli
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    ~Tensor();
    
    // Deep copy with new allocation
    Tensor clone() const;
    
    // Zero-copy view creation (numpy'daki gibi ama GPU memory için)
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor transpose(int dim1, int dim2) const;
    
    // Tensor Core optimizasyonu için format dönüşümü
    Tensor to_tensor_core_format() const;
    
    // Data transfer operations
    void copy_from_host(const void* src);
    void copy_to_host(void* dst) const;
    void* data() { return device_ptr_; }
    const void* data() const { return device_ptr_; }
    
    // Async operations with streams
    void copy_from_host_async(const void* src, cudaStream_t stream);
    void copy_to_host_async(void* dst, cudaStream_t stream) const;
    
    // Lazy allocation - memory'yi gerçekten kullanana kadar allocate etme
    void allocate_if_needed();
    
    // Type conversion
    Tensor to(DataType new_dtype) const;
    
    // Quantization helpers
    std::pair<Tensor, Tensor> quantize_int8() const;  // Returns quantized tensor + scale factors
    
    // Accessor methods
    const TensorDescriptor& descriptor() const { return desc_; }
    cudnnTensorDescriptor_t cudnn_descriptor() const { return cudnn_desc_; }
    bool is_contiguous() const;
    size_t bytes() const { return desc_.bytes(); }
    
    // Debug utilities
    void print_summary() const;
    void dump_to_file(const std::string& filename) const;
    
    // Static factory methods
    static Tensor zeros(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    static Tensor ones(const std::vector<int>& shape, DataType dtype = DataType::FP32);
    static Tensor random_uniform(const std::vector<int>& shape, float min = 0.0f, float max = 1.0f);
    static Tensor from_host(const void* data, const std::vector<int>& shape, DataType dtype = DataType::FP32);
};

// Specialized tensor for INT8 quantized operations
class QuantizedTensor : public Tensor {
private:
    Tensor scale_;
    Tensor zero_point_;
    
public:
    QuantizedTensor(const Tensor& quantized_data, const Tensor& scale, const Tensor& zero_point)
        : Tensor(std::move(const_cast<Tensor&>(quantized_data))), 
          scale_(scale), zero_point_(zero_point) {}
    
    Tensor dequantize() const;
    const Tensor& scale() const { return scale_; }
    const Tensor& zero_point() const { return zero_point_; }
};

} // namespace deep_engine