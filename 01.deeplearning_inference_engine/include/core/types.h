#pragma once

#include <cstdint>
#include <string>

namespace deep_engine {

enum class DataType {
    FP32,
    FP16,
    INT8,
    INT32,
    UINT8
};

inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT32: return 4;
        case DataType::UINT8: return 1;
        default: return 0;
    }
}

inline std::string dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return "FP32";
        case DataType::FP16: return "FP16";
        case DataType::INT8: return "INT8";
        case DataType::INT32: return "INT32";
        case DataType::UINT8: return "UINT8";
        default: return "Unknown";
    }
}

// Activation types
enum class ActivationType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    GELU,
    SWISH,
    MISH,
    HARDSWISH,
    ELU,
    SELU
};

// Pooling types
enum class PoolingType {
    MAX,
    AVERAGE,
    GLOBAL_MAX,
    GLOBAL_AVERAGE
};

// Padding types
enum class PaddingType {
    VALID,
    SAME,
    CUSTOM
};

// Normalization types
enum class NormType {
    BATCH_NORM,
    LAYER_NORM,
    GROUP_NORM,
    INSTANCE_NORM
};

// Memory layout
enum class Layout {
    NCHW,  // Batch, Channel, Height, Width
    NHWC,  // Batch, Height, Width, Channel
    NC,    // Batch, Channel (for 1D)
    NCW,   // Batch, Channel, Width (for 1D conv)
    NCHWD  // Batch, Channel, Height, Width, Depth (for 3D)
};

// Device types
enum class DeviceType {
    CPU,
    CUDA,
    CUDNN,
    TENSORRT
};

// Precision modes
enum class PrecisionMode {
    FP32,
    FP16,
    INT8,
    MIXED
};

// Error codes
enum class ErrorCode {
    SUCCESS = 0,
    CUDA_ERROR,
    CUDNN_ERROR,
    ALLOCATION_ERROR,
    INVALID_ARGUMENT,
    NOT_IMPLEMENTED,
    SHAPE_MISMATCH,
    TYPE_MISMATCH,
    GRAPH_ERROR,
    IO_ERROR
};

// Result type for error handling
template<typename T>
class Result {
public:
    Result(T value) : value_(std::move(value)), error_(ErrorCode::SUCCESS) {}
    Result(ErrorCode error) : error_(error) {}
    
    bool is_ok() const { return error_ == ErrorCode::SUCCESS; }
    bool is_error() const { return error_ != ErrorCode::SUCCESS; }
    
    T& value() { return value_; }
    const T& value() const { return value_; }
    
    ErrorCode error() const { return error_; }
    
private:
    T value_;
    ErrorCode error_;
};

} // namespace deep_engine