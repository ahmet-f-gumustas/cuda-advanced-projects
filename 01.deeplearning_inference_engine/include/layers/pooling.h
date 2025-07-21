#pragma once

#include "../core/layer.h"
#include "../core/tensor.h"
#include "../core/types.h"
#include <cudnn.h>

namespace deep_engine {

class PoolingLayer : public Layer {
public:
    PoolingLayer(PoolingType type, int kernel_size, int stride = -1,
                 int padding = 0, const std::string& name = "");
    
    PoolingLayer(PoolingType type, const std::vector<int>& kernel_size,
                 const std::vector<int>& stride,
                 const std::vector<int>& padding,
                 const std::string& name = "");
    
    ~PoolingLayer();
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override;
    
    size_t flops(const std::vector<int>& input_shape) const override;
    
protected:
    PoolingType pool_type_;
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    
    // cuDNN handles
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnPoolingDescriptor_t pooling_desc_;
    
    void init_cudnn();
};

// Specialized pooling layers
class MaxPool1d : public PoolingLayer {
public:
    MaxPool1d(int kernel_size, int stride = -1, int padding = 0,
              const std::string& name = "")
        : PoolingLayer(PoolingType::MAX, kernel_size, stride, padding, name) {}
    
    std::string type() const override { return "MaxPool1d"; }
};

class MaxPool2d : public PoolingLayer {
public:
    MaxPool2d(int kernel_size, int stride = -1, int padding = 0,
              const std::string& name = "")
        : PoolingLayer(PoolingType::MAX, kernel_size, stride, padding, name) {}
    
    std::string type() const override { return "MaxPool2d"; }
};

class MaxPool3d : public PoolingLayer {
public:
    MaxPool3d(int kernel_size, int stride = -1, int padding = 0,
              const std::string& name = "")
        : PoolingLayer(PoolingType::MAX, kernel_size, stride, padding, name) {}
    
    std::string type() const override { return "MaxPool3d"; }
};

class AvgPool1d : public PoolingLayer {
public:
    AvgPool1d(int kernel_size, int stride = -1, int padding = 0,
              const std::string& name = "")
        : PoolingLayer(PoolingType::AVERAGE, kernel_size, stride, padding, name) {}
    
    std::string type() const override { return "AvgPool1d"; }
};

class AvgPool2d : public PoolingLayer {
public:
    AvgPool2d(int kernel_size, int stride = -1, int padding = 0,
              const std::string& name = "")
        : PoolingLayer(PoolingType::AVERAGE, kernel_size, stride, padding, name) {}
    
    std::string type() const override { return "AvgPool2d"; }
};

class AvgPool3d : public PoolingLayer {
public:
    AvgPool3d(int kernel_size, int stride = -1, int padding = 0,
              const std::string& name = "")
        : PoolingLayer(PoolingType::AVERAGE, kernel_size, stride, padding, name) {}
    
    std::string type() const override { return "AvgPool3d"; }
};

// Global pooling layers
class GlobalMaxPool : public Layer {
public:
    explicit GlobalMaxPool(const std::string& name = "") : Layer(name) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "GlobalMaxPool"; }
};

class GlobalAvgPool : public Layer {
public:
    explicit GlobalAvgPool(const std::string& name = "") : Layer(name) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "GlobalAvgPool"; }
};

// Adaptive pooling layers
class AdaptiveAvgPool2d : public Layer {
public:
    explicit AdaptiveAvgPool2d(const std::vector<int>& output_size,
                              const std::string& name = "")
        : Layer(name), output_size_(output_size) {}
    
    explicit AdaptiveAvgPool2d(int output_size, const std::string& name = "")
        : Layer(name), output_size_({output_size, output_size}) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "AdaptiveAvgPool2d"; }
    
private:
    std::vector<int> output_size_;
};

class AdaptiveMaxPool2d : public Layer {
public:
    explicit AdaptiveMaxPool2d(const std::vector<int>& output_size,
                              const std::string& name = "")
        : Layer(name), output_size_(output_size) {}
    
    explicit AdaptiveMaxPool2d(int output_size, const std::string& name = "")
        : Layer(name), output_size_({output_size, output_size}) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "AdaptiveMaxPool2d"; }
    
private:
    std::vector<int> output_size_;
};

} // namespace deep_engine