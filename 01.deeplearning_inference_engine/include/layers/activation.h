#pragma once

#include "../core/layer.h"
#include "../core/tensor.h"
#include "../core/types.h"
#include <cudnn.h>

namespace deep_engine {

class ActivationLayer : public Layer {
public:
    explicit ActivationLayer(ActivationType type, const std::string& name = "");
    explicit ActivationLayer(const std::string& type_str, const std::string& name = "");
    ~ActivationLayer();
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override;
    
    bool can_fuse_with(const Layer& next) const override;
    
protected:
    ActivationType activation_type_;
    float alpha_; // For LeakyReLU, ELU, etc.
    float beta_;  // For Swish, etc.
    
    // cuDNN handles
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t tensor_desc_;
    cudnnActivationDescriptor_t activation_desc_;
    
    void init_cudnn();
    ActivationType string_to_activation_type(const std::string& type_str);
};

// Specialized activation layers
class ReLU : public ActivationLayer {
public:
    explicit ReLU(const std::string& name = "") 
        : ActivationLayer(ActivationType::RELU, name) {}
    
    std::string type() const override { return "ReLU"; }
};

class LeakyReLU : public ActivationLayer {
public:
    explicit LeakyReLU(float negative_slope = 0.01f, const std::string& name = "")
        : ActivationLayer(ActivationType::LEAKY_RELU, name) {
        alpha_ = negative_slope;
    }
    
    std::string type() const override { return "LeakyReLU"; }
};

class Sigmoid : public ActivationLayer {
public:
    explicit Sigmoid(const std::string& name = "")
        : ActivationLayer(ActivationType::SIGMOID, name) {}
    
    std::string type() const override { return "Sigmoid"; }
};

class Tanh : public ActivationLayer {
public:
    explicit Tanh(const std::string& name = "")
        : ActivationLayer(ActivationType::TANH, name) {}
    
    std::string type() const override { return "Tanh"; }
};

class GELU : public ActivationLayer {
public:
    explicit GELU(const std::string& name = "")
        : ActivationLayer(ActivationType::GELU, name) {}
    
    std::string type() const override { return "GELU"; }
};

class Swish : public ActivationLayer {
public:
    explicit Swish(float beta = 1.0f, const std::string& name = "")
        : ActivationLayer(ActivationType::SWISH, name) {
        beta_ = beta;
    }
    
    std::string type() const override { return "Swish"; }
};

class Mish : public ActivationLayer {
public:
    explicit Mish(const std::string& name = "")
        : ActivationLayer(ActivationType::MISH, name) {}
    
    std::string type() const override { return "Mish"; }
};

class HardSwish : public ActivationLayer {
public:
    explicit HardSwish(const std::string& name = "")
        : ActivationLayer(ActivationType::HARDSWISH, name) {}
    
    std::string type() const override { return "HardSwish"; }
};

class ELU : public ActivationLayer {
public:
    explicit ELU(float alpha = 1.0f, const std::string& name = "")
        : ActivationLayer(ActivationType::ELU, name) {
        alpha_ = alpha;
    }
    
    std::string type() const override { return "ELU"; }
};

class SELU : public ActivationLayer {
public:
    explicit SELU(const std::string& name = "")
        : ActivationLayer(ActivationType::SELU, name) {
        alpha_ = 1.67326f;
        beta_ = 1.0507f;
    }
    
    std::string type() const override { return "SELU"; }
};

// Other activation functions
class Softmax : public Layer {
public:
    explicit Softmax(int axis = -1, const std::string& name = "")
        : Layer(name), axis_(axis) {
        cudnnCreate(&cudnn_handle_);
        cudnnCreateTensorDescriptor(&tensor_desc_);
    }
    
    ~Softmax() {
        cudnnDestroyTensorDescriptor(tensor_desc_);
        cudnnDestroy(cudnn_handle_);
    }
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "Softmax"; }
    
private:
    int axis_;
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t tensor_desc_;
};

class LogSoftmax : public Layer {
public:
    explicit LogSoftmax(int axis = -1, const std::string& name = "")
        : Layer(name), axis_(axis) {
        cudnnCreate(&cudnn_handle_);
        cudnnCreateTensorDescriptor(&tensor_desc_);
    }
    
    ~LogSoftmax() {
        cudnnDestroyTensorDescriptor(tensor_desc_);
        cudnnDestroy(cudnn_handle_);
    }
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "LogSoftmax"; }
    
private:
    int axis_;
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t tensor_desc_;
};

} // namespace deep_engine