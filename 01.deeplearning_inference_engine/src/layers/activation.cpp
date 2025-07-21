#include "../../include/layers/activation.h"
#include "../../include/kernels/activation_kernels.cuh"
#include "../../include/utils/logger.h"
#include <cudnn.h>

namespace deep_engine {

ActivationLayer::ActivationLayer(ActivationType type, const std::string& name)
    : Layer(name), activation_type_(type), alpha_(0.0f), beta_(0.0f) {
    init_cudnn();
    
    // Set default parameters for specific activations
    switch (activation_type_) {
        case ActivationType::LEAKY_RELU:
            alpha_ = 0.01f; // Default negative slope
            break;
        case ActivationType::ELU:
            alpha_ = 1.0f;
            break;
        case ActivationType::SELU:
            alpha_ = 1.67326f;
            beta_ = 1.0507f;
            break;
        default:
            break;
    }
}

ActivationLayer::ActivationLayer(const std::string& type_str, const std::string& name)
    : Layer(name) {
    activation_type_ = string_to_activation_type(type_str);
    init_cudnn();
}

ActivationLayer::~ActivationLayer() {
    if (cudnn_handle_) cudnnDestroy(cudnn_handle_);
    if (tensor_desc_) cudnnDestroyTensorDescriptor(tensor_desc_);
    if (activation_desc_) cudnnDestroyActivationDescriptor(activation_desc_);
}

void ActivationLayer::init_cudnn() {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc_));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc_));
    
    // Map our activation type to cuDNN activation type
    cudnnActivationMode_t cudnn_mode;
    switch (activation_type_) {
        case ActivationType::RELU:
            cudnn_mode = CUDNN_ACTIVATION_RELU;
            break;
        case ActivationType::SIGMOID:
            cudnn_mode = CUDNN_ACTIVATION_SIGMOID;
            break;
        case ActivationType::TANH:
            cudnn_mode = CUDNN_ACTIVATION_TANH;
            break;
        case ActivationType::ELU:
            cudnn_mode = CUDNN_ACTIVATION_ELU;
            break;
        default:
            // For activations not supported by cuDNN, we'll use custom kernels
            cudnn_mode = CUDNN_ACTIVATION_IDENTITY;
            break;
    }
    
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        activation_desc_,
        cudnn_mode,
        CUDNN_NOT_PROPAGATE_NAN,
        alpha_
    ));
}

Tensor ActivationLayer::forward(const Tensor& input, ExecutionContext& ctx) {
    PROFILE_LAYER(name_, type());
    
    Tensor output(input.shape(), input.dtype());
    const auto& shape = input.shape();
    
    // For activations supported by cuDNN
    if (activation_type_ == ActivationType::RELU ||
        activation_type_ == ActivationType::SIGMOID ||
        activation_type_ == ActivationType::TANH ||
        activation_type_ == ActivationType::ELU) {
        
        // Set tensor descriptor
        if (shape.size() == 4) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                shape[0], shape[1], shape[2], shape[3]
            ));
        } else if (shape.size() == 2) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                shape[0], shape[1], 1, 1
            ));
        }
        
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnActivationForward(
            cudnn_handle_,
            activation_desc_,
            &alpha,
            tensor_desc_, input.data(),
            &beta,
            tensor_desc_, output.data()
        ));
    } else {
        // Use custom kernels for other activations
        size_t size = input.size();
        
        switch (activation_type_) {
            case ActivationType::LEAKY_RELU:
                kernels::launch_leaky_relu(
                    input.data<float>(), output.data<float>(), 
                    alpha_, size, ctx.stream()
                );
                break;
            case ActivationType::GELU:
                kernels::launch_gelu(
                    input.data<float>(), output.data<float>(), 
                    size, ctx.stream()
                );
                break;
            case ActivationType::SWISH:
                kernels::launch_swish(
                    input.data<float>(), output.data<float>(), 
                    beta_, size, ctx.stream()
                );
                break;
            case ActivationType::MISH:
                kernels::launch_activation(
                    input.data<float>(), output.data<float>(), size,
                    kernels::ActivationKernelType::MISH, 0.0f, 0.0f,
                    ctx.stream()
                );
                break;
            case ActivationType::HARDSWISH:
                kernels::launch_activation(
                    input.data<float>(), output.data<float>(), size,
                    kernels::ActivationKernelType::HARDSWISH, 0.0f, 0.0f,
                    ctx.stream()
                );
                break;
            case ActivationType::SELU:
                kernels::launch_activation(
                    input.data<float>(), output.data<float>(), size,
                    kernels::ActivationKernelType::SELU, alpha_, beta_,
                    ctx.stream()
                );
                break;
            default:
                throw std::runtime_error("Unsupported activation type");
        }
    }
    
    return output;
}

std::string ActivationLayer::type() const {
    switch (activation_type_) {
        case ActivationType::RELU: return "ReLU";
        case ActivationType::LEAKY_RELU: return "LeakyReLU";
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::GELU: return "GELU";
        case ActivationType::SWISH: return "Swish";
        case ActivationType::MISH: return "Mish";
        case ActivationType::HARDSWISH: return "HardSwish";
        case ActivationType::ELU: return "ELU";
        case ActivationType::SELU: return "SELU";
        default: return "Unknown";
    }
}

ActivationType ActivationLayer::string_to_activation_type(const std::string& type_str) {
    if (type_str == "relu" || type_str == "ReLU") return ActivationType::RELU;
    if (type_str == "leaky_relu" || type_str == "LeakyReLU") return ActivationType::LEAKY_RELU;
    if (type_str == "sigmoid" || type_str == "Sigmoid") return ActivationType::SIGMOID;
    if (type_str == "tanh" || type_str == "Tanh") return ActivationType::TANH;
    if (type_str == "gelu" || type_str == "GELU") return ActivationType::GELU;
    if (type_str == "swish" || type_str == "Swish") return ActivationType::SWISH;
    if (type_str == "mish" || type_str == "Mish") return ActivationType::MISH;
    if (type_str == "hardswish" || type_str == "HardSwish") return ActivationType::HARDSWISH;
    if (type_str == "elu" || type_str == "ELU") return ActivationType::ELU;
    if (type_str == "selu" || type_str == "SELU") return ActivationType::SELU;
    
    throw std::runtime_error("Unknown activation type: " + type_str);
}

bool ActivationLayer::can_fuse_with(const Layer& next) const {
    // Activation layers typically don't fuse with following layers
    // But they can be fused into previous conv/linear layers
    return false;
}

// Softmax implementation
Tensor Softmax::forward(const Tensor& input, ExecutionContext& ctx) {
    PROFILE_LAYER(name_, "Softmax");
    
    Tensor output(input.shape(), input.dtype());
    const auto& shape = input.shape();
    
    // Handle different tensor dimensions
    if (shape.size() == 2) {
        // Batch x Classes
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            shape[0], shape[1], 1, 1
        ));
        
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(
            cudnn_handle_,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            tensor_desc_, input.data(),
            &beta,
            tensor_desc_, output.data()
        ));
    } else if (shape.size() == 4) {
        // For 4D tensors, apply along the channel dimension
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            shape[0], shape[1], shape[2], shape[3]
        ));
        
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(
            cudnn_handle_,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            tensor_desc_, input.data(),
            &beta,
            tensor_desc_, output.data()
        ));
    } else {
        // Use custom kernel for other dimensions
        int batch_size = 1;
        for (int i = 0; i < axis_; ++i) {
            batch_size *= shape[i];
        }
        int classes = shape[axis_];
        
        kernels::launch_softmax(
            input.data<float>(), output.data<float>(),
            batch_size, classes, ctx.stream()
        );
    }
    
    return output;
}

// LogSoftmax implementation
Tensor LogSoftmax::forward(const Tensor& input, ExecutionContext& ctx) {
    PROFILE_LAYER(name_, "LogSoftmax");
    
    Tensor output(input.shape(), input.dtype());
    const auto& shape = input.shape();
    
    if (shape.size() == 2 || shape.size() == 4) {
        // Set tensor descriptor
        if (shape.size() == 2) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                shape[0], shape[1], 1, 1
            ));
        } else {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                shape[0], shape[1], shape[2], shape[3]
            ));
        }
        
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(
            cudnn_handle_,
            CUDNN_SOFTMAX_LOG,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            tensor_desc_, input.data(),
            &beta,
            tensor_desc_, output.data()
        ));
    } else {
        throw std::runtime_error("LogSoftmax only supports 2D and 4D tensors currently");
    }
    
    return output;
}

} // namespace deep_engine