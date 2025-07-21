#include "../../include/layers/convolution.h"
#include "../../include/kernels/conv_kernels.cuh"
#include "../../include/utils/logger.h"
#include <cudnn.h>
#include <algorithm>

namespace deep_engine {

ConvolutionLayer::ConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                                 int stride, int padding, int dilation,
                                 int groups, bool use_bias, const std::string& name)
    : Layer(name), in_channels_(in_channels), out_channels_(out_channels),
      groups_(groups), use_bias_(use_bias), ndim_(2) {
    kernel_size_ = {kernel_size, kernel_size};
    stride_ = {stride, stride};
    padding_ = {padding, padding};
    dilation_ = {dilation, dilation};
    
    // Initialize parameters
    int kernel_elements = in_channels / groups * kernel_size * kernel_size;
    params_["weight"] = Tensor({out_channels, in_channels / groups, kernel_size, kernel_size});
    
    // Xavier initialization
    float fan_in = in_channels / groups * kernel_size * kernel_size;
    float fan_out = out_channels / groups * kernel_size * kernel_size;
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    params_["weight"] = Tensor::random_uniform(params_["weight"].shape(), -scale, scale);
    
    if (use_bias_) {
        params_["bias"] = Tensor::zeros({out_channels});
    }
    
    trainable_ = true;
    init_cudnn();
}

ConvolutionLayer::ConvolutionLayer(int in_channels, int out_channels,
                                 const std::vector<int>& kernel_size,
                                 const std::vector<int>& stride,
                                 const std::vector<int>& padding,
                                 const std::vector<int>& dilation,
                                 int groups, bool use_bias, const std::string& name)
    : Layer(name), in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), groups_(groups), use_bias_(use_bias) {
    
    ndim_ = kernel_size.size();
    
    // Initialize weight shape based on dimensionality
    std::vector<int> weight_shape = {out_channels, in_channels / groups};
    weight_shape.insert(weight_shape.end(), kernel_size.begin(), kernel_size.end());
    
    params_["weight"] = Tensor(weight_shape);
    
    // Xavier initialization
    int kernel_elements = std::accumulate(kernel_size.begin(), kernel_size.end(), 
                                        in_channels / groups, std::multiplies<int>());
    float scale = std::sqrt(6.0f / (kernel_elements + out_channels * kernel_elements / in_channels));
    params_["weight"] = Tensor::random_uniform(weight_shape, -scale, scale);
    
    if (use_bias_) {
        params_["bias"] = Tensor::zeros({out_channels});
    }
    
    trainable_ = true;
    init_cudnn();
}

ConvolutionLayer::~ConvolutionLayer() {
    if (cudnn_handle_) cudnnDestroy(cudnn_handle_);
    if (input_desc_) cudnnDestroyTensorDescriptor(input_desc_);
    if (output_desc_) cudnnDestroyTensorDescriptor(output_desc_);
    if (bias_desc_) cudnnDestroyTensorDescriptor(bias_desc_);
    if (filter_desc_) cudnnDestroyFilterDescriptor(filter_desc_);
    if (conv_desc_) cudnnDestroyConvolutionDescriptor(conv_desc_);
}

void ConvolutionLayer::init_cudnn() {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    
    // Set convolution descriptor
    if (ndim_ == 2) {
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            padding_[0], padding_[1],
            stride_[0], stride_[1],
            dilation_[0], dilation_[1],
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        ));
    }
    
    // Set group count
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, groups_));
    
    // Set filter descriptor
    if (ndim_ == 2) {
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            filter_desc_,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            out_channels_,
            in_channels_ / groups_,
            kernel_size_[0],
            kernel_size_[1]
        ));
    }
}

Tensor ConvolutionLayer::forward(const Tensor& input, ExecutionContext& ctx) {
    PROFILE_LAYER(name_, "Convolution");
    
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
    int out_width = (width + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
    
    Tensor output({batch_size, out_channels_, out_height, out_width});
    
    // Set tensor descriptors
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in_channels_, height, width
    ));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, out_channels_, out_height, out_width
    ));
    
    // Find best algorithm if not already found
    if (workspace_size_ == 0) {
        find_best_algorithm(input_shape);
    }
    
    // Get workspace
    void* workspace = ctx.workspace(workspace_size_);
    
    // Perform convolution
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_handle_,
        &alpha,
        input_desc_, input.data(),
        filter_desc_, params_["weight"].data(),
        conv_desc_,
        fwd_algo_,
        workspace, workspace_size_,
        &beta,
        output_desc_, output.data()
    ));
    
    // Add bias if needed
    if (use_bias_) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, out_channels_, 1, 1
        ));
        
        CUDNN_CHECK(cudnnAddTensor(
            cudnn_handle_,
            &alpha,
            bias_desc_, params_["bias"].data(),
            &alpha,
            output_desc_, output.data()
        ));
    }
    
    return output;
}

void ConvolutionLayer::find_best_algorithm(const std::vector<int>& input_shape) {
    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];
    
    int out_height = (height + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
    int out_width = (width + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
    
    // Set descriptors for algorithm search
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in_channels_, height, width
    ));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, out_channels_, out_height, out_width
    ));
    
    // Find best algorithm
    int requested_algo_count = 8;
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle_,
        input_desc_, filter_desc_, conv_desc_, output_desc_,
        requested_algo_count, &returned_algo_count, perf_results
    ));
    
    // Choose the best algorithm that doesn't require too much workspace
    const size_t max_workspace = 512 * 1024 * 1024; // 512MB
    for (int i = 0; i < returned_algo_count; ++i) {
        if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
            perf_results[i].memory <= max_workspace) {
            fwd_algo_ = perf_results[i].algo;
            workspace_size_ = perf_results[i].memory;
            LOG_DEBUG("Selected convolution algorithm %d with workspace size %zu MB",
                     fwd_algo_, workspace_size_ / (1024 * 1024));
            break;
        }
    }
}

size_t ConvolutionLayer::num_params() const {
    size_t total = params_.at("weight").size();
    if (use_bias_) {
        total += params_.at("bias").size();
    }
    return total;
}

size_t ConvolutionLayer::flops(const std::vector<int>& input_shape) const {
    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];
    
    int out_height = (height + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
    int out_width = (width + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
    
    size_t kernel_ops = in_channels_ / groups_ * kernel_size_[0] * kernel_size_[1];
    size_t output_size = batch_size * out_channels_ * out_height * out_width;
    
    // Multiply-accumulate operations
    size_t mac_ops = output_size * kernel_ops;
    
    // Each MAC is 2 FLOPs
    return 2 * mac_ops;
}

void ConvolutionLayer::save_params(std::ostream& os) const {
    // Save layer configuration
    os.write(reinterpret_cast<const char*>(&in_channels_), sizeof(in_channels_));
    os.write(reinterpret_cast<const char*>(&out_channels_), sizeof(out_channels_));
    os.write(reinterpret_cast<const char*>(&groups_), sizeof(groups_));
    os.write(reinterpret_cast<const char*>(&use_bias_), sizeof(use_bias_));
    
    // Save kernel size, stride, padding, dilation
    int ndim = kernel_size_.size();
    os.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    os.write(reinterpret_cast<const char*>(kernel_size_.data()), ndim * sizeof(int));
    os.write(reinterpret_cast<const char*>(stride_.data()), ndim * sizeof(int));
    os.write(reinterpret_cast<const char*>(padding_.data()), ndim * sizeof(int));
    os.write(reinterpret_cast<const char*>(dilation_.data()), ndim * sizeof(int));
    
    // Save weight and bias tensors
    // TODO: Implement tensor serialization
}

void ConvolutionLayer::load_params(std::istream& is) {
    // Load layer configuration
    is.read(reinterpret_cast<char*>(&in_channels_), sizeof(in_channels_));
    is.read(reinterpret_cast<char*>(&out_channels_), sizeof(out_channels_));
    is.read(reinterpret_cast<char*>(&groups_), sizeof(groups_));
    is.read(reinterpret_cast<char*>(&use_bias_), sizeof(use_bias_));
    
    // Load kernel size, stride, padding, dilation
    int ndim;
    is.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    kernel_size_.resize(ndim);
    stride_.resize(ndim);
    padding_.resize(ndim);
    dilation_.resize(ndim);
    
    is.read(reinterpret_cast<char*>(kernel_size_.data()), ndim * sizeof(int));
    is.read(reinterpret_cast<char*>(stride_.data()), ndim * sizeof(int));
    is.read(reinterpret_cast<char*>(padding_.data()), ndim * sizeof(int));
    is.read(reinterpret_cast<char*>(dilation_.data()), ndim * sizeof(int));
    
    // Load weight and bias tensors
    // TODO: Implement tensor deserialization
}

void ConvolutionLayer::quantize(int bits) {
    // Quantize weights
    auto weight_quant = params_["weight"].quantize_int8();
    params_["weight_int8"] = weight_quant;
    
    if (use_bias_) {
        // Bias typically stays in higher precision
        params_["bias_scale"] = Tensor({1});
        // Calculate bias scale based on input and weight scales
    }
}

bool ConvolutionLayer::can_fuse_with(const Layer& next) const {
    // Can fuse with BatchNorm or Activation
    return next.type() == "BatchNorm" || 
           next.type() == "ReLU" || 
           next.type() == "BatchNorm2d" ||
           next.type() == "ReLU";
}

std::unique_ptr<Layer> ConvolutionLayer::fuse_with(const Layer& next) const {
    // TODO: Implement layer fusion
    return nullptr;
}

} // namespace deep_engine