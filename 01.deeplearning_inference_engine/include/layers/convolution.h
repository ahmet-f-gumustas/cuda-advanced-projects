#pragma once

#include "../core/layer.h"
#include "../core/tensor.h"
#include <cudnn.h>

namespace deep_engine {

class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                     int stride = 1, int padding = 0, int dilation = 1,
                     int groups = 1, bool use_bias = true,
                     const std::string& name = "");
    
    ConvolutionLayer(int in_channels, int out_channels, 
                     const std::vector<int>& kernel_size,
                     const std::vector<int>& stride,
                     const std::vector<int>& padding,
                     const std::vector<int>& dilation,
                     int groups = 1, bool use_bias = true,
                     const std::string& name = "");
    
    ~ConvolutionLayer();
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "Convolution"; }
    
    size_t num_params() const override;
    size_t flops(const std::vector<int>& input_shape) const override;
    
    void save_params(std::ostream& os) const override;
    void load_params(std::istream& is) override;
    
    bool supports_quantization() const override { return true; }
    void quantize(int bits = 8) override;
    
    bool can_fuse_with(const Layer& next) const override;
    std::unique_ptr<Layer> fuse_with(const Layer& next) const override;
    
    // Getters
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    const std::vector<int>& kernel_size() const { return kernel_size_; }
    const std::vector<int>& stride() const { return stride_; }
    const std::vector<int>& padding() const { return padding_; }
    const std::vector<int>& dilation() const { return dilation_; }
    int groups() const { return groups_; }
    bool use_bias() const { return use_bias_; }
    
    // Direct parameter access
    Tensor& weight() { return params_["weight"]; }
    const Tensor& weight() const { return params_.at("weight"); }
    
    Tensor& bias() { return params_["bias"]; }
    const Tensor& bias() const { return params_.at("bias"); }
    
protected:
    int in_channels_;
    int out_channels_;
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    std::vector<int> dilation_;
    int groups_;
    bool use_bias_;
    int ndim_;
    
    // cuDNN handles
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnConvolutionFwdAlgo_t fwd_algo_;
    size_t workspace_size_;
    
    void init_cudnn();
    void find_best_algorithm(const std::vector<int>& input_shape);
};

// Specialized convolution layers
class Conv1d : public ConvolutionLayer {
public:
    Conv1d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, int dilation = 1,
           int groups = 1, bool use_bias = true,
           const std::string& name = "")
        : ConvolutionLayer(in_channels, out_channels, kernel_size,
                          stride, padding, dilation, groups, use_bias, name) {}
    
    std::string type() const override { return "Conv1d"; }
};

class Conv2d : public ConvolutionLayer {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, int dilation = 1,
           int groups = 1, bool use_bias = true,
           const std::string& name = "")
        : ConvolutionLayer(in_channels, out_channels, kernel_size,
                          stride, padding, dilation, groups, use_bias, name) {}
    
    std::string type() const override { return "Conv2d"; }
};

class Conv3d : public ConvolutionLayer {
public:
    Conv3d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, int dilation = 1,
           int groups = 1, bool use_bias = true,
           const std::string& name = "")
        : ConvolutionLayer(in_channels, out_channels, kernel_size,
                          stride, padding, dilation, groups, use_bias, name) {}
    
    std::string type() const override { return "Conv3d"; }
};

// Special convolution types
class DepthwiseConv2d : public Conv2d {
public:
    DepthwiseConv2d(int channels, int kernel_size,
                    int stride = 1, int padding = 0, int dilation = 1,
                    bool use_bias = true, const std::string& name = "")
        : Conv2d(channels, channels, kernel_size, stride, padding, dilation,
                channels, use_bias, name) {}
    
    std::string type() const override { return "DepthwiseConv2d"; }
};

class PointwiseConv2d : public Conv2d {
public:
    PointwiseConv2d(int in_channels, int out_channels,
                    bool use_bias = true, const std::string& name = "")
        : Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, use_bias, name) {}
    
    std::string type() const override { return "PointwiseConv2d"; }
};

class TransposedConv2d : public Layer {
public:
    TransposedConv2d(int in_channels, int out_channels, int kernel_size,
                     int stride = 1, int padding = 0, int output_padding = 0,
                     int dilation = 1, int groups = 1, bool use_bias = true,
                     const std::string& name = "");
    
    ~TransposedConv2d();
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "TransposedConv2d"; }
    
    size_t num_params() const override;
    size_t flops(const std::vector<int>& input_shape) const override;
    
private:
    int in_channels_;
    int out_channels_;
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    std::vector<int> output_padding_;
    std::vector<int> dilation_;
    int groups_;
    bool use_bias_;
    
    // cuDNN handles
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
};

} // namespace deep_engine