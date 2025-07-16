#pragma once

#include "../core/layer.h"
#include <cudnn.h>

namespace deep_engine {

// Convolution parametreleri
struct ConvParams {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    int groups;
    bool use_bias;
    
    ConvParams() : in_channels(1), out_channels(1), kernel_h(3), kernel_w(3),
                   stride_h(1), stride_w(1), pad_h(1), pad_w(1),
                   dilation_h(1), dilation_w(1), groups(1), use_bias(true) {}
};

class ConvolutionLayer : public Layer {
private:
    ConvParams params_;
    
    // cuDNN descriptors
    cudnnFilterDescriptor_t weight_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    
    // Algorithm selection - cuDNN'in auto-tuning'i için
    mutable cudnnConvolutionFwdAlgo_t fwd_algo_;
    mutable bool algo_selected_;
    
    // Winograd optimization flag
    bool use_winograd_;
    
    // Im2col buffer for custom kernels
    mutable Tensor im2col_buffer_;
    
    void create_descriptors();
    void destroy_descriptors();
    void select_algorithm(const Tensor& input, const Tensor& output, ExecutionContext& ctx) const;
    
    // Custom CUDA kernels for special cases
    void conv_1x1_optimized(const Tensor& input, Tensor& output, ExecutionContext& ctx) const;
    void depthwise_conv_optimized(const Tensor& input, Tensor& output, ExecutionContext& ctx) const;
    void conv_3x3_winograd(const Tensor& input, Tensor& output, ExecutionContext& ctx) const;
    
public:
    explicit ConvolutionLayer(const LayerConfig& config);
    ConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                     int stride = 1, int padding = 0, bool use_bias = true);
    ~ConvolutionLayer();
    
    // Layer interface implementation
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::vector<int> infer_output_shape(const std::vector<int>& input_shape) const override;
    std::string type() const override { return "Convolution"; }
    
    // Optimization methods
    bool can_fuse_with(const Layer* next) const override;
    bool supports_int8() const override { return true; }
    
    // INT8 inference için özel forward
    Tensor forward_int8(const QuantizedTensor& input, ExecutionContext& ctx);
    
    // Weight initialization
    void init_weights(const std::string& method = "xavier");
    
    // Convolution-specific optimizations
    void enable_winograd(bool enable) { use_winograd_ = enable; }
    bool is_depthwise() const { return params_.groups == params_.in_channels; }
    bool is_pointwise() const { return params_.kernel_h == 1 && params_.kernel_w == 1; }
    
    // Static method for im2col (diğer layer'lar da kullanabilir)
    static void im2col_gpu(const float* data_im, int channels,
                          int height, int width, int kernel_h, int kernel_w,
                          int pad_h, int pad_w, int stride_h, int stride_w,
                          int dilation_h, int dilation_w, float* data_col);
};

// Fused Conv+BatchNorm+ReLU layer
class ConvBNReLU : public Layer {
private:
    std::unique_ptr<ConvolutionLayer> conv_;
    Tensor bn_scale_;
    Tensor bn_bias_;
    Tensor bn_mean_;
    Tensor bn_var_;
    float epsilon_;
    
public:
    ConvBNReLU(const ConvParams& conv_params, int num_features, float eps = 1e-5);
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::vector<int> infer_output_shape(const std::vector<int>& input_shape) const override;
    std::string type() const override { return "ConvBNReLU"; }
    
    // Fusion from separate layers
    static std::unique_ptr<ConvBNReLU> fuse(ConvolutionLayer* conv, 
                                            BatchNormLayer* bn, 
                                            ActivationLayer* relu);
};

// Specialized convolution variants
class DepthwiseConvolution : public ConvolutionLayer {
public:
    DepthwiseConvolution(int channels, int kernel_size, int stride = 1, int padding = 0);
    std::string type() const override { return "DepthwiseConvolution"; }
};

class PointwiseConvolution : public ConvolutionLayer {
public:
    PointwiseConvolution(int in_channels, int out_channels, bool use_bias = true);
    std::string type() const override { return "PointwiseConvolution"; }
};

// MobileNet-style separable convolution
class SeparableConvolution : public Layer {
private:
    std::unique_ptr<DepthwiseConvolution> depthwise_;
    std::unique_ptr<PointwiseConvolution> pointwise_;
    
public:
    SeparableConvolution(int in_channels, int out_channels, int kernel_size,
                        int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::vector<int> infer_output_shape(const std::vector<int>& input_shape) const override;
    std::string type() const override { return "SeparableConvolution"; }
};

} // namespace deep_engine