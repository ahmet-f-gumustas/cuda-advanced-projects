#pragma once

#include "../core/layer.h"
#include "../core/tensor.h"
#include <cudnn.h>
#include <cublas_v2.h>

namespace deep_engine {

class DenseLayer : public Layer {
public:
    DenseLayer(int in_features, int out_features, bool use_bias = true,
               const std::string& name = "");
    
    ~DenseLayer();
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "Dense"; }
    
    size_t num_params() const override;
    size_t flops(const std::vector<int>& input_shape) const override;
    
    void save_params(std::ostream& os) const override;
    void load_params(std::istream& is) override;
    
    bool supports_quantization() const override { return true; }
    void quantize(int bits = 8) override;
    
    bool can_fuse_with(const Layer& next) const override;
    std::unique_ptr<Layer> fuse_with(const Layer& next) const override;
    
    // Getters
    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }
    bool use_bias() const { return use_bias_; }
    
    // Direct parameter access
    Tensor& weight() { return params_["weight"]; }
    const Tensor& weight() const { return params_.at("weight"); }
    
    Tensor& bias() { return params_["bias"]; }
    const Tensor& bias() const { return params_.at("bias"); }
    
protected:
    int in_features_;
    int out_features_;
    bool use_bias_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
};

// Aliases
using Linear = DenseLayer;
using FullyConnected = DenseLayer;
using FC = DenseLayer;

// Specialized dense layers with fused operations
class DenseReLU : public DenseLayer {
public:
    DenseReLU(int in_features, int out_features, bool use_bias = true,
              const std::string& name = "")
        : DenseLayer(in_features, out_features, use_bias, name) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "DenseReLU"; }
};

class DenseGeLU : public DenseLayer {
public:
    DenseGeLU(int in_features, int out_features, bool use_bias = true,
              const std::string& name = "")
        : DenseLayer(in_features, out_features, use_bias, name) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "DenseGeLU"; }
};

// Multi-layer perceptron
class MLP : public Layer {
public:
    MLP(int in_features, const std::vector<int>& hidden_sizes, int out_features,
        const std::string& activation = "relu", float dropout_rate = 0.0f,
        const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "MLP"; }
    
    size_t num_params() const override;
    
    void set_dropout_rate(float rate) { dropout_rate_ = rate; }
    
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    float dropout_rate_;
    bool training_;
};

// Embedding layer
class Embedding : public Layer {
public:
    Embedding(int num_embeddings, int embedding_dim,
              int padding_idx = -1, float max_norm = -1.0f,
              bool scale_grad_by_freq = false,
              const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "Embedding"; }
    
    size_t num_params() const override;
    
    // Direct parameter access
    Tensor& weight() { return params_["weight"]; }
    const Tensor& weight() const { return params_.at("weight"); }
    
    // Sparse gradient support
    void set_sparse_gradients(bool sparse) { sparse_gradients_ = sparse; }
    
protected:
    int num_embeddings_;
    int embedding_dim_;
    int padding_idx_;
    float max_norm_;
    bool scale_grad_by_freq_;
    bool sparse_gradients_;
};

} // namespace deep_engine