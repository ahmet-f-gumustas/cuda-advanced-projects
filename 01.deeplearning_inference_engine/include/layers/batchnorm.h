#pragma once

#include "../core/layer.h"
#include "../core/tensor.h"
#include <cudnn.h>

namespace deep_engine {

class BatchNormalization : public Layer {
public:
    BatchNormalization(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                      bool affine = true, bool track_running_stats = true,
                      const std::string& name = "");
    
    ~BatchNormalization();
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "BatchNorm"; }
    
    size_t num_params() const override;
    size_t flops(const std::vector<int>& input_shape) const override;
    
    void save_params(std::ostream& os) const override;
    void load_params(std::istream& is) override;
    
    bool supports_quantization() const override { return true; }
    void quantize(int bits = 8) override;
    
    bool can_fuse_with(const Layer& next) const override;
    std::unique_ptr<Layer> fuse_with(const Layer& next) const override;
    
    // Mode control
    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }
    
    // Access to parameters
    const Tensor& running_mean() const { return params_.at("running_mean"); }
    const Tensor& running_var() const { return params_.at("running_var"); }
    const Tensor& weight() const { return params_.at("weight"); }
    const Tensor& bias() const { return params_.at("bias"); }
    
protected:
    int num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    bool track_running_stats_;
    bool training_;
    
    // cuDNN handles
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnTensorDescriptor_t data_desc_;
    cudnnBatchNormMode_t bn_mode_;
    
    void init_cudnn();
};

// Specialized normalization layers
class BatchNorm1d : public BatchNormalization {
public:
    BatchNorm1d(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true,
                const std::string& name = "")
        : BatchNormalization(num_features, eps, momentum, affine, 
                           track_running_stats, name) {}
    
    std::string type() const override { return "BatchNorm1d"; }
};

class BatchNorm2d : public BatchNormalization {
public:
    BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true,
                const std::string& name = "")
        : BatchNormalization(num_features, eps, momentum, affine, 
                           track_running_stats, name) {}
    
    std::string type() const override { return "BatchNorm2d"; }
};

class BatchNorm3d : public BatchNormalization {
public:
    BatchNorm3d(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true,
                const std::string& name = "")
        : BatchNormalization(num_features, eps, momentum, affine, 
                           track_running_stats, name) {}
    
    std::string type() const override { return "BatchNorm3d"; }
};

// Layer Normalization
class LayerNorm : public Layer {
public:
    LayerNorm(const std::vector<int>& normalized_shape, float eps = 1e-5f,
              bool elementwise_affine = true, const std::string& name = "");
    
    LayerNorm(int normalized_size, float eps = 1e-5f,
              bool elementwise_affine = true, const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "LayerNorm"; }
    
    size_t num_params() const override;
    
protected:
    std::vector<int> normalized_shape_;
    float eps_;
    bool elementwise_affine_;
};

// Group Normalization
class GroupNorm : public Layer {
public:
    GroupNorm(int num_groups, int num_channels, float eps = 1e-5f,
              bool affine = true, const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "GroupNorm"; }
    
    size_t num_params() const override;
    
protected:
    int num_groups_;
    int num_channels_;
    float eps_;
    bool affine_;
};

// Instance Normalization
class InstanceNorm : public Layer {
public:
    InstanceNorm(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                 bool affine = false, bool track_running_stats = false,
                 const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "InstanceNorm"; }
    
    size_t num_params() const override;
    
protected:
    int num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    bool track_running_stats_;
};

// Synchronized Batch Normalization (for multi-GPU)
class SyncBatchNorm : public BatchNormalization {
public:
    SyncBatchNorm(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                  bool affine = true, bool track_running_stats = true,
                  const std::string& name = "")
        : BatchNormalization(num_features, eps, momentum, affine,
                           track_running_stats, name) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "SyncBatchNorm"; }
    
private:
    void sync_stats_across_gpus();
};

} // namespace deep_engine