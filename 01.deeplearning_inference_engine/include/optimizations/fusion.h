#pragma once

#include "../core/layer.h"
#include "../core/graph.h"
#include <memory>
#include <vector>

namespace deep_engine {

// Fused layer types
class ConvBNReLU : public Layer {
public:
    ConvBNReLU(std::shared_ptr<ConvolutionLayer> conv,
               std::shared_ptr<BatchNormalization> bn,
               std::shared_ptr<ActivationLayer> relu);
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "ConvBNReLU"; }
    
    size_t num_params() const override;
    size_t flops(const std::vector<int>& input_shape) const override;
    
    bool supports_quantization() const override { return true; }
    void quantize(int bits = 8) override;
    
private:
    std::shared_ptr<ConvolutionLayer> conv_;
    std::shared_ptr<BatchNormalization> bn_;
    std::shared_ptr<ActivationLayer> relu_;
    
    // Fused parameters
    Tensor fused_weight_;
    Tensor fused_bias_;
    bool parameters_fused_ = false;
    
    void fuse_parameters();
};

class ConvBN : public Layer {
public:
    ConvBN(std::shared_ptr<ConvolutionLayer> conv,
           std::shared_ptr<BatchNormalization> bn);
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "ConvBN"; }
    
    size_t num_params() const override;
    size_t flops(const std::vector<int>& input_shape) const override;
    
private:
    std::shared_ptr<ConvolutionLayer> conv_;
    std::shared_ptr<BatchNormalization> bn_;
    
    Tensor fused_weight_;
    Tensor fused_bias_;
    bool parameters_fused_ = false;
    
    void fuse_parameters();
};

class LinearReLU : public Layer {
public:
    LinearReLU(std::shared_ptr<DenseLayer> linear,
               std::shared_ptr<ActivationLayer> relu);
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "LinearReLU"; }
    
    size_t num_params() const override { return linear_->num_params(); }
    
private:
    std::shared_ptr<DenseLayer> linear_;
    std::shared_ptr<ActivationLayer> relu_;
};

// Multi-head attention fusion
class FusedMultiHeadAttention : public Layer {
public:
    FusedMultiHeadAttention(int embed_dim, int num_heads,
                           float dropout = 0.0f,
                           bool bias = true,
                           const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs,
                               ExecutionContext& ctx) override;
    std::string type() const override { return "FusedMultiHeadAttention"; }
    
    size_t num_params() const override;
    
private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_;
    bool bias_;
    
    // Fused QKV projection
    std::unique_ptr<DenseLayer> qkv_proj_;
    std::unique_ptr<DenseLayer> out_proj_;
};

// Layer fusion patterns
struct FusionPattern {
    std::vector<std::string> pattern;  // e.g., ["Conv2d", "BatchNorm2d", "ReLU"]
    std::function<std::unique_ptr<Layer>(const std::vector<std::shared_ptr<Layer>>&)> fuser;
};

// Fusion optimizer
class FusionOptimizer : public GraphOptimizer {
public:
    FusionOptimizer();
    
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "FusionOptimizer"; }
    
    // Add custom fusion pattern
    void add_pattern(const FusionPattern& pattern);
    
private:
    std::vector<FusionPattern> patterns_;
    
    void initialize_default_patterns();
    bool match_pattern(const ComputationGraph& graph,
                      ComputationGraph::NodeId start_node,
                      const std::vector<std::string>& pattern);
    std::vector<ComputationGraph::NodeId> 
    get_pattern_nodes(const ComputationGraph& graph,
                     ComputationGraph::NodeId start_node,
                     const std::vector<std::string>& pattern);
};

// Horizontal fusion (parallel ops)
class HorizontalFusion : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "HorizontalFusion"; }
    
private:
    bool can_fuse_horizontally(const Layer& layer1, const Layer& layer2);
    std::unique_ptr<Layer> fuse_horizontally(const std::vector<std::shared_ptr<Layer>>& layers);
};

// Vertical fusion (sequential ops)
class VerticalFusion : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "VerticalFusion"; }
    
private:
    bool can_fuse_vertically(const Layer& layer1, const Layer& layer2);
    std::unique_ptr<Layer> fuse_vertically(const Layer& layer1, const Layer& layer2);
};

// Element-wise operation fusion
class ElementwiseFusion : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "ElementwiseFusion"; }
};

// Memory-bound operation fusion
class MemoryBoundFusion : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "MemoryBoundFusion"; }
};

} // namespace deep_engine