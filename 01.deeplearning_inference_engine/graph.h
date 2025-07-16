#pragma once

#include "tensor.h"
#include "layer.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <functional>

namespace deep_engine {

// Forward declarations
class GraphOptimizer;
class MemoryPlanner;

// Node in the computation graph
class GraphNode {
public:
    using NodeId = size_t;
    
private:
    NodeId id_;
    std::string name_;
    std::shared_ptr<Layer> layer_;
    std::vector<NodeId> inputs_;
    std::vector<NodeId> outputs_;
    
    // For topological sort
    mutable int in_degree_;
    
    // Memory planning info
    size_t workspace_size_;
    size_t output_size_;
    
    // Profiling data
    mutable double avg_time_ms_;
    mutable int num_runs_;
    
public:
    GraphNode(NodeId id, const std::string& name, std::shared_ptr<Layer> layer)
        : id_(id), name_(name), layer_(layer), in_degree_(0), 
          workspace_size_(0), output_size_(0), avg_time_ms_(0), num_runs_(0) {}
    
    NodeId id() const { return id_; }
    const std::string& name() const { return name_; }
    std::shared_ptr<Layer> layer() { return layer_; }
    const std::shared_ptr<Layer> layer() const { return layer_; }
    
    void add_input(NodeId input) { inputs_.push_back(input); }
    void add_output(NodeId output) { outputs_.push_back(output); }
    
    const std::vector<NodeId>& inputs() const { return inputs_; }
    const std::vector<NodeId>& outputs() const { return outputs_; }
    
    // For graph algorithms
    void set_in_degree(int degree) const { in_degree_ = degree; }
    int in_degree() const { return in_degree_; }
    
    void update_profiling(double time_ms) const {
        avg_time_ms_ = (avg_time_ms_ * num_runs_ + time_ms) / (num_runs_ + 1);
        num_runs_++;
    }
    
    double avg_time() const { return avg_time_ms_; }
};

// Main computation graph class
class ComputationGraph {
private:
    std::vector<std::shared_ptr<GraphNode>> nodes_;
    std::unordered_map<std::string, GraphNode::NodeId> name_to_id_;
    std::vector<GraphNode::NodeId> input_nodes_;
    std::vector<GraphNode::NodeId> output_nodes_;
    
    // Execution order after topological sort
    std::vector<GraphNode::NodeId> execution_order_;
    
    // Memory management
    std::shared_ptr<MemoryPlanner> memory_planner_;
    std::unordered_map<GraphNode::NodeId, size_t> output_memory_offsets_;
    
    // Optimization flags
    bool is_optimized_;
    bool enable_memory_reuse_;
    bool enable_kernel_fusion_;
    
    // Stream management for parallel execution
    std::vector<cudaStream_t> streams_;
    int num_streams_;
    
    void topological_sort();
    void allocate_memory();
    
public:
    ComputationGraph() 
        : is_optimized_(false), enable_memory_reuse_(true), 
          enable_kernel_fusion_(true), num_streams_(1) {}
    
    ~ComputationGraph();
    
    // Graph construction
    GraphNode::NodeId add_node(const std::string& name, std::shared_ptr<Layer> layer);
    void add_edge(GraphNode::NodeId from, GraphNode::NodeId to);
    void add_edge(const std::string& from, const std::string& to);
    
    void mark_input(GraphNode::NodeId node) { input_nodes_.push_back(node); }
    void mark_output(GraphNode::NodeId node) { output_nodes_.push_back(node); }
    
    // Graph optimization
    void optimize(const GraphOptimizer& optimizer);
    void finalize();  // Prepare for execution
    
    // Execution
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs, ExecutionContext& ctx);
    std::unordered_map<std::string, Tensor> forward_dict(
        const std::unordered_map<std::string, Tensor>& inputs, ExecutionContext& ctx);
    
    // Memory optimization
    size_t get_total_memory_usage() const;
    void enable_memory_reuse(bool enable) { enable_memory_reuse_ = enable; }
    
    // Parallel execution
    void set_num_streams(int num) { num_streams_ = num; }
    
    // Profiling and debugging
    void print_summary() const;
    void export_to_dot(const std::string& filename) const;
    std::unordered_map<std::string, double> get_layer_timings() const;
    
    // Model loading/saving
    void save(const std::string& path) const;
    static std::unique_ptr<ComputationGraph> load(const std::string& path);
    
    // ONNX interop
    static std::unique_ptr<ComputationGraph> from_onnx(const std::string& onnx_path);
    void export_onnx(const std::string& onnx_path) const;
};

// Graph optimization passes
class GraphOptimizer {
public:
    virtual ~GraphOptimizer() = default;
    virtual void optimize(ComputationGraph& graph) const = 0;
};

// Layer fusion optimizer
class LayerFusionOptimizer : public GraphOptimizer {
private:
    // Fusion patterns
    bool try_fuse_conv_bn_relu(ComputationGraph& graph, GraphNode& conv_node) const;
    bool try_fuse_linear_relu(ComputationGraph& graph, GraphNode& linear_node) const;
    
public:
    void optimize(ComputationGraph& graph) const override;
};

// Constant folding optimizer
class ConstantFoldingOptimizer : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) const override;
};

// Memory reuse optimizer
class MemoryPlanner {
private:
    struct MemoryBlock {
        size_t offset;
        size_t size;
        int last_use_time;
    };
    
    std::vector<MemoryBlock> blocks_;
    size_t total_memory_;
    
public:
    MemoryPlanner() : total_memory_(0) {}
    
    void plan(const ComputationGraph& graph);
    size_t allocate(size_t size, int time);
    void free(size_t offset, int time);
    size_t total_memory() const { return total_memory_; }
};

// Quantization optimizer
class QuantizationOptimizer : public GraphOptimizer {
private:
    int bits_;
    bool per_channel_;
    std::vector<std::string> skip_layers_;
    
public:
    QuantizationOptimizer(int bits = 8, bool per_channel = true) 
        : bits_(bits), per_channel_(per_channel) {}
    
    void add_skip_layer(const std::string& name) { skip_layers_.push_back(name); }
    void optimize(ComputationGraph& graph) const override;
};

// Graph executor with different strategies
class GraphExecutor {
public:
    enum class Strategy {
        SEQUENTIAL,      // Simple sequential execution
        PARALLEL,        // Multi-stream parallel execution
        DYNAMIC_BATCH,   // Dynamic batching for inference server
        PIPELINE         // Pipeline parallelism
    };
    
private:
    Strategy strategy_;
    int max_batch_size_;
    std::queue<std::pair<Tensor, std::promise<Tensor>>> pending_requests_;
    
public:
    GraphExecutor(Strategy strategy = Strategy::SEQUENTIAL) 
        : strategy_(strategy), max_batch_size_(32) {}
    
    void set_strategy(Strategy strategy) { strategy_ = strategy; }
    void set_max_batch_size(int size) { max_batch_size_ = size; }
    
    std::future<Tensor> submit(ComputationGraph& graph, const Tensor& input);
    void execute_batch(ComputationGraph& graph);
};

// Subgraph extraction for distributed execution
class SubgraphExtractor {
public:
    struct Subgraph {
        std::vector<GraphNode::NodeId> nodes;
        std::vector<GraphNode::NodeId> inputs;
        std::vector<GraphNode::NodeId> outputs;
        size_t memory_usage;
        double estimated_time;
    };
    
    std::vector<Subgraph> extract(const ComputationGraph& graph, int num_partitions);
    
private:
    double estimate_communication_cost(const Subgraph& sg1, const Subgraph& sg2);
};

} // namespace deep_engine