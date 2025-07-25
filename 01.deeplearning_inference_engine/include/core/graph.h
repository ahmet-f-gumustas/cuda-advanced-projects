#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <queue>
#include "layer.h"
#include "tensor.h"

namespace deep_engine {

class ComputationGraph {
public:
    using NodeId = size_t;
    
    struct Node {
        std::string name;
        std::shared_ptr<Layer> layer;
        std::vector<NodeId> inputs;
        std::vector<NodeId> outputs;
        bool is_input = false;
        bool is_output = false;
    };
    
    ComputationGraph() = default;
    
    // Graph construction
    NodeId add_node(const std::string& name, std::shared_ptr<Layer> layer);
    void add_edge(NodeId from, NodeId to);
    void mark_input(NodeId node);
    void mark_output(NodeId node);
    
    // Graph manipulation
    void remove_node(NodeId node);
    void replace_node(NodeId old_node, NodeId new_node);
    void merge_nodes(NodeId node1, NodeId node2, std::shared_ptr<Layer> merged_layer);
    
    // Graph analysis
    std::vector<NodeId> topological_sort() const;
    bool has_cycle() const;
    std::vector<std::vector<NodeId>> find_subgraphs() const;
    std::unordered_set<NodeId> find_unused_nodes() const;
    
    // Optimization passes
    void optimize(class GraphOptimizer& optimizer);
    void finalize();
    
    // Execution
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs, ExecutionContext& ctx);
    
    // Model I/O
    static std::unique_ptr<ComputationGraph> from_onnx(const std::string& path);
    static std::unique_ptr<ComputationGraph> from_tensorflow(const std::string& path);
    void save(const std::string& path) const;
    static std::unique_ptr<ComputationGraph> load(const std::string& path);
    
    // Accessors
    const Node& get_node(NodeId id) const { return nodes_.at(id); }
    Node& get_node(NodeId id) { return nodes_.at(id); }
    size_t num_nodes() const { return nodes_.size(); }
    
    const std::vector<NodeId>& input_nodes() const { return input_nodes_; }
    const std::vector<NodeId>& output_nodes() const { return output_nodes_; }
    
    // Friend classes for optimization
    friend class QuantizationOptimizer;
    friend class LayerFusionOptimizer;
    friend class ConstantFoldingOptimizer;
    
    // Debug
    void print_graph() const;
    std::string to_dot() const;
    
private:
    std::unordered_map<NodeId, Node> nodes_;
    std::vector<NodeId> input_nodes_;
    std::vector<NodeId> output_nodes_;
    NodeId next_id_ = 0;
    bool finalized_ = false;
    
    // Execution order cache
    std::vector<NodeId> execution_order_;
};

// Base class for graph optimizers
class GraphOptimizer {
public:
    virtual ~GraphOptimizer() = default;
    virtual void optimize(ComputationGraph& graph) = 0;
    virtual std::string name() const = 0;
};

// Common graph optimizations
class ConstantFoldingOptimizer : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "ConstantFolding"; }
};

class DeadNodeEliminationOptimizer : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "DeadNodeElimination"; }
};

class LayerFusionOptimizer : public GraphOptimizer {
public:
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "LayerFusion"; }
};

class QuantizationOptimizer : public GraphOptimizer {
public:
    explicit QuantizationOptimizer(int bits = 8) : bits_(bits) {}
    void optimize(ComputationGraph& graph) override;
    std::string name() const override { return "Quantization"; }
    
private:
    int bits_;
};

// Graph executor strategies
enum class ExecutionStrategy {
    SEQUENTIAL,
    PARALLEL,
    PIPELINE,
    DYNAMIC_BATCHING
};

class GraphExecutor {
public:
    explicit GraphExecutor(ExecutionStrategy strategy = ExecutionStrategy::SEQUENTIAL)
        : strategy_(strategy) {}
    
    std::vector<Tensor> execute(ComputationGraph& graph, 
                                const std::vector<Tensor>& inputs,
                                ExecutionContext& ctx);
    
    void set_strategy(ExecutionStrategy strategy) { strategy_ = strategy; }
    
private:
    ExecutionStrategy strategy_;
    
    std::vector<Tensor> execute_sequential(ComputationGraph& graph,
                                          const std::vector<Tensor>& inputs,
                                          ExecutionContext& ctx);
    
    std::vector<Tensor> execute_parallel(ComputationGraph& graph,
                                        const std::vector<Tensor>& inputs,
                                        ExecutionContext& ctx);
};

// Subgraph extraction for multi-GPU
class SubgraphExtractor {
public:
    std::vector<std::unique_ptr<ComputationGraph>> 
    extract(const ComputationGraph& graph, int num_partitions);
    
private:
    std::vector<std::vector<ComputationGraph::NodeId>> 
    partition_nodes(const ComputationGraph& graph, int num_partitions);
};

} // namespace deep_engine