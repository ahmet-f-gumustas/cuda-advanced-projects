#include "../../include/core/graph.h"
#include "../../include/utils/logger.h"
#include "../../include/utils/profiler.h"
#include <queue>
#include <stack>
#include <algorithm>

namespace deep_engine {

ComputationGraph::NodeId ComputationGraph::add_node(const std::string& name, 
                                                   std::shared_ptr<Layer> layer) {
    NodeId id = next_id_++;
    Node node;
    node.name = name;
    node.layer = layer;
    nodes_[id] = node;
    return id;
}

void ComputationGraph::add_edge(NodeId from, NodeId to) {
    if (nodes_.find(from) == nodes_.end() || nodes_.find(to) == nodes_.end()) {
        throw std::runtime_error("Invalid node ID in add_edge");
    }
    
    nodes_[from].outputs.push_back(to);
    nodes_[to].inputs.push_back(from);
}

void ComputationGraph::mark_input(NodeId node) {
    if (nodes_.find(node) == nodes_.end()) {
        throw std::runtime_error("Invalid node ID in mark_input");
    }
    
    nodes_[node].is_input = true;
    input_nodes_.push_back(node);
}

void ComputationGraph::mark_output(NodeId node) {
    if (nodes_.find(node) == nodes_.end()) {
        throw std::runtime_error("Invalid node ID in mark_output");
    }
    
    nodes_[node].is_output = true;
    output_nodes_.push_back(node);
}

void ComputationGraph::remove_node(NodeId node) {
    if (nodes_.find(node) == nodes_.end()) {
        return;
    }
    
    // Remove edges
    for (NodeId input : nodes_[node].inputs) {
        auto& outputs = nodes_[input].outputs;
        outputs.erase(std::remove(outputs.begin(), outputs.end(), node), outputs.end());
    }
    
    for (NodeId output : nodes_[node].outputs) {
        auto& inputs = nodes_[output].inputs;
        inputs.erase(std::remove(inputs.begin(), inputs.end(), node), inputs.end());
    }
    
    // Remove from input/output lists
    input_nodes_.erase(std::remove(input_nodes_.begin(), input_nodes_.end(), node), 
                      input_nodes_.end());
    output_nodes_.erase(std::remove(output_nodes_.begin(), output_nodes_.end(), node), 
                       output_nodes_.end());
    
    // Remove node
    nodes_.erase(node);
}

void ComputationGraph::replace_node(NodeId old_node, NodeId new_node) {
    if (nodes_.find(old_node) == nodes_.end() || nodes_.find(new_node) == nodes_.end()) {
        throw std::runtime_error("Invalid node ID in replace_node");
    }
    
    // Copy connections
    nodes_[new_node].inputs = nodes_[old_node].inputs;
    nodes_[new_node].outputs = nodes_[old_node].outputs;
    
    // Update connections in other nodes
    for (NodeId input : nodes_[old_node].inputs) {
        auto& outputs = nodes_[input].outputs;
        std::replace(outputs.begin(), outputs.end(), old_node, new_node);
    }
    
    for (NodeId output : nodes_[old_node].outputs) {
        auto& inputs = nodes_[output].inputs;
        std::replace(inputs.begin(), inputs.end(), old_node, new_node);
    }
    
    // Update input/output lists
    std::replace(input_nodes_.begin(), input_nodes_.end(), old_node, new_node);
    std::replace(output_nodes_.begin(), output_nodes_.end(), old_node, new_node);
    
    // Remove old node
    nodes_.erase(old_node);
}

std::vector<ComputationGraph::NodeId> ComputationGraph::topological_sort() const {
    std::vector<NodeId> result;
    std::unordered_map<NodeId, int> in_degree;
    
    // Calculate in-degrees
    for (const auto& [id, node] : nodes_) {
        in_degree[id] = node.inputs.size();
    }
    
    // Find nodes with no inputs
    std::queue<NodeId> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(id);
        }
    }
    
    // Process nodes
    while (!queue.empty()) {
        NodeId current = queue.front();
        queue.pop();
        result.push_back(current);
        
        // Reduce in-degree of output nodes
        for (NodeId output : nodes_.at(current).outputs) {
            in_degree[output]--;
            if (in_degree[output] == 0) {
                queue.push(output);
            }
        }
    }
    
    if (result.size() != nodes_.size()) {
        throw std::runtime_error("Graph contains cycles");
    }
    
    return result;
}

bool ComputationGraph::has_cycle() const {
    try {
        topological_sort();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

std::unordered_set<ComputationGraph::NodeId> ComputationGraph::find_unused_nodes() const {
    std::unordered_set<NodeId> reachable;
    std::queue<NodeId> queue;
    
    // Start from output nodes
    for (NodeId output : output_nodes_) {
        queue.push(output);
        reachable.insert(output);
    }
    
    // Traverse backwards
    while (!queue.empty()) {
        NodeId current = queue.front();
        queue.pop();
        
        for (NodeId input : nodes_.at(current).inputs) {
            if (reachable.find(input) == reachable.end()) {
                reachable.insert(input);
                queue.push(input);
            }
        }
    }
    
    // Find unreachable nodes
    std::unordered_set<NodeId> unused;
    for (const auto& [id, node] : nodes_) {
        if (reachable.find(id) == reachable.end()) {
            unused.insert(id);
        }
    }
    
    return unused;
}

void ComputationGraph::optimize(GraphOptimizer& optimizer) {
    LOG_INFO("Applying optimization: %s", optimizer.name().c_str());
    optimizer.optimize(*this);
}

void ComputationGraph::finalize() {
    if (finalized_) {
        return;
    }
    
    // Validate graph
    if (input_nodes_.empty()) {
        throw std::runtime_error("No input nodes marked");
    }
    if (output_nodes_.empty()) {
        throw std::runtime_error("No output nodes marked");
    }
    
    // Compute execution order
    execution_order_ = topological_sort();
    
    finalized_ = true;
    LOG_INFO("Graph finalized with %zu nodes", nodes_.size());
}

std::vector<Tensor> ComputationGraph::forward(const std::vector<Tensor>& inputs, 
                                             ExecutionContext& ctx) {
    if (!finalized_) {
        throw std::runtime_error("Graph must be finalized before execution");
    }
    
    if (inputs.size() != input_nodes_.size()) {
        throw std::runtime_error("Number of inputs doesn't match graph input nodes");
    }
    
    // Store intermediate results
    std::unordered_map<NodeId, Tensor> activations;
    
    // Set input activations
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
        activations[input_nodes_[i]] = inputs[i];
    }
    
    // Execute nodes in topological order
    for (NodeId node_id : execution_order_) {
        const Node& node = nodes_[node_id];
        
        // Skip if already computed (input node)
        if (activations.find(node_id) != activations.end()) {
            continue;
        }
        
        // Gather inputs
        std::vector<Tensor> node_inputs;
        for (NodeId input_id : node.inputs) {
            node_inputs.push_back(activations.at(input_id));
        }
        
        // Execute layer
        {
            ProfileScope scope(node.name, node.layer->type());
            
            std::vector<Tensor> node_outputs;
            if (node_inputs.size() == 1) {
                node_outputs = {node.layer->forward(node_inputs[0], ctx)};
            } else {
                node_outputs = node.layer->forward(node_inputs, ctx);
            }
            
            // Store output (assuming single output per node for now)
            activations[node_id] = node_outputs[0];
        }
        
        // Free memory of intermediate activations that are no longer needed
        for (NodeId input_id : node.inputs) {
            bool still_needed = false;
            for (NodeId other_id : nodes_.at(input_id).outputs) {
                if (activations.find(other_id) == activations.end()) {
                    still_needed = true;
                    break;
                }
            }
            if (!still_needed && !nodes_.at(input_id).is_output) {
                activations.erase(input_id);
            }
        }
    }
    
    // Gather outputs
    std::vector<Tensor> outputs;
    for (NodeId output_id : output_nodes_) {
        outputs.push_back(activations.at(output_id));
    }
    
    return outputs;
}

void ComputationGraph::print_graph() const {
    std::cout << "Computation Graph:" << std::endl;
    std::cout << "  Nodes: " << nodes_.size() << std::endl;
    std::cout << "  Inputs: ";
    for (NodeId id : input_nodes_) {
        std::cout << nodes_.at(id).name << " ";
    }
    std::cout << std::endl;
    std::cout << "  Outputs: ";
    for (NodeId id : output_nodes_) {
        std::cout << nodes_.at(id).name << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n  Nodes:" << std::endl;
    for (const auto& [id, node] : nodes_) {
        std::cout << "    " << node.name << " (" << node.layer->type() << ")";
        if (!node.inputs.empty()) {
            std::cout << " <- ";
            for (NodeId input : node.inputs) {
                std::cout << nodes_.at(input).name << " ";
            }
        }
        std::cout << std::endl;
    }
}

std::string ComputationGraph::to_dot() const {
    std::stringstream ss;
    ss << "digraph ComputationGraph {" << std::endl;
    ss << "  rankdir=TB;" << std::endl;
    
    // Node definitions
    for (const auto& [id, node] : nodes_) {
        ss << "  " << id << " [label=\"" << node.name << "\\n" 
           << node.layer->type() << "\"";
        
        if (node.is_input) {
            ss << ", style=filled, fillcolor=lightblue";
        } else if (node.is_output) {
            ss << ", style=filled, fillcolor=lightgreen";
        }
        
        ss << "];" << std::endl;
    }
    
    // Edges
    for (const auto& [id, node] : nodes_) {
        for (NodeId output : node.outputs) {
            ss << "  " << id << " -> " << output << ";" << std::endl;
        }
    }
    
    ss << "}" << std::endl;
    return ss.str();
}

// Graph optimizers implementation

void ConstantFoldingOptimizer::optimize(ComputationGraph& graph) {
    // TODO: Implement constant folding
    // Identify nodes with constant inputs and precompute their outputs
}

void DeadNodeEliminationOptimizer::optimize(ComputationGraph& graph) {
    auto unused_nodes = graph.find_unused_nodes();
    
    for (auto node_id : unused_nodes) {
        LOG_DEBUG("Removing unused node: %s", graph.get_node(node_id).name.c_str());
        graph.remove_node(node_id);
    }
    
    if (!unused_nodes.empty()) {
        LOG_INFO("Removed %zu unused nodes", unused_nodes.size());
    }
}

void LayerFusionOptimizer::optimize(ComputationGraph& graph) {
    // Common fusion patterns
    std::vector<std::tuple<std::string, std::string, std::string>> patterns = {
        {"Conv2d", "BatchNorm2d", "ConvBN"},
        {"Conv2d", "ReLU", "ConvReLU"},
        {"ConvBN", "ReLU", "ConvBNReLU"},
        {"Linear", "ReLU", "LinearReLU"},
    };
    
    // TODO: Implement pattern matching and fusion
    // For each pattern, find matching sequences in the graph and fuse them
}

void QuantizationOptimizer::optimize(ComputationGraph& graph) {
    // Quantize eligible layers
    // Get all node IDs first
    std::vector<ComputationGraph::NodeId> node_ids;
    for (const auto& input_id : graph.input_nodes()) {
        node_ids.push_back(input_id);
    }
    for (const auto& output_id : graph.output_nodes()) {
        if (std::find(node_ids.begin(), node_ids.end(), output_id) == node_ids.end()) {
            node_ids.push_back(output_id);
        }
    }
    
    // Use topological sort to get all nodes
    auto all_nodes = graph.topological_sort();
    
    for (auto node_id : all_nodes) {
        auto& node = graph.get_node(node_id);
        if (node.layer->supports_quantization()) {
            LOG_DEBUG("Quantizing layer: %s", node.name.c_str());
            node.layer->quantize(bits_);
        }
    }
}

} // namespace deep_engine