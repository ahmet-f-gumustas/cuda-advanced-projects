#include <gtest/gtest.h>
#include "core/graph.h"
#include "layers/convolution.h"
#include "layers/activation.h"
#include "layers/pooling.h"
#include "layers/dense.h"
#include "test_layers_helpers.h"
#include <memory>

using namespace deep_engine;

class GraphTest : public ::testing::Test {
protected:
    std::unique_ptr<ComputationGraph> graph_;
    ExecutionContext ctx_;
    
    void SetUp() override {
        graph_ = std::make_unique<ComputationGraph>();
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

TEST_F(GraphTest, AddNodes) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    
    auto conv_id = graph_->add_node("conv1", conv);
    auto relu_id = graph_->add_node("relu1", relu);
    
    EXPECT_EQ(graph_->num_nodes(), 2);
    EXPECT_EQ(graph_->get_node(conv_id).name, "conv1");
    EXPECT_EQ(graph_->get_node(relu_id).name, "relu1");
}

TEST_F(GraphTest, AddEdges) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    
    auto conv_id = graph_->add_node("conv", conv);
    auto relu_id = graph_->add_node("relu", relu);
    
    graph_->add_edge(conv_id, relu_id);
    
    EXPECT_EQ(graph_->get_node(conv_id).outputs.size(), 1);
    EXPECT_EQ(graph_->get_node(conv_id).outputs[0], relu_id);
    EXPECT_EQ(graph_->get_node(relu_id).inputs.size(), 1);
    EXPECT_EQ(graph_->get_node(relu_id).inputs[0], conv_id);
}

TEST_F(GraphTest, MarkInputOutput) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto pool = std::make_shared<MaxPool2d>(2);
    
    auto conv_id = graph_->add_node("conv", conv);
    auto pool_id = graph_->add_node("pool", pool);
    
    graph_->add_edge(conv_id, pool_id);
    graph_->mark_input(conv_id);
    graph_->mark_output(pool_id);
    
    EXPECT_TRUE(graph_->get_node(conv_id).is_input);
    EXPECT_TRUE(graph_->get_node(pool_id).is_output);
    EXPECT_EQ(graph_->input_nodes().size(), 1);
    EXPECT_EQ(graph_->output_nodes().size(), 1);
}

TEST_F(GraphTest, TopologicalSort) {
    // Create a simple linear graph: conv -> relu -> pool
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    auto pool = std::make_shared<MaxPool2d>(2);
    
    auto conv_id = graph_->add_node("conv", conv);
    auto relu_id = graph_->add_node("relu", relu);
    auto pool_id = graph_->add_node("pool", pool);
    
    graph_->add_edge(conv_id, relu_id);
    graph_->add_edge(relu_id, pool_id);
    
    auto sorted = graph_->topological_sort();
    
    EXPECT_EQ(sorted.size(), 3);
    // Conv should come before relu, relu before pool
    auto conv_pos = std::find(sorted.begin(), sorted.end(), conv_id);
    auto relu_pos = std::find(sorted.begin(), sorted.end(), relu_id);
    auto pool_pos = std::find(sorted.begin(), sorted.end(), pool_id);
    
    EXPECT_LT(conv_pos, relu_pos);
    EXPECT_LT(relu_pos, pool_pos);
}

TEST_F(GraphTest, CycleDetection) {
    auto layer1 = std::make_shared<ReLU>();
    auto layer2 = std::make_shared<ReLU>();
    
    auto id1 = graph_->add_node("layer1", layer1);
    auto id2 = graph_->add_node("layer2", layer2);
    
    // Create cycle
    graph_->add_edge(id1, id2);
    graph_->add_edge(id2, id1);
    
    EXPECT_TRUE(graph_->has_cycle());
}

TEST_F(GraphTest, NoCycle) {
    auto layer1 = std::make_shared<ReLU>();
    auto layer2 = std::make_shared<ReLU>();
    auto layer3 = std::make_shared<ReLU>();
    
    auto id1 = graph_->add_node("layer1", layer1);
    auto id2 = graph_->add_node("layer2", layer2);
    auto id3 = graph_->add_node("layer3", layer3);
    
    // Create DAG
    graph_->add_edge(id1, id2);
    graph_->add_edge(id1, id3);
    graph_->add_edge(id2, id3);
    
    EXPECT_FALSE(graph_->has_cycle());
}

TEST_F(GraphTest, RemoveNode) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    auto pool = std::make_shared<MaxPool2d>(2);
    
    auto conv_id = graph_->add_node("conv", conv);
    auto relu_id = graph_->add_node("relu", relu);
    auto pool_id = graph_->add_node("pool", pool);
    
    graph_->add_edge(conv_id, relu_id);
    graph_->add_edge(relu_id, pool_id);
    
    // Remove middle node
    graph_->remove_node(relu_id);
    
    EXPECT_EQ(graph_->num_nodes(), 2);
    // Conv should no longer have outputs
    EXPECT_EQ(graph_->get_node(conv_id).outputs.size(), 0);
    // Pool should no longer have inputs
    EXPECT_EQ(graph_->get_node(pool_id).inputs.size(), 0);
}

TEST_F(GraphTest, ReplaceNode) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    auto pool = std::make_shared<MaxPool2d>(2);
    auto gelu = std::make_shared<GELU>();
    
    auto conv_id = graph_->add_node("conv", conv);
    auto relu_id = graph_->add_node("relu", relu);
    auto pool_id = graph_->add_node("pool", pool);
    auto gelu_id = graph_->add_node("gelu", gelu);
    
    graph_->add_edge(conv_id, relu_id);
    graph_->add_edge(relu_id, pool_id);
    
    // Replace ReLU with GELU
    graph_->replace_node(relu_id, gelu_id);
    
    EXPECT_EQ(graph_->num_nodes(), 3);
    // Check connections
    EXPECT_EQ(graph_->get_node(conv_id).outputs[0], gelu_id);
    EXPECT_EQ(graph_->get_node(gelu_id).inputs[0], conv_id);
    EXPECT_EQ(graph_->get_node(gelu_id).outputs[0], pool_id);
    EXPECT_EQ(graph_->get_node(pool_id).inputs[0], gelu_id);
}

TEST_F(GraphTest, FindUnusedNodes) {
    auto conv1 = std::make_shared<Conv2d>(3, 64, 3);
    auto conv2 = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    auto pool = std::make_shared<MaxPool2d>(2);
    
    auto conv1_id = graph_->add_node("conv1", conv1);
    auto conv2_id = graph_->add_node("conv2", conv2);  // Unused
    auto relu_id = graph_->add_node("relu", relu);
    auto pool_id = graph_->add_node("pool", pool);
    
    graph_->add_edge(conv1_id, relu_id);
    graph_->add_edge(relu_id, pool_id);
    
    graph_->mark_input(conv1_id);
    graph_->mark_output(pool_id);
    
    auto unused = graph_->find_unused_nodes();
    
    EXPECT_EQ(unused.size(), 1);
    EXPECT_TRUE(unused.find(conv2_id) != unused.end());
}

TEST_F(GraphTest, SimpleForward) {
    // Build simple graph: input -> conv -> relu -> output
    auto conv = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
    auto relu = std::make_shared<ReLU>();
    
    auto conv_id = graph_->add_node("conv", conv);
    auto relu_id = graph_->add_node("relu", relu);
    
    graph_->add_edge(conv_id, relu_id);
    graph_->mark_input(conv_id);
    graph_->mark_output(relu_id);
    
    graph_->finalize();
    
    // Run forward pass
    Tensor input({1, 3, 32, 32});
    auto outputs = graph_->forward({input}, ctx_);
    
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].shape(), std::vector<int>({1, 16, 32, 32}));
}

TEST_F(GraphTest, MultiInputOutput) {
    // Graph with multiple inputs and outputs
    auto conv1 = std::make_shared<Conv2d>(3, 32, 3);
    auto conv2 = std::make_shared<Conv2d>(3, 32, 3);
    auto concat = std::make_shared<ConcatLayer>(1);  // Concat along channel
    auto split = std::make_shared<SplitLayer>(2, 1); // Split into 2 along channel
    
    auto conv1_id = graph_->add_node("conv1", conv1);
    auto conv2_id = graph_->add_node("conv2", conv2);
    auto concat_id = graph_->add_node("concat", concat);
    auto split_id = graph_->add_node("split", split);
    
    graph_->add_edge(conv1_id, concat_id);
    graph_->add_edge(conv2_id, concat_id);
    graph_->add_edge(concat_id, split_id);
    
    graph_->mark_input(conv1_id);
    graph_->mark_input(conv2_id);
    graph_->mark_output(split_id);
    
    graph_->finalize();
    
    // Run with two inputs
    Tensor input1({1, 3, 32, 32});
    Tensor input2({1, 3, 32, 32});
    
    auto outputs = graph_->forward({input1, input2}, ctx_);
    
    // Split should produce 2 outputs
    EXPECT_EQ(outputs.size(), 2);
}

TEST_F(GraphTest, ResidualConnection) {
    // Graph with residual connection
    auto conv1 = std::make_shared<Conv2d>(64, 64, 3, 1, 1);
    auto relu = std::make_shared<ReLU>();
    auto conv2 = std::make_shared<Conv2d>(64, 64, 3, 1, 1);
    auto add = std::make_shared<AddLayer>();
    
    auto input_id = graph_->add_node("input", std::make_shared<IdentityLayer>());
    auto conv1_id = graph_->add_node("conv1", conv1);
    auto relu_id = graph_->add_node("relu", relu);
    auto conv2_id = graph_->add_node("conv2", conv2);
    auto add_id = graph_->add_node("add", add);
    
    // Main path
    graph_->add_edge(input_id, conv1_id);
    graph_->add_edge(conv1_id, relu_id);
    graph_->add_edge(relu_id, conv2_id);
    graph_->add_edge(conv2_id, add_id);
    
    // Residual connection
    graph_->add_edge(input_id, add_id);
    
    graph_->mark_input(input_id);
    graph_->mark_output(add_id);
    
    auto sorted = graph_->topological_sort();
    
    // Verify that add comes after both conv2 and input
    auto add_pos = std::find(sorted.begin(), sorted.end(), add_id);
    auto conv2_pos = std::find(sorted.begin(), sorted.end(), conv2_id);
    auto input_pos = std::find(sorted.begin(), sorted.end(), input_id);
    
    EXPECT_LT(conv2_pos, add_pos);
    EXPECT_LT(input_pos, add_pos);
}

TEST_F(GraphTest, DeadNodeElimination) {
    // Create graph with unused nodes
    auto conv1 = std::make_shared<Conv2d>(3, 64, 3);
    auto conv2 = std::make_shared<Conv2d>(3, 64, 3);  // Will be unused
    auto conv3 = std::make_shared<Conv2d>(3, 64, 3);  // Will be unused
    auto relu = std::make_shared<ReLU>();
    
    auto conv1_id = graph_->add_node("conv1", conv1);
    auto conv2_id = graph_->add_node("conv2", conv2);
    auto conv3_id = graph_->add_node("conv3", conv3);
    auto relu_id = graph_->add_node("relu", relu);
    
    graph_->add_edge(conv1_id, relu_id);
    graph_->add_edge(conv2_id, conv3_id);  // Disconnected subgraph
    
    graph_->mark_input(conv1_id);
    graph_->mark_output(relu_id);
    
    // Apply dead node elimination
    DeadNodeEliminationOptimizer optimizer;
    graph_->optimize(optimizer);
    
    // Only conv1 and relu should remain
    EXPECT_EQ(graph_->num_nodes(), 2);
}

TEST_F(GraphTest, GraphToDot) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3);
    auto relu = std::make_shared<ReLU>();
    auto pool = std::make_shared<MaxPool2d>(2);
    
    auto conv_id = graph_->add_node("conv", conv);
    auto relu_id = graph_->add_node("relu", relu);
    auto pool_id = graph_->add_node("pool", pool);
    
    graph_->add_edge(conv_id, relu_id);
    graph_->add_edge(relu_id, pool_id);
    
    graph_->mark_input(conv_id);
    graph_->mark_output(pool_id);
    
    std::string dot = graph_->to_dot();
    
    // Check that DOT format contains expected elements
    EXPECT_TRUE(dot.find("digraph ComputationGraph") != std::string::npos);
    EXPECT_TRUE(dot.find("conv") != std::string::npos);
    EXPECT_TRUE(dot.find("relu") != std::string::npos);
    EXPECT_TRUE(dot.find("pool") != std::string::npos);
    EXPECT_TRUE(dot.find("->") != std::string::npos);
}

// Test error cases
TEST_F(GraphTest, InvalidNodeId) {
    EXPECT_THROW(graph_->get_node(999), std::exception);
}

TEST_F(GraphTest, FinalizeWithoutInputs) {
    auto relu = std::make_shared<ReLU>();
    auto relu_id = graph_->add_node("relu", relu);
    graph_->mark_output(relu_id);
    
    EXPECT_THROW(graph_->finalize(), std::runtime_error);
}

TEST_F(GraphTest, FinalizeWithoutOutputs) {
    auto relu = std::make_shared<ReLU>();
    auto relu_id = graph_->add_node("relu", relu);
    graph_->mark_input(relu_id);
    
    EXPECT_THROW(graph_->finalize(), std::runtime_error);
}

TEST_F(GraphTest, ForwardBeforeFinalize) {
    auto relu = std::make_shared<ReLU>();
    auto relu_id = graph_->add_node("relu", relu);
    graph_->mark_input(relu_id);
    graph_->mark_output(relu_id);
    
    Tensor input({1, 3, 32, 32});
    EXPECT_THROW(graph_->forward({input}, ctx_), std::runtime_error);
}

// Performance test
TEST_F(GraphTest, DISABLED_LargeGraphConstruction) {
    // Build a large sequential graph
    const int num_layers = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ComputationGraph::NodeId prev_id;
    for (int i = 0; i < num_layers; ++i) {
        auto layer = std::make_shared<ReLU>();
        auto id = graph_->add_node("layer_" + std::to_string(i), layer);
        
        if (i == 0) {
            graph_->mark_input(id);
        } else {
            graph_->add_edge(prev_id, id);
        }
        
        if (i == num_layers - 1) {
            graph_->mark_output(id);
        }
        
        prev_id = id;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Large graph construction took: " << duration.count() << " us" << std::endl;
    
    // Test topological sort performance
    start = std::chrono::high_resolution_clock::now();
    auto sorted = graph_->topological_sort();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Topological sort took: " << duration.count() << " us" << std::endl;
    
    EXPECT_EQ(sorted.size(), num_layers);
}