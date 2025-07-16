#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "core/tensor.h"
#include "core/graph.h"
#include "core/layer.h"
#include "layers/convolution.h"
#include "layers/batchnorm.h"
#include "layers/activation.h"
#include "layers/pooling.h"
#include "layers/dense.h"
#include "utils/model_loader.h"
#include "optimizations/quantization.h"

using namespace deep_engine;
using namespace std::chrono;

// ResNet building blocks
class ResidualBlock : public Layer {
private:
    std::unique_ptr<ConvolutionLayer> conv1_;
    std::unique_ptr<BatchNormLayer> bn1_;
    std::unique_ptr<ActivationLayer> relu1_;
    std::unique_ptr<ConvolutionLayer> conv2_;
    std::unique_ptr<BatchNormLayer> bn2_;
    std::unique_ptr<ConvolutionLayer> downsample_;
    std::unique_ptr<BatchNormLayer> bn_downsample_;
    std::unique_ptr<ActivationLayer> relu2_;
    
    bool has_downsample_;
    
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1, const std::string& name = "")
        : Layer(name), has_downsample_(stride != 1 || in_channels != out_channels) {
        
        // First conv block
        conv1_ = std::make_unique<ConvolutionLayer>(in_channels, out_channels, 3, stride, 1);
        bn1_ = std::make_unique<BatchNormLayer>(out_channels);
        relu1_ = std::make_unique<ActivationLayer>("relu");
        
        // Second conv block
        conv2_ = std::make_unique<ConvolutionLayer>(out_channels, out_channels, 3, 1, 1);
        bn2_ = std::make_unique<BatchNormLayer>(out_channels);
        
        // Downsample if needed
        if (has_downsample_) {
            downsample_ = std::make_unique<ConvolutionLayer>(in_channels, out_channels, 1, stride, 0);
            bn_downsample_ = std::make_unique<BatchNormLayer>(out_channels);
        }
        
        relu2_ = std::make_unique<ActivationLayer>("relu");
    }
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override {
        // Main path
        Tensor out = conv1_->forward(input, ctx);
        out = bn1_->forward(out, ctx);
        out = relu1_->forward(out, ctx);
        
        out = conv2_->forward(out, ctx);
        out = bn2_->forward(out, ctx);
        
        // Residual connection
        Tensor residual = input;
        if (has_downsample_) {
            residual = downsample_->forward(input, ctx);
            residual = bn_downsample_->forward(residual, ctx);
        }
        
        // Element-wise addition
        const float* out_data = static_cast<const float*>(out.data());
        const float* res_data = static_cast<const float*>(residual.data());
        float* result_data = static_cast<float*>(out.data());
        
        size_t num_elements = out.descriptor().total_elements();
        
        // CUDA kernel launch for residual add
        const int threads = 256;
        const int blocks = (num_elements + threads - 1) / threads;
        
        residual_add_kernel<<<blocks, threads, 0, ctx.stream()>>>(
            result_data, res_data, num_elements);
        
        return relu2_->forward(out, ctx);
    }
    
    std::vector<int> infer_output_shape(const std::vector<int>& input_shape) const override {
        return conv2_->infer_output_shape(conv1_->infer_output_shape(input_shape));
    }
    
    std::string type() const override { return "ResidualBlock"; }
    
private:
    __global__ void residual_add_kernel(float* output, const float* residual, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] += residual[idx];
        }
    }
};

// Build ResNet-50 model
std::unique_ptr<ComputationGraph> build_resnet50() {
    auto graph = std::make_unique<ComputationGraph>();
    
    // Initial convolution
    auto input_id = graph->add_node("input", nullptr);  // Placeholder
    
    auto conv1 = std::make_shared<ConvolutionLayer>(3, 64, 7, 2, 3);
    auto conv1_id = graph->add_node("conv1", conv1);
    
    auto bn1 = std::make_shared<BatchNormLayer>(64);
    auto bn1_id = graph->add_node("bn1", bn1);
    
    auto relu1 = std::make_shared<ActivationLayer>("relu");
    auto relu1_id = graph->add_node("relu1", relu1);
    
    auto maxpool = std::make_shared<MaxPoolingLayer>(3, 2, 1);
    auto maxpool_id = graph->add_node("maxpool", maxpool);
    
    // Connect initial layers
    graph->add_edge(input_id, conv1_id);
    graph->add_edge(conv1_id, bn1_id);
    graph->add_edge(bn1_id, relu1_id);
    graph->add_edge(relu1_id, maxpool_id);
    
    // Layer configurations for ResNet-50
    struct StageConfig {
        int num_blocks;
        int in_channels;
        int out_channels;
        int stride;
    };
    
    std::vector<StageConfig> stages = {
        {3, 64, 256, 1},    // conv2_x
        {4, 256, 512, 2},   // conv3_x
        {6, 512, 1024, 2},  // conv4_x
        {3, 1024, 2048, 2}  // conv5_x
    };
    
    GraphNode::NodeId prev_id = maxpool_id;
    int block_idx = 0;
    
    // Build residual stages
    for (int stage = 0; stage < stages.size(); ++stage) {
        auto& config = stages[stage];
        
        for (int block = 0; block < config.num_blocks; ++block) {
            int in_ch = (block == 0) ? config.in_channels : config.out_channels;
            int stride = (block == 0) ? config.stride : 1;
            
            auto res_block = std::make_shared<ResidualBlock>(
                in_ch, config.out_channels, stride,
                "res" + std::to_string(stage+2) + "_" + std::to_string(block)
            );
            
            auto block_id = graph->add_node(res_block->name(), res_block);
            graph->add_edge(prev_id, block_id);
            prev_id = block_id;
            block_idx++;
        }
    }
    
    // Global average pooling
    auto gap = std::make_shared<GlobalAveragePoolingLayer>();
    auto gap_id = graph->add_node("gap", gap);
    graph->add_edge(prev_id, gap_id);
    
    // Final FC layer
    auto fc = std::make_shared<DenseLayer>(2048, 1000);  // ImageNet classes
    auto fc_id = graph->add_node("fc", fc);
    graph->add_edge(gap_id, fc_id);
    
    // Mark input/output
    graph->mark_input(input_id);
    graph->mark_output(fc_id);
    
    return graph;
}

// Image preprocessing
Tensor preprocess_image(const cv::Mat& img) {
    // Resize to 224x224
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    
    // Convert to float and normalize
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    // ImageNet normalization
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    
    cv::Mat normalized;
    cv::subtract(float_img, mean, normalized);
    cv::divide(normalized, std, normalized);
    
    // Convert to CHW format
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    
    // Create tensor
    Tensor input_tensor({1, 3, 224, 224});
    
    float* tensor_data = new float[3 * 224 * 224];
    for (int c = 0; c < 3; ++c) {
        memcpy(tensor_data + c * 224 * 224, 
               channels[c].data, 
               224 * 224 * sizeof(float));
    }
    
    input_tensor.copy_from_host(tensor_data);
    delete[] tensor_data;
    
    return input_tensor;
}

// Load ImageNet class names
std::vector<std::string> load_class_names(const std::string& filename) {
    std::vector<std::string> classes;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        classes.push_back(line);
    }
    
    return classes;
}

// Get top-k predictions
std::vector<std::pair<int, float>> get_topk(const Tensor& output, int k = 5) {
    std::vector<float> scores(1000);
    output.copy_to_host(scores.data());
    
    // Softmax
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum = 0;
    for (auto& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
    }
    for (auto& s : scores) {
        s /= sum;
    }
    
    // Get top-k
    std::vector<std::pair<int, float>> indexed_scores;
    for (int i = 0; i < 1000; ++i) {
        indexed_scores.push_back({i, scores[i]});
    }
    
    std::partial_sort(indexed_scores.begin(), 
                      indexed_scores.begin() + k,
                      indexed_scores.end(),
                      [](const auto& a, const auto& b) {
                          return a.second > b.second;
                      });
    
    indexed_scores.resize(k);
    return indexed_scores;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [model_path] [--quantize]" << std::endl;
        return 1;
    }
    
    std::string image_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "resnet50.bin";
    bool use_quantization = (argc > 3 && std::string(argv[3]) == "--quantize");
    
    try {
        // Load image
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }
        
        std::cout << "Building ResNet-50 model..." << std::endl;
        auto graph = build_resnet50();
        
        // Load pretrained weights if available
        if (std::filesystem::exists(model_path)) {
            std::cout << "Loading weights from " << model_path << std::endl;
            graph = ComputationGraph::load(model_path);
        } else {
            std::cout << "Warning: Model file not found, using random weights" << std::endl;
        }
        
        // Apply optimizations
        LayerFusionOptimizer fusion_opt;
        fusion_opt.optimize(*graph);
        
        if (use_quantization) {
            std::cout << "Applying INT8 quantization..." << std::endl;
            QuantizationOptimizer quant_opt(8, true);
            quant_opt.add_skip_layer("fc");  // Don't quantize final layer
            quant_opt.optimize(*graph);
        }
        
        graph->finalize();
        
        // Print model summary
        graph->print_summary();
        std::cout << "Total memory usage: " << graph->get_total_memory_usage() / (1024*1024) << " MB\n";
        
        // Preprocess image
        std::cout << "\nPreprocessing image..." << std::endl;
        Tensor input = preprocess_image(img);
        
        // Create execution context
        ExecutionContext ctx;
        ctx.enable_profiling(true);
        
        // Warmup runs
        std::cout << "Running warmup..." << std::endl;
        for (int i = 0; i < 10; ++i) {
            graph->forward({input}, ctx);
        }
        ctx.synchronize();
        
        // Benchmark
        std::cout << "\nRunning inference benchmark..." << std::endl;
        const int num_runs = 100;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            graph->forward({input}, ctx);
        }
        ctx.synchronize();
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        
        std::cout << "Average inference time: " << duration / num_runs / 1000.0 << " ms" << std::endl;
        std::cout << "Throughput: " << num_runs * 1000000.0 / duration << " images/sec" << std::endl;
        
        // Get predictions
        auto outputs = graph->forward({input}, ctx);
        auto predictions = get_topk(outputs[0], 5);
        
        // Load class names
        std::vector<std::string> classes;
        if (std::filesystem::exists("imagenet_classes.txt")) {
            classes = load_class_names("imagenet_classes.txt");
        }
        
        // Print results
        std::cout << "\nTop 5 predictions:" << std::endl;
        for (const auto& [idx, score] : predictions) {
            if (!classes.empty() && idx < classes.size()) {
                std::cout << "  " << classes[idx] << ": " << score * 100 << "%" << std::endl;
            } else {
                std::cout << "  Class " << idx << ": " << score * 100 << "%" << std::endl;
            }
        }
        
        // Print layer timings
        if (ctx.enable_profiling) {
            std::cout << "\nLayer timings:" << std::endl;
            auto timings = graph->get_layer_timings();
            
            std::vector<std::pair<std::string, double>> sorted_timings(
                timings.begin(), timings.end());
            
            std::sort(sorted_timings.begin(), sorted_timings.end(),
                     [](const auto& a, const auto& b) {
                         return a.second > b.second;
                     });
            
            for (const auto& [name, time] : sorted_timings) {
                if (time > 0.01) {  // Only show layers taking > 0.01ms
                    std::cout << "  " << name << ": " << time << " ms" << std::endl;
                }
            }
        }
        
        // Export graph visualization
        graph->export_to_dot("resnet50_graph.dot");
        std::cout << "\nGraph exported to resnet50_graph.dot" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}