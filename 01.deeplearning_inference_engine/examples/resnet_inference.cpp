#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../include/core/graph.h"
#include "../include/core/tensor.h"
#include "../include/utils/model_loader.h"
#include "../include/utils/profiler.h"
#include "../include/utils/logger.h"
#include "../include/optimizations/quantization.h"
#include "../include/optimizations/fusion.h"

using namespace deep_engine;

// ImageNet preprocessing
Tensor preprocess_image(const std::string& image_path) {
    // Read image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    // Resize to 224x224
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    
    // Convert BGR to RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    
    // Create tensor
    Tensor input({1, 3, 224, 224}, DataType::FP32);
    
    // Copy data with channel-first format
    float* data = input.data<float>();
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                data[c * 224 * 224 + h * 224 + w] = 
                    resized.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    // Apply ImageNet normalization
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[] = {0.229f, 0.224f, 0.225f};
    
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 224 * 224; ++i) {
            data[c * 224 * 224 + i] = 
                (data[c * 224 * 224 + i] - mean[c]) / std[c];
        }
    }
    
    return input;
}

// Load class names
std::vector<std::string> load_class_names(const std::string& path) {
    std::vector<std::string> classes;
    std::ifstream file(path);
    std::string line;
    
    while (std::getline(file, line)) {
        classes.push_back(line);
    }
    
    return classes;
}

// Get top-k predictions
std::vector<std::pair<int, float>> get_topk_predictions(const Tensor& output, int k = 5) {
    std::vector<std::pair<int, float>> predictions;
    const float* data = output.data<float>();
    int num_classes = output.shape()[1];
    
    // Get all scores
    for (int i = 0; i < num_classes; ++i) {
        predictions.push_back({i, data[i]});
    }
    
    // Sort by score
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top-k
    predictions.resize(k);
    return predictions;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --quantize    Use INT8 quantization" << std::endl;
        std::cerr << "  --profile     Enable profiling" << std::endl;
        std::cerr << "  --batch <n>   Batch size (default: 1)" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    
    // Parse options
    bool use_quantization = false;
    bool enable_profiling = false;
    int batch_size = 1;
    
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quantize") {
            use_quantization = true;
        } else if (arg == "--profile") {
            enable_profiling = true;
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        }
    }
    
    try {
        // Initialize logger
        Logger::instance().set_level(LogLevel::INFO);
        LOG_INFO("Deep Learning Inference Engine - ResNet Demo");
        
        // Enable profiling if requested
        if (enable_profiling) {
            Profiler::instance().enable(true);
        }
        
        // Load model
        LOG_INFO("Loading model from: %s", model_path.c_str());
        auto loader = ModelLoaderFactory::create_from_file(model_path);
        auto graph = loader->load(model_path);
        
        // Apply optimizations
        LOG_INFO("Applying optimizations...");
        
        // Layer fusion
        LayerFusionOptimizer fusion_opt;
        fusion_opt.optimize(*graph);
        
        // Quantization
        if (use_quantization) {
            LOG_INFO("Applying INT8 quantization...");
            QuantizationOptimizer quant_opt(8);
            quant_opt.optimize(*graph);
        }
        
        // Dead node elimination
        DeadNodeEliminationOptimizer dead_opt;
        dead_opt.optimize(*graph);
        
        // Finalize graph
        graph->finalize();
        
        // Preprocess image
        LOG_INFO("Preprocessing image: %s", image_path.c_str());
        Tensor input = preprocess_image(image_path);
        
        // Handle batching
        if (batch_size > 1) {
            std::vector<Tensor> batch_inputs;
            for (int i = 0; i < batch_size; ++i) {
                batch_inputs.push_back(input.clone());
            }
            input = cat(batch_inputs, 0);
        }
        
        // Create execution context
        ExecutionContext ctx;
        
        // Warm up
        LOG_INFO("Warming up...");
        for (int i = 0; i < 3; ++i) {
            graph->forward({input}, ctx);
        }
        ctx.synchronize();
        
        // Benchmark
        LOG_INFO("Running inference...");
        const int num_runs = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            graph->forward({input}, ctx);
        }
        ctx.synchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        float avg_latency = duration.count() / 1000.0f / num_runs;
        float throughput = batch_size * 1000.0f / avg_latency;
        
        LOG_INFO("Performance Results:");
        LOG_INFO("  Batch size: %d", batch_size);
        LOG_INFO("  Average latency: %.2f ms", avg_latency);
        LOG_INFO("  Throughput: %.2f images/sec", throughput);
        
        // Get final predictions
        auto outputs = graph->forward({input}, ctx);
        auto& output = outputs[0];
        
        // Load class names
        std::vector<std::string> class_names;
        try {
            class_names = load_class_names("imagenet_classes.txt");
        } catch (...) {
            LOG_WARNING("Could not load class names");
        }
        
        // Show top-5 predictions for first image in batch
        if (batch_size > 1) {
            output = output.slice(0, 0, 1);
        }
        
        auto predictions = get_topk_predictions(output, 5);
        
        LOG_INFO("\nTop-5 Predictions:");
        for (const auto& pred : predictions) {
            std::string class_name = pred.first < class_names.size() ? 
                                    class_names[pred.first] : 
                                    "Class " + std::to_string(pred.first);
            LOG_INFO("  %s: %.2f%%", class_name.c_str(), pred.second * 100);
        }
        
        // Print profiling results
        if (enable_profiling) {
            LOG_INFO("\nProfiling Results:");
            Profiler::instance().print_summary();
            Profiler::instance().export_chrome_trace("resnet_trace.json");
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error: %s", e.what());
        return 1;
    }
    
    return 0;
}