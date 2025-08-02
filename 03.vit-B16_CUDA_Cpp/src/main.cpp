#include "vit_model.hpp"
#include "weights.hpp"
#include "timers.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <getopt.h>

namespace fs = std::filesystem;

struct Args {
    std::string image_path;
    std::string weights_path;
    int batch_size = 1;
    std::string precision = "fp16";
    int device_id = 0;
    bool benchmark = false;
    bool help = false;
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -i, --image <path>      Input image path (required)\n"
              << "  -w, --weights <path>    Weights file path\n"
              << "  -b, --batch <size>      Batch size (default: 1)\n"
              << "  -p, --precision <type>  Precision fp16/fp32 (default: fp16)\n"
              << "  -d, --device <id>       GPU device ID (default: 0)\n"
              << "  --benchmark             Run benchmark mode\n"
              << "  -h, --help              Show this help message\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    
    static struct option long_options[] = {
        {"image", required_argument, 0, 'i'},
        {"weights", required_argument, 0, 'w'},
        {"batch", required_argument, 0, 'b'},
        {"precision", required_argument, 0, 'p'},
        {"device", required_argument, 0, 'd'},
        {"benchmark", no_argument, 0, 0},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "i:w:b:p:d:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'i':
                args.image_path = optarg;
                break;
            case 'w':
                args.weights_path = optarg;
                break;
            case 'b':
                args.batch_size = std::stoi(optarg);
                break;
            case 'p':
                args.precision = optarg;
                break;
            case 'd':
                args.device_id = std::stoi(optarg);
                break;
            case 0:
                if (std::string(long_options[option_index].name) == "benchmark") {
                    args.benchmark = true;
                }
                break;
            case 'h':
                args.help = true;
                break;
            default:
                args.help = true;
                break;
        }
    }
    
    return args;
}

void run_inference(const std::string& image_path, 
                  const ViTWeights& weights,
                  const ViTConfig& config,
                  int batch_size,
                  int device_id,
                  bool benchmark) {
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Create model
    ViTModel model(config);
    model.allocate_workspace(batch_size);
    
    // Load and preprocess image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    // Resize to model input size
    cv::resize(img, img, cv::Size(config.image_size, config.image_size));
    
    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // Create batch by repeating the image
    std::vector<uint8_t> batch_data(batch_size * config.image_size * 
                                   config.image_size * 3);
    for (int b = 0; b < batch_size; ++b) {
        std::memcpy(batch_data.data() + b * img.total() * img.channels(),
                   img.data, img.total() * img.channels());
    }
    
    // Upload to GPU
    Tensor images_uint8({batch_size, config.image_size, config.image_size, 3}, 
                       Tensor::DataType::FP32); // Using FP32 for uint8 data temporarily
    images_uint8.copy_from_host(batch_data.data());
    
    // Convert to normalized FP16
    Tensor images_normalized({batch_size, 3, config.image_size, config.image_size},
                           Tensor::DataType::FP16);
    model.preprocess(images_uint8, images_normalized);
    
    // Allocate output
    Tensor logits({batch_size, config.num_classes}, Tensor::DataType::FP16);
    
    // Warmup
    if (benchmark) {
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 10; ++i) {
            model.forward(images_normalized, weights, logits);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timing
    GPUTimer gpu_timer;
    const int num_runs = benchmark ? 100 : 1;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        gpu_timer.start("inference");
        model.forward(images_normalized, weights, logits);
        gpu_timer.end("inference");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_time_ms = duration.count() / 1000.0f / num_runs;
    
    // Get predictions
    std::vector<__half> logits_host(batch_size * config.num_classes);
    logits.copy_to_host(logits_host.data());
    
    // Find top-1 prediction for first image
    int top_class = 0;
    float top_score = __half2float(logits_host[0]);
    for (int i = 1; i < config.num_classes; ++i) {
        float score = __half2float(logits_host[i]);
        if (score > top_score) {
            top_score = score;
            top_class = i;
        }
    }
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Top-1 class: " << top_class << " (score: " << top_score << ")" << std::endl;
    
    if (benchmark) {
        std::cout << "\nPerformance:" << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Average latency: " << avg_time_ms << " ms" << std::endl;
        std::cout << "Throughput: " << (batch_size * 1000.0f / avg_time_ms) << " images/sec" << std::endl;
        
        gpu_timer.print_summary();
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        
        if (args.help || args.image_path.empty()) {
            print_usage(argv[0]);
            return args.help ? 0 : 1;
        }
        
        // Set precision
        Tensor::DataType dtype = (args.precision == "fp32") ? 
            Tensor::DataType::FP32 : Tensor::DataType::FP16;
        
        // Load or generate weights
        ViTConfig config;
        ViTWeights weights;
        
        if (!args.weights_path.empty()) {
            std::cout << "Loading weights from: " << args.weights_path << std::endl;
            weights = WeightsLoader::load_binary(args.weights_path, config);
        } else {
            std::cout << "No weights provided. Generating random weights..." << std::endl;
            
            // Check if gen_dummy_weights exists and run it
            if (!fs::exists("weights.bin")) {
                std::string gen_path = "./tools/gen_dummy_weights";
                if (fs::exists(gen_path)) {
                    std::cout << "Running " << gen_path << " to generate weights..." << std::endl;
                    std::system(gen_path.c_str());
                }
            }
            
            if (fs::exists("weights.bin")) {
                weights = WeightsLoader::load_binary("weights.bin", config);
            } else {
                weights = WeightsLoader::generate_random(config, dtype);
                WeightsLoader::save_binary("weights.bin", weights, config);
            }
        }
        
        // Run inference
        run_inference(args.image_path, weights, config, args.batch_size, 
                     args.device_id, args.benchmark);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}