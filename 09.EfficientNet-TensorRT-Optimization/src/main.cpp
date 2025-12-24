#include "trt_engine.h"
#include "cuda_preprocess.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION_MAIN
#include "stb_image.h"

using namespace efficientnet;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nRequired:\n"
              << "  --model, -m <path>       Path to ONNX model file\n"
              << "  --image, -i <path>       Path to input image\n"
              << "\nOptional:\n"
              << "  --precision, -p <mode>   Precision mode: fp32, fp16, int8 (default: fp16)\n"
              << "  --engine, -e <path>      Path to save/load TensorRT engine\n"
              << "  --labels, -l <path>      Path to class labels file\n"
              << "  --calib, -c <path>       Path to calibration data (required for int8)\n"
              << "  --warmup, -w <n>         Number of warmup iterations (default: 10)\n"
              << "  --repeat, -r <n>         Number of inference iterations (default: 100)\n"
              << "  --topk, -k <n>           Show top K predictions (default: 5)\n"
              << "  --help, -h               Show this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // Default options
    std::string model_path;
    std::string image_path;
    std::string engine_path;
    std::string labels_path;
    std::string calib_path;
    std::string precision_str = "fp16";
    int warmup = 10;
    int repeat = 100;
    int topk = 5;

    // Parse arguments
    static struct option long_options[] = {
        {"model",     required_argument, 0, 'm'},
        {"image",     required_argument, 0, 'i'},
        {"precision", required_argument, 0, 'p'},
        {"engine",    required_argument, 0, 'e'},
        {"labels",    required_argument, 0, 'l'},
        {"calib",     required_argument, 0, 'c'},
        {"warmup",    required_argument, 0, 'w'},
        {"repeat",    required_argument, 0, 'r'},
        {"topk",      required_argument, 0, 'k'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:i:p:e:l:c:w:r:k:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'i': image_path = optarg; break;
            case 'p': precision_str = optarg; break;
            case 'e': engine_path = optarg; break;
            case 'l': labels_path = optarg; break;
            case 'c': calib_path = optarg; break;
            case 'w': warmup = std::stoi(optarg); break;
            case 'r': repeat = std::stoi(optarg); break;
            case 'k': topk = std::stoi(optarg); break;
            case 'h':
            default:
                printUsage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    // Validate required arguments
    if (model_path.empty() || image_path.empty()) {
        std::cerr << "Error: --model and --image are required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Determine precision
    Precision precision = Precision::FP16;
    if (precision_str == "fp32") {
        precision = Precision::FP32;
    } else if (precision_str == "fp16") {
        precision = Precision::FP16;
    } else if (precision_str == "int8") {
        precision = Precision::INT8;
        if (calib_path.empty()) {
            std::cerr << "Error: INT8 precision requires --calib path" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: Invalid precision mode: " << precision_str << std::endl;
        return 1;
    }

    // Generate engine path if not specified
    if (engine_path.empty()) {
        engine_path = model_path + "." + precision_str + ".engine";
    }

    std::cout << "========================================" << std::endl;
    std::cout << "EfficientNet TensorRT Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model:     " << model_path << std::endl;
    std::cout << "Image:     " << image_path << std::endl;
    std::cout << "Precision: " << precision_str << std::endl;
    std::cout << "Engine:    " << engine_path << std::endl;
    std::cout << "========================================" << std::endl;

    // Load class labels
    std::vector<std::string> class_names;
    if (!labels_path.empty()) {
        class_names = loadClassNames(labels_path);
        std::cout << "Loaded " << class_names.size() << " class labels" << std::endl;
    }

    // Create and build engine
    TrtEngine engine;
    EngineConfig config;
    config.onnx_path = model_path;
    config.engine_path = engine_path;
    config.precision = precision;
    config.max_batch_size = 1;
    config.workspace_size = 1ULL << 30;  // 1GB
    config.calibration_data_path = calib_path;

    std::cout << "\nBuilding/Loading TensorRT engine..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();

    if (!engine.buildEngine(config)) {
        std::cerr << "Failed to build engine" << std::endl;
        return 1;
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end - build_start).count();
    std::cout << "Engine ready in " << build_time << " ms" << std::endl;

    // Load and preprocess image
    std::cout << "\nLoading image..." << std::endl;
    int img_width, img_height, img_channels;
    unsigned char* img_data = stbi_load(image_path.c_str(),
                                        &img_width, &img_height, &img_channels, 3);
    if (!img_data) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    std::cout << "Image size: " << img_width << "x" << img_height << "x" << img_channels << std::endl;

    // Preprocess on CPU (for simplicity)
    const int input_size = 224;
    const int num_classes = 1000;
    std::vector<float> input_tensor(3 * input_size * input_size);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    int plane_size = input_size * input_size;
    for (int y = 0; y < input_size; y++) {
        for (int x = 0; x < input_size; x++) {
            float src_x = (x + 0.5f) * img_width / input_size - 0.5f;
            float src_y = (y + 0.5f) * img_height / input_size - 0.5f;

            src_x = std::max(0.0f, std::min(src_x, static_cast<float>(img_width - 1)));
            src_y = std::max(0.0f, std::min(src_y, static_cast<float>(img_height - 1)));

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, img_width - 1);
            int y1 = std::min(y0 + 1, img_height - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < 3; c++) {
                float v00 = img_data[(y0 * img_width + x0) * 3 + c];
                float v01 = img_data[(y0 * img_width + x1) * 3 + c];
                float v10 = img_data[(y1 * img_width + x0) * 3 + c];
                float v11 = img_data[(y1 * img_width + x1) * 3 + c];

                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float val = v0 * (1 - dy) + v1 * dy;

                input_tensor[c * plane_size + y * input_size + x] =
                    (val / 255.0f - mean[c]) / std_dev[c];
            }
        }
    }
    stbi_image_free(img_data);

    // Allocate output
    std::vector<float> output(num_classes);

    // Warmup
    std::cout << "\nWarming up (" << warmup << " iterations)..." << std::endl;
    for (int i = 0; i < warmup; i++) {
        engine.infer(input_tensor.data(), output.data());
    }

    // Benchmark
    std::cout << "Running inference (" << repeat << " iterations)..." << std::endl;
    std::vector<float> latencies;
    latencies.reserve(repeat);

    for (int i = 0; i < repeat; i++) {
        engine.infer(input_tensor.data(), output.data());
        latencies.push_back(engine.getLastInferenceTime());
    }

    // Calculate statistics
    float total = 0.0f;
    float min_lat = latencies[0];
    float max_lat = latencies[0];
    for (float lat : latencies) {
        total += lat;
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
    }
    float avg_lat = total / repeat;

    // Calculate std deviation
    float var = 0.0f;
    for (float lat : latencies) {
        var += (lat - avg_lat) * (lat - avg_lat);
    }
    float std_lat = std::sqrt(var / repeat);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Average latency: " << avg_lat << " ms" << std::endl;
    std::cout << "Min latency:     " << min_lat << " ms" << std::endl;
    std::cout << "Max latency:     " << max_lat << " ms" << std::endl;
    std::cout << "Std deviation:   " << std_lat << " ms" << std::endl;
    std::cout << "Throughput:      " << 1000.0f / avg_lat << " images/sec" << std::endl;

    // Apply softmax
    float max_logit = *std::max_element(output.begin(), output.end());
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[i] = std::exp(output[i] - max_logit);
        sum_exp += output[i];
    }
    for (int i = 0; i < num_classes; i++) {
        output[i] /= sum_exp;
    }

    // Get predictions
    auto predictions = getTopKPredictions(output.data(), num_classes, class_names, topk);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Top " << topk << " Predictions" << std::endl;
    std::cout << "========================================" << std::endl;
    for (int i = 0; i < static_cast<int>(predictions.size()); i++) {
        const auto& pred = predictions[i];
        std::cout << (i + 1) << ". ";
        if (!pred.class_name.empty()) {
            std::cout << pred.class_name;
        } else {
            std::cout << "class_" << pred.class_id;
        }
        std::cout << " (" << pred.class_id << "): "
                  << std::fixed << std::setprecision(2)
                  << pred.confidence * 100.0f << "%" << std::endl;
    }

    return 0;
}
