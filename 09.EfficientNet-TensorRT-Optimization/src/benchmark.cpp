#include "trt_engine.h"
#include "cuda_preprocess.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <getopt.h>

using namespace efficientnet;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nRequired:\n"
              << "  --model, -m <path>       Path to ONNX model file\n"
              << "\nOptional:\n"
              << "  --batch, -b <n>          Batch size (default: 1)\n"
              << "  --warmup, -w <n>         Warmup iterations (default: 50)\n"
              << "  --repeat, -r <n>         Benchmark iterations (default: 500)\n"
              << "  --compare, -c            Compare all precision modes\n"
              << "  --calib, -C <path>       Calibration data path for INT8\n"
              << "  --help, -h               Show this help message\n"
              << std::endl;
}

struct BenchmarkResult {
    std::string name;
    float avg_latency_ms;
    float min_latency_ms;
    float max_latency_ms;
    float std_dev_ms;
    float throughput;
    float p50_latency_ms;
    float p95_latency_ms;
    float p99_latency_ms;
};

BenchmarkResult runBenchmark(
    TrtEngine& engine,
    const std::string& name,
    int batch_size,
    int warmup,
    int repeat
) {
    const int input_size = 224;
    const int num_classes = 1000;

    // Generate random input
    std::vector<float> input(batch_size * 3 * input_size * input_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : input) {
        val = dist(gen);
    }

    std::vector<float> output(batch_size * num_classes);

    // Warmup
    std::cout << "  Warming up (" << warmup << " iterations)..." << std::flush;
    for (int i = 0; i < warmup; i++) {
        engine.infer(input.data(), output.data(), batch_size);
    }
    std::cout << " done" << std::endl;

    // Benchmark
    std::cout << "  Running benchmark (" << repeat << " iterations)..." << std::flush;
    std::vector<float> latencies;
    latencies.reserve(repeat);

    for (int i = 0; i < repeat; i++) {
        engine.infer(input.data(), output.data(), batch_size);
        latencies.push_back(engine.getLastInferenceTime());
    }
    std::cout << " done" << std::endl;

    // Sort for percentiles
    std::vector<float> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    // Calculate statistics
    BenchmarkResult result;
    result.name = name;

    float total = 0.0f;
    result.min_latency_ms = latencies[0];
    result.max_latency_ms = latencies[0];

    for (float lat : latencies) {
        total += lat;
        result.min_latency_ms = std::min(result.min_latency_ms, lat);
        result.max_latency_ms = std::max(result.max_latency_ms, lat);
    }

    result.avg_latency_ms = total / repeat;

    // Std deviation
    float var = 0.0f;
    for (float lat : latencies) {
        var += (lat - result.avg_latency_ms) * (lat - result.avg_latency_ms);
    }
    result.std_dev_ms = std::sqrt(var / repeat);

    // Percentiles
    result.p50_latency_ms = sorted_latencies[repeat / 2];
    result.p95_latency_ms = sorted_latencies[static_cast<int>(repeat * 0.95)];
    result.p99_latency_ms = sorted_latencies[static_cast<int>(repeat * 0.99)];

    // Throughput
    result.throughput = 1000.0f * batch_size / result.avg_latency_ms;

    return result;
}

void printResult(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::left << std::setw(12) << r.name << ": "
              << std::setw(8) << r.avg_latency_ms << " ms avg, "
              << std::setw(8) << r.min_latency_ms << " ms min, "
              << std::setw(8) << r.max_latency_ms << " ms max, "
              << std::setw(8) << r.throughput << " img/s" << std::endl;
}

void printDetailedResult(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n" << r.name << " Results:" << std::endl;
    std::cout << "  Average:    " << std::setw(10) << r.avg_latency_ms << " ms" << std::endl;
    std::cout << "  Min:        " << std::setw(10) << r.min_latency_ms << " ms" << std::endl;
    std::cout << "  Max:        " << std::setw(10) << r.max_latency_ms << " ms" << std::endl;
    std::cout << "  Std Dev:    " << std::setw(10) << r.std_dev_ms << " ms" << std::endl;
    std::cout << "  P50:        " << std::setw(10) << r.p50_latency_ms << " ms" << std::endl;
    std::cout << "  P95:        " << std::setw(10) << r.p95_latency_ms << " ms" << std::endl;
    std::cout << "  P99:        " << std::setw(10) << r.p99_latency_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setw(10) << r.throughput << " images/sec" << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string calib_path;
    int batch_size = 1;
    int warmup = 50;
    int repeat = 500;
    bool compare_all = false;

    static struct option long_options[] = {
        {"model",   required_argument, 0, 'm'},
        {"batch",   required_argument, 0, 'b'},
        {"warmup",  required_argument, 0, 'w'},
        {"repeat",  required_argument, 0, 'r'},
        {"compare", no_argument,       0, 'c'},
        {"calib",   required_argument, 0, 'C'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:b:w:r:cC:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'b': batch_size = std::stoi(optarg); break;
            case 'w': warmup = std::stoi(optarg); break;
            case 'r': repeat = std::stoi(optarg); break;
            case 'c': compare_all = true; break;
            case 'C': calib_path = optarg; break;
            case 'h':
            default:
                printUsage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "EfficientNet TensorRT Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model:       " << model_path << std::endl;
    std::cout << "Batch size:  " << batch_size << std::endl;
    std::cout << "Warmup:      " << warmup << std::endl;
    std::cout << "Iterations:  " << repeat << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<BenchmarkResult> results;

    if (compare_all) {
        // Compare all precision modes
        std::vector<std::pair<std::string, Precision>> modes = {
            {"FP32", Precision::FP32},
            {"FP16", Precision::FP16}
        };

        if (!calib_path.empty()) {
            modes.push_back({"INT8", Precision::INT8});
        }

        for (const auto& mode : modes) {
            std::cout << "\n--- Building " << mode.first << " engine ---" << std::endl;

            TrtEngine engine;
            EngineConfig config;
            config.onnx_path = model_path;
            config.engine_path = model_path + "." + mode.first + ".engine";
            config.precision = mode.second;
            config.max_batch_size = batch_size;
            config.workspace_size = 1ULL << 30;
            config.calibration_data_path = calib_path;

            if (!engine.buildEngine(config)) {
                std::cerr << "Failed to build " << mode.first << " engine" << std::endl;
                continue;
            }

            auto result = runBenchmark(engine, mode.first, batch_size, warmup, repeat);
            results.push_back(result);
        }

        // Print comparison
        std::cout << "\n========================================" << std::endl;
        std::cout << "Precision Comparison" << std::endl;
        std::cout << "========================================" << std::endl;

        for (const auto& r : results) {
            printResult(r);
        }

        // Speedup comparison
        if (results.size() >= 2) {
            float fp32_latency = results[0].avg_latency_ms;
            std::cout << "\nSpeedup vs FP32:" << std::endl;
            for (size_t i = 1; i < results.size(); i++) {
                float speedup = fp32_latency / results[i].avg_latency_ms;
                std::cout << "  " << results[i].name << ": "
                          << std::setprecision(2) << speedup << "x" << std::endl;
            }
        }

    } else {
        // Single precision benchmark (FP16 default)
        TrtEngine engine;
        EngineConfig config;
        config.onnx_path = model_path;
        config.engine_path = model_path + ".fp16.engine";
        config.precision = Precision::FP16;
        config.max_batch_size = batch_size;
        config.workspace_size = 1ULL << 30;

        std::cout << "\n--- Building FP16 engine ---" << std::endl;
        if (!engine.buildEngine(config)) {
            std::cerr << "Failed to build engine" << std::endl;
            return 1;
        }

        auto result = runBenchmark(engine, "FP16", batch_size, warmup, repeat);
        printDetailedResult(result);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
