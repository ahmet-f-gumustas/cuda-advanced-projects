#include "common/logger.h"
#include <iostream>
#include <argparse/argparse.hpp>

// Forward declaration
extern "C" void run_cudnn_mbconv_benchmark(int batch_size, int height, int width,
                                          int in_channels, int out_channels,
                                          int warmup, int runs);

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("bench_cudnn");
    
    program.add_argument("--batch-size")
        .default_value(1)
        .scan<'i', int>()
        .help("Batch size");
    
    program.add_argument("--height")
        .default_value(224)
        .scan<'i', int>()
        .help("Input height");
    
    program.add_argument("--width")
        .default_value(224)
        .scan<'i', int>()
        .help("Input width");
    
    program.add_argument("--in-channels")
        .default_value(32)
        .scan<'i', int>()
        .help("Input channels");
    
    program.add_argument("--out-channels")
        .default_value(32)
        .scan<'i', int>()
        .help("Output channels");
    
    program.add_argument("--warmup")
        .default_value(10)
        .scan<'i', int>()
        .help("Number of warmup iterations");
    
    program.add_argument("--runs")
        .default_value(100)
        .scan<'i', int>()
        .help("Number of timing iterations");
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    
    std::cout << "\n=== cuDNN MBConv Benchmark ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size:    " << program.get<int>("--batch-size") << std::endl;
    std::cout << "  Input shape:   " << program.get<int>("--height") << "x" 
              << program.get<int>("--width") << std::endl;
    std::cout << "  Channels:      " << program.get<int>("--in-channels") << " -> "
              << program.get<int>("--out-channels") << std::endl;
    std::cout << "  Warmup runs:   " << program.get<int>("--warmup") << std::endl;
    std::cout << "  Timing runs:   " << program.get<int>("--runs") << std::endl;
    std::cout << std::endl;
    
    try {
        run_cudnn_mbconv_benchmark(
            program.get<int>("--batch-size"),
            program.get<int>("--height"),
            program.get<int>("--width"),
            program.get<int>("--in-channels"),
            program.get<int>("--out-channels"),
            program.get<int>("--warmup"),
            program.get<int>("--runs")
        );
    } catch (const std::exception& e) {
        LOG_ERROR("Benchmark failed:", e.what());
        return 1;
    }
    
    return 0;
}