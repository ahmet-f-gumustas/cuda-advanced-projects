#include "cuda_utils.h"
#include "mamba_block.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int integer_argument(int argc, char** argv, const std::string& option, int fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == option) {
            return std::max(1, std::atoi(argv[index + 1]));
        }
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "A CUDA-capable GPU is required.\n";
        return 1;
    }

    MambaConfig config;
    config.seq_len = integer_argument(argc, argv, "--seq", 256);
    config.model_dim = integer_argument(argc, argv, "--dim", 64);
    config.inner_dim = integer_argument(argc, argv, "--inner", 2 * config.model_dim);
    config.state_size = integer_argument(argc, argv, "--state", 16);
    config.dt_rank = integer_argument(argc, argv, "--dt-rank", 8);

    print_device_info();
    std::cout << "Mamba block: batch=" << config.batch
              << " seq=" << config.seq_len
              << " model_dim=" << config.model_dim
              << " inner_dim=" << config.inner_dim
              << " state=" << config.state_size << "\n";

    const size_t elements = static_cast<size_t>(config.batch) * config.seq_len *
                            config.model_dim;
    std::vector<float> input(elements);
    std::vector<float> output(elements);
    std::mt19937 generator(7);
    std::normal_distribution<float> distribution(0.0f, 0.3f);
    for (float& value : input) {
        value = distribution(generator);
    }

    float* d_input = device_alloc<float>(elements);
    float* d_output = device_alloc<float>(elements);
    copy_to_device(d_input, input.data(), elements);

    MambaBlock block(config);
    block.forward(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int iterations = 20;
    CudaTimer timer;
    timer.start();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        block.forward(d_input, d_output);
    }
    const float latency_ms = timer.stop() / iterations;
    copy_to_host(output.data(), d_output, elements);

    double checksum = 0.0;
    for (float value : output) {
        checksum += value;
    }
    std::cout << std::fixed << std::setprecision(3)
              << "Latency: " << latency_ms << " ms\n"
              << "Throughput: " << config.seq_len / (latency_ms * 1.0e-3f)
              << " tokens/s\n"
              << std::setprecision(6) << "Output checksum: " << checksum << "\n"
              << "First token: [";
    for (int channel = 0; channel < std::min(config.model_dim, 8); ++channel) {
        std::cout << (channel == 0 ? "" : ", ") << output[channel];
    }
    std::cout << "]\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
