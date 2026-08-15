#include "cuda_utils.h"
#include "selective_scan.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct DeviceInputs {
    float* u = nullptr;
    float* delta = nullptr;
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* D = nullptr;
    float* y = nullptr;
    float* state = nullptr;

    ~DeviceInputs() {
        cudaFree(u);
        cudaFree(delta);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(D);
        cudaFree(y);
        cudaFree(state);
    }
};

float measure(DeviceInputs& data,
              const SelectiveScanConfig& config,
              ScanAlgorithm algorithm,
              int iterations) {
    for (int warmup = 0; warmup < 3; ++warmup) {
        selective_scan_forward(data.u, data.delta, data.A, data.B, data.C, data.D,
                               data.y, data.state, config, algorithm);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer timer;
    timer.start();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        selective_scan_forward(data.u, data.delta, data.A, data.B, data.C, data.D,
                               data.y, data.state, config, algorithm);
    }
    return timer.stop() / iterations;
}

const char* name(ScanAlgorithm algorithm) {
    switch (algorithm) {
        case ScanAlgorithm::Naive: return "naive recurrent";
        case ScanAlgorithm::Parallel: return "custom scan";
        case ScanAlgorithm::Cub: return "CUB BlockScan";
        case ScanAlgorithm::Fused: return "fused recurrent";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "A CUDA-capable GPU is required.\n";
        return 1;
    }
    print_device_info();

    std::vector<int> lengths{256, 512, 1024, 2048, 4096};
    if (argc == 3 && std::string(argv[1]) == "--seq") {
        lengths = {std::max(1, std::atoi(argv[2]))};
    }

    std::cout << "Selective scan benchmark (batch=1, dim=64, state=16)\n\n"
              << std::left << std::setw(8) << "seq"
              << std::setw(20) << "algorithm"
              << std::right << std::setw(12) << "latency"
              << std::setw(16) << "tokens/s"
              << std::setw(18) << "workspace" << '\n'
              << std::string(74, '-') << '\n';

    for (int length : lengths) {
        SelectiveScanConfig config;
        config.seq_len = length;
        config.dim = 64;
        config.state_size = 16;

        std::mt19937 generator(123);
        std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
        std::vector<float> u(config.token_elements());
        std::vector<float> delta(config.token_elements(), -2.0f);
        std::vector<float> A(static_cast<size_t>(config.dim) * config.state_size);
        std::vector<float> B(config.parameter_elements());
        std::vector<float> C(config.parameter_elements());
        std::vector<float> D(config.dim, 1.0f);
        for (float& value : u) value = distribution(generator);
        for (float& value : B) value = distribution(generator);
        for (float& value : C) value = distribution(generator);
        for (int channel = 0; channel < config.dim; ++channel) {
            for (int state = 0; state < config.state_size; ++state) {
                A[static_cast<size_t>(channel) * config.state_size + state] = -(state + 1.0f);
            }
        }

        DeviceInputs device;
        device.u = device_alloc<float>(u.size());
        device.delta = device_alloc<float>(delta.size());
        device.A = device_alloc<float>(A.size());
        device.B = device_alloc<float>(B.size());
        device.C = device_alloc<float>(C.size());
        device.D = device_alloc<float>(D.size());
        device.y = device_alloc<float>(u.size());
        device.state = device_alloc<float>(config.state_elements());
        copy_to_device(device.u, u.data(), u.size());
        copy_to_device(device.delta, delta.data(), delta.size());
        copy_to_device(device.A, A.data(), A.size());
        copy_to_device(device.B, B.data(), B.size());
        copy_to_device(device.C, C.data(), C.size());
        copy_to_device(device.D, D.data(), D.size());

        for (ScanAlgorithm algorithm : {ScanAlgorithm::Naive, ScanAlgorithm::Parallel,
                                        ScanAlgorithm::Cub, ScanAlgorithm::Fused}) {
            const int iterations = length >= 2048 ? 10 : 30;
            const float milliseconds = measure(device, config, algorithm, iterations);
            const double tokens_per_second = length / (milliseconds * 1.0e-3);
            const double workspace_mb =
                selective_scan_workspace_bytes(config, algorithm) / (1024.0 * 1024.0);
            std::cout << std::left << std::setw(8) << length
                      << std::setw(20) << name(algorithm)
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(9) << milliseconds << " ms"
                      << std::setw(16) << static_cast<long long>(tokens_per_second)
                      << std::setw(14) << workspace_mb << " MB\n";
        }
        std::cout << '\n';
    }
    return 0;
}
