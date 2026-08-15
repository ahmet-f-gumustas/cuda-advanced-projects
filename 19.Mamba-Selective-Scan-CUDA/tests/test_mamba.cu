#include "cuda_utils.h"
#include "mamba_block.h"
#include "mamba_reference.h"
#include "selective_scan.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct ScanFixture {
    SelectiveScanConfig config;
    std::vector<float> u;
    std::vector<float> delta;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> C;
    std::vector<float> D;
    std::vector<float> reference;

    explicit ScanFixture(int length) {
        config.batch = 2;
        config.seq_len = length;
        config.dim = 5;
        config.state_size = 7;
        config.delta_bias = 0.1f;
        u.resize(config.token_elements());
        delta.resize(config.token_elements());
        A.resize(static_cast<size_t>(config.dim) * config.state_size);
        B.resize(config.parameter_elements());
        C.resize(config.parameter_elements());
        D.resize(config.dim);
        reference.resize(config.token_elements());

        std::mt19937 generator(91 + length);
        std::uniform_real_distribution<float> values(-0.35f, 0.35f);
        std::uniform_real_distribution<float> steps(-3.0f, 0.5f);
        for (float& value : u) value = values(generator);
        for (float& value : delta) value = steps(generator);
        for (float& value : B) value = values(generator);
        for (float& value : C) value = values(generator);
        for (float& value : D) value = values(generator);
        for (int channel = 0; channel < config.dim; ++channel) {
            for (int state = 0; state < config.state_size; ++state) {
                A[static_cast<size_t>(channel) * config.state_size + state] =
                    -0.1f * (state + 1);
            }
        }
        selective_scan_reference(u.data(), delta.data(), A.data(), B.data(), C.data(),
                                 D.data(), reference.data(), config);
    }
};

float maximum_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    float error = 0.0f;
    for (size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool run_scan_test(const std::string& label,
                   ScanAlgorithm algorithm,
                   int length,
                   float tolerance) {
    ScanFixture fixture(length);
    float* d_u = device_alloc<float>(fixture.u.size());
    float* d_delta = device_alloc<float>(fixture.delta.size());
    float* d_A = device_alloc<float>(fixture.A.size());
    float* d_B = device_alloc<float>(fixture.B.size());
    float* d_C = device_alloc<float>(fixture.C.size());
    float* d_D = device_alloc<float>(fixture.D.size());
    float* d_y = device_alloc<float>(fixture.u.size());
    float* d_state = algorithm == ScanAlgorithm::Fused
                         ? nullptr
                         : device_alloc<float>(fixture.config.state_elements());
    copy_to_device(d_u, fixture.u.data(), fixture.u.size());
    copy_to_device(d_delta, fixture.delta.data(), fixture.delta.size());
    copy_to_device(d_A, fixture.A.data(), fixture.A.size());
    copy_to_device(d_B, fixture.B.data(), fixture.B.size());
    copy_to_device(d_C, fixture.C.data(), fixture.C.size());
    copy_to_device(d_D, fixture.D.data(), fixture.D.size());

    selective_scan_forward(d_u, d_delta, d_A, d_B, d_C, d_D, d_y, d_state,
                           fixture.config, algorithm);
    std::vector<float> actual(fixture.u.size());
    copy_to_host(actual.data(), d_y, actual.size());
    const float error = maximum_error(fixture.reference, actual);
    const bool passed = error <= tolerance;
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] "
              << std::left << std::setw(31) << label
              << " max_error=" << std::scientific << error << '\n';

    cudaFree(d_u);
    cudaFree(d_delta);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_y);
    cudaFree(d_state);
    return passed;
}

bool run_fp16_test() {
    ScanFixture fixture(73);
    std::vector<__half> u(fixture.u.size());
    std::vector<__half> delta(fixture.delta.size());
    std::vector<__half> A(fixture.A.size());
    std::vector<__half> B(fixture.B.size());
    std::vector<__half> C(fixture.C.size());
    std::vector<__half> D(fixture.D.size());
    std::transform(fixture.u.begin(), fixture.u.end(), u.begin(), __float2half);
    std::transform(fixture.delta.begin(), fixture.delta.end(), delta.begin(), __float2half);
    std::transform(fixture.A.begin(), fixture.A.end(), A.begin(), __float2half);
    std::transform(fixture.B.begin(), fixture.B.end(), B.begin(), __float2half);
    std::transform(fixture.C.begin(), fixture.C.end(), C.begin(), __float2half);
    std::transform(fixture.D.begin(), fixture.D.end(), D.begin(), __float2half);

    __half* d_u = device_alloc<__half>(u.size());
    __half* d_delta = device_alloc<__half>(delta.size());
    __half* d_A = device_alloc<__half>(A.size());
    __half* d_B = device_alloc<__half>(B.size());
    __half* d_C = device_alloc<__half>(C.size());
    __half* d_D = device_alloc<__half>(D.size());
    __half* d_y = device_alloc<__half>(u.size());
    copy_to_device(d_u, u.data(), u.size());
    copy_to_device(d_delta, delta.data(), delta.size());
    copy_to_device(d_A, A.data(), A.size());
    copy_to_device(d_B, B.data(), B.size());
    copy_to_device(d_C, C.data(), C.size());
    copy_to_device(d_D, D.data(), D.size());
    selective_scan_forward_fp16(d_u, d_delta, d_A, d_B, d_C, d_D, d_y,
                                fixture.config);

    std::vector<__half> half_output(u.size());
    copy_to_host(half_output.data(), d_y, half_output.size());
    std::vector<float> output(u.size());
    std::transform(half_output.begin(), half_output.end(), output.begin(), __half2float);
    const float error = maximum_error(fixture.reference, output);
    const bool passed = error <= 2.0e-3f;
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] "
              << std::left << std::setw(31) << "FP16 fused vs CPU"
              << " max_error=" << std::scientific << error << '\n';

    cudaFree(d_u);
    cudaFree(d_delta);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_y);
    return passed;
}

bool run_block_smoke_test() {
    MambaConfig config;
    config.batch = 2;
    config.seq_len = 31;
    config.model_dim = 12;
    config.inner_dim = 24;
    config.state_size = 8;
    config.dt_rank = 4;
    const size_t elements = static_cast<size_t>(config.batch) * config.seq_len * config.model_dim;
    std::vector<float> input(elements);
    std::mt19937 generator(5);
    std::normal_distribution<float> distribution(0.0f, 0.2f);
    for (float& value : input) value = distribution(generator);
    float* d_input = device_alloc<float>(elements);
    float* d_first = device_alloc<float>(elements);
    float* d_second = device_alloc<float>(elements);
    copy_to_device(d_input, input.data(), elements);

    MambaBlock block(config);
    block.forward(d_input, d_first);
    block.forward(d_input, d_second);
    std::vector<float> first(elements);
    std::vector<float> second(elements);
    copy_to_host(first.data(), d_first, elements);
    copy_to_host(second.data(), d_second, elements);
    bool finite = true;
    bool changed = false;
    for (size_t index = 0; index < elements; ++index) {
        finite = finite && std::isfinite(first[index]);
        changed = changed || std::abs(first[index] - input[index]) > 1.0e-6f;
    }
    const float repeat_error = maximum_error(first, second);
    const bool passed = finite && changed && repeat_error == 0.0f;
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] "
              << std::left << std::setw(31) << "complete Mamba block"
              << " repeat_error=" << std::scientific << repeat_error << '\n';
    cudaFree(d_input);
    cudaFree(d_first);
    cudaFree(d_second);
    return passed;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA-capable GPU is available.\n";
        return 77;
    }

    print_device_info();
    int failures = 0;
    failures += !run_scan_test("naive recurrent vs CPU", ScanAlgorithm::Naive, 37, 2.0e-5f);
    failures += !run_scan_test("custom scan vs CPU", ScanAlgorithm::Parallel, 37, 3.0e-5f);
    failures += !run_scan_test("custom scan cross-tile", ScanAlgorithm::Parallel, 513, 2.0e-4f);
    failures += !run_scan_test("CUB BlockScan vs CPU", ScanAlgorithm::Cub, 513, 2.0e-4f);
    failures += !run_scan_test("fused recurrent vs CPU", ScanAlgorithm::Fused, 73, 3.0e-5f);
    failures += !run_fp16_test();
    failures += !run_block_smoke_test();

    if (failures == 0) {
        std::cout << "All 7 tests passed.\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed.\n";
    return 1;
}
