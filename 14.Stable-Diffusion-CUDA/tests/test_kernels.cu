#include "cuda_utils.h"
#include "diffusion_kernels.cuh"
#include "scheduler.h"
#include <cmath>
#include <numeric>

// ============================================================
// CPU reference implementations
// ============================================================

void cpu_group_norm(const float* in, float* out,
                    const float* gamma, const float* beta,
                    int C, int H, int W, int num_groups, float eps)
{
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;

    for (int g = 0; g < num_groups; g++) {
        // Mean
        float mean = 0.0f;
        for (int c = 0; c < channels_per_group; c++) {
            int ch = g * channels_per_group + c;
            for (int hw = 0; hw < H * W; hw++) {
                mean += in[ch * H * W + hw];
            }
        }
        mean /= group_size;

        // Variance
        float var = 0.0f;
        for (int c = 0; c < channels_per_group; c++) {
            int ch = g * channels_per_group + c;
            for (int hw = 0; hw < H * W; hw++) {
                float diff = in[ch * H * W + hw] - mean;
                var += diff * diff;
            }
        }
        var /= group_size;

        float inv_std = 1.0f / sqrtf(var + eps);

        // Normalize
        for (int c = 0; c < channels_per_group; c++) {
            int ch = g * channels_per_group + c;
            for (int hw = 0; hw < H * W; hw++) {
                int idx = ch * H * W + hw;
                out[idx] = (in[idx] - mean) * inv_std * gamma[ch] + beta[ch];
            }
        }
    }
}

void cpu_silu(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = in[i] / (1.0f + expf(-in[i]));
    }
}

void cpu_gelu(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        float x = in[i];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[i] = x * cdf;
    }
}

void cpu_layer_norm(const float* in, float* out,
                    const float* gamma, const float* beta,
                    int rows, int D, float eps)
{
    for (int r = 0; r < rows; r++) {
        const float* row = in + r * D;
        float* orow = out + r * D;

        float mean = 0.0f;
        for (int i = 0; i < D; i++) mean += row[i];
        mean /= D;

        float var = 0.0f;
        for (int i = 0; i < D; i++) {
            float diff = row[i] - mean;
            var += diff * diff;
        }
        var /= D;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < D; i++) {
            orow[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

void cpu_softmax(float* data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        float max_val = *std::max_element(row, row + cols);
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }
        for (int i = 0; i < cols; i++) {
            row[i] /= sum;
        }
    }
}

// ============================================================
// Test runner
// ============================================================

struct TestResult {
    std::string name;
    bool passed;
    float max_error;
};

float maxAbsError(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        max_err = std::max(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

// ============================================================
// Tests
// ============================================================

TestResult testGroupNorm() {
    int C = 64, H = 8, W = 8, num_groups = 8;
    float eps = 1e-5f;
    int size = C * H * W;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_in(size), h_gamma(C, 1.0f), h_beta(C, 0.0f);
    for (int i = 0; i < size; i++) h_in[i] = dist(rng);
    for (int i = 0; i < C; i++) { h_gamma[i] = 0.5f + dist(rng) * 0.1f; h_beta[i] = dist(rng) * 0.1f; }

    // CPU reference
    std::vector<float> cpu_out(size);
    cpu_group_norm(h_in.data(), cpu_out.data(), h_gamma.data(), h_beta.data(),
                   C, H, W, num_groups, eps);

    // GPU
    half* d_in = cudaMallocDevice<half>(size);
    half* d_out = cudaMallocDevice<half>(size);
    half* d_gamma = cudaMallocDevice<half>(C);
    half* d_beta = cudaMallocDevice<half>(C);

    std::vector<half> h_in_h(size), h_gamma_h(C), h_beta_h(C);
    float_to_half_array(h_in.data(), h_in_h.data(), size);
    float_to_half_array(h_gamma.data(), h_gamma_h.data(), C);
    float_to_half_array(h_beta.data(), h_beta_h.data(), C);
    cudaMemcpyH2D(d_in, h_in_h.data(), size);
    cudaMemcpyH2D(d_gamma, h_gamma_h.data(), C);
    cudaMemcpyH2D(d_beta, h_beta_h.data(), C);

    int threads = std::min(C / num_groups * H * W, 1024);
    group_norm_kernel<<<num_groups, threads>>>(d_out, d_in, d_gamma, d_beta,
                                                C, H, W, num_groups, eps);
    CUDA_CHECK_LAST_ERROR();

    std::vector<half> h_out_h(size);
    std::vector<float> gpu_out(size);
    cudaMemcpyD2H(h_out_h.data(), d_out, size);
    half_to_float_array(h_out_h.data(), gpu_out.data(), size);

    float max_err = maxAbsError(cpu_out.data(), gpu_out.data(), size);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    return {"GroupNorm", max_err < 0.05f, max_err};
}

TestResult testSiLU() {
    int size = 1024;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> h_in(size);
    for (int i = 0; i < size; i++) h_in[i] = dist(rng);

    std::vector<float> cpu_out(size);
    cpu_silu(h_in.data(), cpu_out.data(), size);

    half* d_in = cudaMallocDevice<half>(size);
    half* d_out = cudaMallocDevice<half>(size);
    std::vector<half> h_in_h(size);
    float_to_half_array(h_in.data(), h_in_h.data(), size);
    cudaMemcpyH2D(d_in, h_in_h.data(), size);

    silu_kernel<<<(size + 255) / 256, 256>>>(d_out, d_in, size);
    CUDA_CHECK_LAST_ERROR();

    std::vector<half> h_out_h(size);
    std::vector<float> gpu_out(size);
    cudaMemcpyD2H(h_out_h.data(), d_out, size);
    half_to_float_array(h_out_h.data(), gpu_out.data(), size);

    float max_err = maxAbsError(cpu_out.data(), gpu_out.data(), size);

    cudaFree(d_in); cudaFree(d_out);
    return {"SiLU", max_err < 0.01f, max_err};
}

TestResult testGELU() {
    int size = 1024;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> h_in(size);
    for (int i = 0; i < size; i++) h_in[i] = dist(rng);

    std::vector<float> cpu_out(size);
    cpu_gelu(h_in.data(), cpu_out.data(), size);

    half* d_in = cudaMallocDevice<half>(size);
    half* d_out = cudaMallocDevice<half>(size);
    std::vector<half> h_in_h(size);
    float_to_half_array(h_in.data(), h_in_h.data(), size);
    cudaMemcpyH2D(d_in, h_in_h.data(), size);

    gelu_kernel<<<(size + 255) / 256, 256>>>(d_out, d_in, size);
    CUDA_CHECK_LAST_ERROR();

    std::vector<half> h_out_h(size);
    std::vector<float> gpu_out(size);
    cudaMemcpyD2H(h_out_h.data(), d_out, size);
    half_to_float_array(h_out_h.data(), gpu_out.data(), size);

    float max_err = maxAbsError(cpu_out.data(), gpu_out.data(), size);

    cudaFree(d_in); cudaFree(d_out);
    return {"GELU", max_err < 0.01f, max_err};
}

TestResult testLayerNorm() {
    int rows = 4, D = 128;
    float eps = 1e-5f;
    int size = rows * D;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_in(size), h_gamma(D, 1.0f), h_beta(D, 0.0f);
    for (int i = 0; i < size; i++) h_in[i] = dist(rng);
    for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f + dist(rng) * 0.1f; h_beta[i] = dist(rng) * 0.1f; }

    std::vector<float> cpu_out(size);
    cpu_layer_norm(h_in.data(), cpu_out.data(), h_gamma.data(), h_beta.data(), rows, D, eps);

    half* d_in = cudaMallocDevice<half>(size);
    half* d_out = cudaMallocDevice<half>(size);
    half* d_gamma = cudaMallocDevice<half>(D);
    half* d_beta = cudaMallocDevice<half>(D);

    std::vector<half> h_in_h(size), h_gamma_h(D), h_beta_h(D);
    float_to_half_array(h_in.data(), h_in_h.data(), size);
    float_to_half_array(h_gamma.data(), h_gamma_h.data(), D);
    float_to_half_array(h_beta.data(), h_beta_h.data(), D);
    cudaMemcpyH2D(d_in, h_in_h.data(), size);
    cudaMemcpyH2D(d_gamma, h_gamma_h.data(), D);
    cudaMemcpyH2D(d_beta, h_beta_h.data(), D);

    layer_norm_kernel<<<rows, std::min(D, 1024)>>>(d_out, d_in, d_gamma, d_beta, D, eps);
    CUDA_CHECK_LAST_ERROR();

    std::vector<half> h_out_h(size);
    std::vector<float> gpu_out(size);
    cudaMemcpyD2H(h_out_h.data(), d_out, size);
    half_to_float_array(h_out_h.data(), gpu_out.data(), size);

    float max_err = maxAbsError(cpu_out.data(), gpu_out.data(), size);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    return {"LayerNorm", max_err < 0.01f, max_err};
}

TestResult testSoftmax() {
    int n_heads = 4, seq = 16;
    int size = n_heads * seq * seq;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_scores(size);
    for (int i = 0; i < size; i++) h_scores[i] = dist(rng);

    // CPU reference
    std::vector<float> cpu_scores = h_scores;
    for (int h = 0; h < n_heads; h++) {
        cpu_softmax(cpu_scores.data() + h * seq * seq, seq, seq);
    }

    // GPU
    float* d_scores;
    CUDA_CHECK(cudaMalloc(&d_scores, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(n_heads, seq);
    softmax_2d_kernel<<<grid, std::min(seq, 256)>>>(d_scores, n_heads, seq, seq);
    CUDA_CHECK_LAST_ERROR();

    std::vector<float> gpu_scores(size);
    CUDA_CHECK(cudaMemcpy(gpu_scores.data(), d_scores, size * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = maxAbsError(cpu_scores.data(), gpu_scores.data(), size);

    cudaFree(d_scores);
    return {"Softmax2D", max_err < 1e-5f, max_err};
}

TestResult testUpsample() {
    int C = 4, H = 4, W = 4;
    int in_size = C * H * W;
    int out_size = C * (H * 2) * (W * 2);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_in(in_size);
    for (int i = 0; i < in_size; i++) h_in[i] = dist(rng);

    // CPU reference
    std::vector<float> cpu_out(out_size);
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H * 2; h++) {
            for (int w = 0; w < W * 2; w++) {
                cpu_out[c * (H * 2) * (W * 2) + h * (W * 2) + w] =
                    h_in[c * H * W + (h / 2) * W + (w / 2)];
            }
        }
    }

    // GPU
    half* d_in = cudaMallocDevice<half>(in_size);
    half* d_out = cudaMallocDevice<half>(out_size);
    std::vector<half> h_in_h(in_size);
    float_to_half_array(h_in.data(), h_in_h.data(), in_size);
    cudaMemcpyH2D(d_in, h_in_h.data(), in_size);

    upsample_nearest_2x_kernel<<<(out_size + 255) / 256, 256>>>(d_out, d_in, C, H, W);
    CUDA_CHECK_LAST_ERROR();

    std::vector<half> h_out_h(out_size);
    std::vector<float> gpu_out(out_size);
    cudaMemcpyD2H(h_out_h.data(), d_out, out_size);
    half_to_float_array(h_out_h.data(), gpu_out.data(), out_size);

    float max_err = maxAbsError(cpu_out.data(), gpu_out.data(), out_size);

    cudaFree(d_in); cudaFree(d_out);
    return {"Upsample2x", max_err < 0.001f, max_err};
}

TestResult testDDIMScheduler() {
    SchedulerConfig cfg;
    cfg.num_train_timesteps = 1000;
    cfg.num_inference_steps = 20;
    cfg.beta_start = 0.00085f;
    cfg.beta_end = 0.012f;

    DDIMScheduler scheduler;
    scheduler.init(cfg);
    scheduler.setTimesteps(20);

    // Verify timestep schedule is monotonically decreasing
    bool monotonic = true;
    for (int i = 1; i < 20; i++) {
        if (scheduler.getTimestep(i) >= scheduler.getTimestep(i - 1)) {
            monotonic = false;
            break;
        }
    }

    // Verify first and last timesteps
    bool first_ok = scheduler.getTimestep(0) > 900;  // Should be near 999
    bool last_ok = scheduler.getTimestep(19) < 100;   // Should be near 0

    bool passed = monotonic && first_ok && last_ok;
    float err = passed ? 0.0f : 1.0f;
    return {"DDIMScheduler", passed, err};
}

TestResult testConcatChannels() {
    int Ca = 2, Cb = 3, H = 4, W = 4;
    int size_a = Ca * H * W;
    int size_b = Cb * H * W;
    int size_out = (Ca + Cb) * H * W;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_a(size_a), h_b(size_b);
    for (auto& v : h_a) v = dist(rng);
    for (auto& v : h_b) v = dist(rng);

    // CPU reference
    std::vector<float> cpu_out(size_out);
    for (int c = 0; c < Ca; c++)
        for (int hw = 0; hw < H * W; hw++)
            cpu_out[c * H * W + hw] = h_a[c * H * W + hw];
    for (int c = 0; c < Cb; c++)
        for (int hw = 0; hw < H * W; hw++)
            cpu_out[(Ca + c) * H * W + hw] = h_b[c * H * W + hw];

    // GPU
    half* d_a = cudaMallocDevice<half>(size_a);
    half* d_b = cudaMallocDevice<half>(size_b);
    half* d_out = cudaMallocDevice<half>(size_out);

    std::vector<half> h_a_h(size_a), h_b_h(size_b);
    float_to_half_array(h_a.data(), h_a_h.data(), size_a);
    float_to_half_array(h_b.data(), h_b_h.data(), size_b);
    cudaMemcpyH2D(d_a, h_a_h.data(), size_a);
    cudaMemcpyH2D(d_b, h_b_h.data(), size_b);

    concat_channels_kernel<<<(size_out + 255) / 256, 256>>>(
        d_out, d_a, Ca, d_b, Cb, H, W);
    CUDA_CHECK_LAST_ERROR();

    std::vector<half> h_out_h(size_out);
    std::vector<float> gpu_out(size_out);
    cudaMemcpyD2H(h_out_h.data(), d_out, size_out);
    half_to_float_array(h_out_h.data(), gpu_out.data(), size_out);

    float max_err = maxAbsError(cpu_out.data(), gpu_out.data(), size_out);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return {"ConcatChannels", max_err < 0.001f, max_err};
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("============================================\n");
    printf("  Stable Diffusion CUDA - Kernel Tests\n");
    printf("============================================\n\n");

    printDeviceInfo();

    std::vector<TestResult> results;
    results.push_back(testGroupNorm());
    results.push_back(testSiLU());
    results.push_back(testGELU());
    results.push_back(testLayerNorm());
    results.push_back(testSoftmax());
    results.push_back(testUpsample());
    results.push_back(testDDIMScheduler());
    results.push_back(testConcatChannels());

    printf("\n--- Test Results ---\n");
    int passed = 0, total = (int)results.size();
    for (const auto& r : results) {
        printf("  [Test] %-20s %s  max_err=%.6f\n",
               r.name.c_str(),
               r.passed ? "PASS" : "FAIL",
               r.max_error);
        if (r.passed) passed++;
    }

    printf("\n%d/%d tests passed\n", passed, total);
    printf("============================================\n");

    return (passed == total) ? 0 : 1;
}
