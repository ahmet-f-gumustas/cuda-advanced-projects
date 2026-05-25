// Per-kernel micro-benchmark: runs v1 and v2 of each optimized kernel
// side-by-side, verifies output equivalence on a tolerance, then reports
// mean / p50 / p99 latency.
//
// Run: ./build/kernel_bench [--iters N] [--warmup N]
//
// Each kernel gets:
//   * One verification pass with diff tolerance (max abs / max rel)
//   * `iters` timed launches with cudaEvent timing
//   * v2 speedup reported as v1 / v2

#include "yolo_kernels.cuh"
#include "cuda_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <random>
#include <string>
#include <cmath>

// ============================================================
// Generic timing + reporting
// ============================================================

struct BenchStats {
    float mean = 0.0f, p50 = 0.0f, p99 = 0.0f;
};

static BenchStats compute_stats(std::vector<float> ts) {
    BenchStats s;
    if (ts.empty()) return s;
    double sum = 0.0;
    for (float t : ts) sum += t;
    s.mean = (float)(sum / ts.size());
    std::sort(ts.begin(), ts.end());
    s.p50 = ts[ts.size() / 2];
    s.p99 = ts[std::min(ts.size() - 1, (size_t)(ts.size() * 0.99))];
    return s;
}

struct BenchResult {
    const char* name;
    const char* shape;
    BenchStats v1, v2;
    float max_abs_diff;
    bool ok;
};

static std::vector<BenchResult> g_results;

static void print_header() {
    printf("\n%-32s %-22s %12s %12s %12s %8s\n",
           "Kernel", "Shape", "v1 (ms)", "v2 (ms)", "speedup", "diff");
    printf("%s\n", std::string(102, '-').c_str());
}

static void record(const char* name, const char* shape,
                   BenchStats v1, BenchStats v2, float diff, bool ok) {
    g_results.push_back({name, shape, v1, v2, diff, ok});
    float sp = (v2.mean > 0.0f) ? (v1.mean / v2.mean) : 0.0f;
    printf("%-32s %-22s %5.4f/%5.4f %5.4f/%5.4f %11.2fx %8.1e %s\n",
           name, shape, v1.mean, v1.p99, v2.mean, v2.p99, sp, diff,
           ok ? "" : " [MISMATCH]");
}

// Time a kernel `iters` times after `warmup` warmups; sync at the end of each
// timed launch. Returns vector of per-iteration ms.
template <typename LaunchFn>
static std::vector<float> time_kernel(LaunchFn launch, int iters, int warmup) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int i = 0; i < warmup; ++i) launch();
    cudaDeviceSynchronize();
    std::vector<float> out;
    out.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(s);
        launch();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        out.push_back(ms);
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return out;
}

// Compute max abs diff between two device buffers (float).
static float max_abs_diff_f(const float* da, const float* db, size_t n) {
    std::vector<float> a(n), b(n);
    cudaMemcpy(a.data(), da, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), db, sizeof(float) * n, cudaMemcpyDeviceToHost);
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ============================================================
// Preprocess
// ============================================================

static void bench_hwc_to_chw(int iters, int warmup) {
    const int W = 640, H = 640;
    const int N = W * H * 3;
    std::vector<uint8_t> h(N);
    std::mt19937 rng(0);
    for (auto& v : h) v = (uint8_t)(rng() & 0xff);

    uint8_t* d_src; float *d_a, *d_b;
    cudaMalloc(&d_src, N);
    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMalloc(&d_b, sizeof(float) * N);
    cudaMemcpy(d_src, h.data(), N, cudaMemcpyHostToDevice);

    launch_hwc_uint8_to_chw_float   (d_src, d_a, W, H);
    launch_hwc_uint8_to_chw_float_v2(d_src, d_b, W, H);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_a, d_b, N);

    auto v1 = time_kernel([&] { launch_hwc_uint8_to_chw_float   (d_src, d_a, W, H); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_hwc_uint8_to_chw_float_v2(d_src, d_b, W, H); }, iters, warmup);
    record("hwc_uint8_to_chw_float", "640x640", compute_stats(v1), compute_stats(v2),
           diff, diff < 1e-6f);

    cudaFree(d_src); cudaFree(d_a); cudaFree(d_b);
}

static void bench_bgr_to_rgb_chw(int iters, int warmup) {
    const int W = 1280, H = 720;
    const int N = W * H * 3;
    std::vector<uint8_t> h(N);
    std::mt19937 rng(1);
    for (auto& v : h) v = (uint8_t)(rng() & 0xff);

    uint8_t* d_src; float* d_out;
    cudaMalloc(&d_src, N);
    cudaMalloc(&d_out, sizeof(float) * N);
    cudaMemcpy(d_src, h.data(), N, cudaMemcpyHostToDevice);

    // Reference: host BGR->RGB then HWC->CHW (matches what yolo_camera v1 does).
    std::vector<uint8_t> rgb(N);
    for (int i = 0; i < W * H; ++i) {
        rgb[i * 3 + 0] = h[i * 3 + 2];
        rgb[i * 3 + 1] = h[i * 3 + 1];
        rgb[i * 3 + 2] = h[i * 3 + 0];
    }
    uint8_t* d_rgb;
    cudaMalloc(&d_rgb, N);
    cudaMemcpy(d_rgb, rgb.data(), N, cudaMemcpyHostToDevice);
    float* d_ref;
    cudaMalloc(&d_ref, sizeof(float) * N);
    launch_hwc_uint8_to_chw_float(d_rgb, d_ref, W, H);
    launch_bgr_uint8_to_rgb_chw_float(d_src, d_out, W, H);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_ref, d_out, N);

    // v1 = host BGR->RGB + memcpy + hwc->chw kernel (we approximate by timing only the GPU half).
    auto v1 = time_kernel([&] {
        // emulate v1: assume already-RGB input on device; this isolates the kernel.
        launch_hwc_uint8_to_chw_float(d_rgb, d_ref, W, H);
    }, iters, warmup);
    auto v2 = time_kernel([&] {
        launch_bgr_uint8_to_rgb_chw_float(d_src, d_out, W, H);
    }, iters, warmup);
    record("bgr_uint8_to_rgb_chw (fused)", "1280x720", compute_stats(v1), compute_stats(v2),
           diff, diff < 1e-6f);

    cudaFree(d_src); cudaFree(d_out); cudaFree(d_rgb); cudaFree(d_ref);
}

// ============================================================
// Activations
// ============================================================

static void bench_silu(int iters, int warmup) {
    const int N = 1 * 32 * 320 * 320;   // 3.2M, realistic backbone tensor size
    std::vector<float> h(N);
    std::mt19937 rng(2);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto& v : h) v = d(rng);

    float *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMalloc(&d_b, sizeof(float) * N);
    cudaMemcpy(d_a, h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    launch_silu   (d_a, N);
    launch_silu_v2(d_b, N);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_a, d_b, N);

    // Reset for fair timing (silu modifies in place; doesn't matter for timing).
    auto v1 = time_kernel([&] { cudaMemcpyAsync(d_a, h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
                                launch_silu(d_a, N); }, iters, warmup);
    auto v2 = time_kernel([&] { cudaMemcpyAsync(d_b, h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
                                launch_silu_v2(d_b, N); }, iters, warmup);
    // Subtract the host->device copy: we time silu separately below as well to compare cleanly.
    auto v1_pure = time_kernel([&] { launch_silu(d_a, N); }, iters, warmup);
    auto v2_pure = time_kernel([&] { launch_silu_v2(d_b, N); }, iters, warmup);
    (void)v1; (void)v2;
    record("silu", "1x32x320x320", compute_stats(v1_pure), compute_stats(v2_pure),
           diff, diff < 5e-4f);

    cudaFree(d_a); cudaFree(d_b);
}

static void bench_bn_silu(int iters, int warmup) {
    const int N = 1, C = 64, H = 80, W = 80;   // realistic P3-level tensor
    const int total = N * C * H * W;
    std::vector<float> h(total);
    std::mt19937 rng(3);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto& v : h) v = d(rng);
    std::vector<float> mean(C), rstd(C), gamma(C), beta(C);
    for (int c = 0; c < C; ++c) {
        mean[c]  = 0.05f * c;
        rstd[c]  = 1.0f / std::sqrt(1.0f + 0.01f * c);
        gamma[c] = 0.5f + 0.5f * (c % 4);
        beta[c]  = -0.1f * c;
    }

    float *d_a, *d_b, *d_mean, *d_rstd, *d_gamma, *d_beta;
    cudaMalloc(&d_a, sizeof(float) * total);
    cudaMalloc(&d_b, sizeof(float) * total);
    cudaMalloc(&d_mean, sizeof(float) * C);
    cudaMalloc(&d_rstd, sizeof(float) * C);
    cudaMalloc(&d_gamma, sizeof(float) * C);
    cudaMalloc(&d_beta, sizeof(float) * C);
    cudaMemcpy(d_a, h.data(), sizeof(float) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h.data(), sizeof(float) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,  mean.data(),  sizeof(float) * C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rstd,  rstd.data(),  sizeof(float) * C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), sizeof(float) * C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  beta.data(),  sizeof(float) * C, cudaMemcpyHostToDevice);

    launch_bn_silu   (d_a, d_mean, d_rstd, d_gamma, d_beta, N, C, H, W);
    launch_bn_silu_v2(d_b, d_mean, d_rstd, d_gamma, d_beta, N, C, H, W);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_a, d_b, total);

    auto v1 = time_kernel([&] {
        cudaMemcpyAsync(d_a, h.data(), sizeof(float) * total, cudaMemcpyHostToDevice);
        launch_bn_silu(d_a, d_mean, d_rstd, d_gamma, d_beta, N, C, H, W);
    }, iters, warmup);
    auto v2 = time_kernel([&] {
        cudaMemcpyAsync(d_b, h.data(), sizeof(float) * total, cudaMemcpyHostToDevice);
        launch_bn_silu_v2(d_b, d_mean, d_rstd, d_gamma, d_beta, N, C, H, W);
    }, iters, warmup);
    auto v1_pure = time_kernel([&] { launch_bn_silu   (d_a, d_mean, d_rstd, d_gamma, d_beta, N, C, H, W); }, iters, warmup);
    auto v2_pure = time_kernel([&] { launch_bn_silu_v2(d_b, d_mean, d_rstd, d_gamma, d_beta, N, C, H, W); }, iters, warmup);
    (void)v1; (void)v2;
    record("bn_silu", "1x64x80x80", compute_stats(v1_pure), compute_stats(v2_pure),
           diff, diff < 5e-4f);

    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_mean); cudaFree(d_rstd); cudaFree(d_gamma); cudaFree(d_beta);
}

// ============================================================
// Topology
// ============================================================

static void bench_concat_channel(int iters, int warmup) {
    const int N = 1, CA = 128, CB = 128, H = 40, W = 40;   // realistic neck-cat
    int total = N * (CA + CB) * H * W;
    int na = N * CA * H * W, nb = N * CB * H * W;
    std::vector<float> ha(na), hb(nb);
    std::mt19937 rng(4);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto& v : ha) v = d(rng);
    for (auto& v : hb) v = d(rng);
    float *d_a, *d_b, *d_o1, *d_o2;
    cudaMalloc(&d_a, sizeof(float) * na);
    cudaMalloc(&d_b, sizeof(float) * nb);
    cudaMalloc(&d_o1, sizeof(float) * total);
    cudaMalloc(&d_o2, sizeof(float) * total);
    cudaMemcpy(d_a, ha.data(), sizeof(float) * na, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), sizeof(float) * nb, cudaMemcpyHostToDevice);
    launch_concat_channel   (d_a, CA, d_b, CB, d_o1, N, H, W);
    launch_concat_channel_v2(d_a, CA, d_b, CB, d_o2, N, H, W);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_o1, d_o2, total);

    auto v1 = time_kernel([&] { launch_concat_channel   (d_a, CA, d_b, CB, d_o1, N, H, W); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_concat_channel_v2(d_a, CA, d_b, CB, d_o2, N, H, W); }, iters, warmup);
    record("concat_channel", "256x40x40", compute_stats(v1), compute_stats(v2),
           diff, diff == 0.0f);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_o1); cudaFree(d_o2);
}

static void bench_upsample_2x(int iters, int warmup) {
    const int N = 1, C = 128, H = 20, W = 20;
    int in_n = N * C * H * W;
    int out_n = in_n * 4;
    std::vector<float> hin(in_n);
    std::mt19937 rng(5);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto& v : hin) v = d(rng);
    float *d_in, *d_o1, *d_o2;
    cudaMalloc(&d_in, sizeof(float) * in_n);
    cudaMalloc(&d_o1, sizeof(float) * out_n);
    cudaMalloc(&d_o2, sizeof(float) * out_n);
    cudaMemcpy(d_in, hin.data(), sizeof(float) * in_n, cudaMemcpyHostToDevice);
    launch_upsample_nearest_2x   (d_in, d_o1, N, C, H, W);
    launch_upsample_nearest_2x_v2(d_in, d_o2, N, C, H, W);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_o1, d_o2, out_n);

    auto v1 = time_kernel([&] { launch_upsample_nearest_2x   (d_in, d_o1, N, C, H, W); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_upsample_nearest_2x_v2(d_in, d_o2, N, C, H, W); }, iters, warmup);
    record("upsample_nearest_2x", "1x128x20x20", compute_stats(v1), compute_stats(v2),
           diff, diff == 0.0f);
    cudaFree(d_in); cudaFree(d_o1); cudaFree(d_o2);
}

static void bench_maxpool_sppf(int iters, int warmup) {
    // SPPF input: 128ch, 20x20, k=5
    const int N = 1, C = 128, H = 20, W = 20, K = 5;
    int n = N * C * H * W;
    std::vector<float> hin(n);
    std::mt19937 rng(6);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto& v : hin) v = d(rng);
    float *d_in, *d_o1, *d_o2;
    cudaMalloc(&d_in, sizeof(float) * n);
    cudaMalloc(&d_o1, sizeof(float) * n);
    cudaMalloc(&d_o2, sizeof(float) * n);
    cudaMemcpy(d_in, hin.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    launch_maxpool2d_same   (d_in, d_o1, N, C, H, W, K);
    launch_maxpool2d_same_v2(d_in, d_o2, N, C, H, W, K);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_o1, d_o2, n);

    auto v1 = time_kernel([&] { launch_maxpool2d_same   (d_in, d_o1, N, C, H, W, K); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_maxpool2d_same_v2(d_in, d_o2, N, C, H, W, K); }, iters, warmup);
    record("maxpool2d_same k=5 (SPPF)", "1x128x20x20", compute_stats(v1), compute_stats(v2),
           diff, diff == 0.0f);
    cudaFree(d_in); cudaFree(d_o1); cudaFree(d_o2);
}

// ============================================================
// Postprocess
// ============================================================

static void bench_dfl(int iters, int warmup) {
    const int N = 1, A = 8400, REG = 16;
    int n = N * A * 4 * REG;
    std::vector<float> hin(n);
    std::mt19937 rng(7);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto& v : hin) v = d(rng);
    float *d_in, *d_o1, *d_o2;
    cudaMalloc(&d_in, sizeof(float) * n);
    cudaMalloc(&d_o1, sizeof(float) * N * A * 4);
    cudaMalloc(&d_o2, sizeof(float) * N * A * 4);
    cudaMemcpy(d_in, hin.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    launch_dfl_decode   (d_in, d_o1, N, A, REG);
    launch_dfl_decode_v2(d_in, d_o2, N, A, REG);
    cudaDeviceSynchronize();
    float diff = max_abs_diff_f(d_o1, d_o2, N * A * 4);

    auto v1 = time_kernel([&] { launch_dfl_decode   (d_in, d_o1, N, A, REG); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_dfl_decode_v2(d_in, d_o2, N, A, REG); }, iters, warmup);
    record("dfl_decode", "anchors=8400 reg=16", compute_stats(v1), compute_stats(v2),
           diff, diff < 1e-3f);
    cudaFree(d_in); cudaFree(d_o1); cudaFree(d_o2);
}

static void bench_score_filter(int iters, int warmup) {
    const int N_IN = 8400;
    const int MAX_OUT = 1024;
    const float THRESH = 0.25f;
    std::vector<float> hboxes(N_IN * 4), hscores(N_IN);
    std::vector<int> hcls(N_IN);
    std::mt19937 rng(8);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (int i = 0; i < N_IN; ++i) {
        hboxes[i * 4 + 0] = u01(rng) * 500.0f;
        hboxes[i * 4 + 1] = u01(rng) * 500.0f;
        hboxes[i * 4 + 2] = hboxes[i * 4 + 0] + 20.0f + u01(rng) * 50.0f;
        hboxes[i * 4 + 3] = hboxes[i * 4 + 1] + 20.0f + u01(rng) * 50.0f;
        hscores[i] = u01(rng);   // ~ half above threshold
        hcls[i]    = (int)(u01(rng) * 80.0f);
    }
    float *d_b, *d_s, *d_bo1, *d_bo2, *d_so1, *d_so2;
    int *d_c, *d_co1, *d_co2, *d_cnt1, *d_cnt2;
    cudaMalloc(&d_b, sizeof(float) * N_IN * 4);
    cudaMalloc(&d_s, sizeof(float) * N_IN);
    cudaMalloc(&d_c, sizeof(int)   * N_IN);
    cudaMalloc(&d_bo1, sizeof(float) * MAX_OUT * 4);
    cudaMalloc(&d_bo2, sizeof(float) * MAX_OUT * 4);
    cudaMalloc(&d_so1, sizeof(float) * MAX_OUT);
    cudaMalloc(&d_so2, sizeof(float) * MAX_OUT);
    cudaMalloc(&d_co1, sizeof(int)   * MAX_OUT);
    cudaMalloc(&d_co2, sizeof(int)   * MAX_OUT);
    cudaMalloc(&d_cnt1, sizeof(int));
    cudaMalloc(&d_cnt2, sizeof(int));
    cudaMemcpy(d_b, hboxes.data(),  sizeof(float) * N_IN * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, hscores.data(), sizeof(float) * N_IN,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, hcls.data(),    sizeof(int)   * N_IN,     cudaMemcpyHostToDevice);

    launch_score_filter   (d_b, d_s, d_c, d_bo1, d_so1, d_co1, d_cnt1, N_IN, THRESH, MAX_OUT);
    launch_score_filter_v2(d_b, d_s, d_c, d_bo2, d_so2, d_co2, d_cnt2, N_IN, THRESH, MAX_OUT);
    cudaDeviceSynchronize();
    int cnt1 = 0, cnt2 = 0;
    cudaMemcpy(&cnt1, d_cnt1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cnt2, d_cnt2, sizeof(int), cudaMemcpyDeviceToHost);
    // Output order is implementation-defined for both; we only verify counts.
    float diff = (float)std::abs(cnt1 - cnt2);

    auto v1 = time_kernel([&] {
        launch_score_filter(d_b, d_s, d_c, d_bo1, d_so1, d_co1, d_cnt1, N_IN, THRESH, MAX_OUT);
    }, iters, warmup);
    auto v2 = time_kernel([&] {
        launch_score_filter_v2(d_b, d_s, d_c, d_bo2, d_so2, d_co2, d_cnt2, N_IN, THRESH, MAX_OUT);
    }, iters, warmup);
    record("score_filter (warp-agg atomic)", "n=8400 keep~half",
           compute_stats(v1), compute_stats(v2), diff, cnt1 == cnt2);

    cudaFree(d_b); cudaFree(d_s); cudaFree(d_c);
    cudaFree(d_bo1); cudaFree(d_bo2); cudaFree(d_so1); cudaFree(d_so2);
    cudaFree(d_co1); cudaFree(d_co2); cudaFree(d_cnt1); cudaFree(d_cnt2);
}

static void bench_nms(int iters, int warmup) {
    const int K = 300;          // typical post-filter survivor count
    const int MAX_OUT = 300;
    const float IOU = 0.45f;
    std::vector<float> hboxes(K * 4);
    std::vector<float> hscores(K);
    std::vector<int> hcls(K);
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (int i = 0; i < K; ++i) {
        float cx = u01(rng) * 600.0f, cy = u01(rng) * 600.0f;
        float w = 20.0f + u01(rng) * 40.0f, h = 20.0f + u01(rng) * 40.0f;
        hboxes[i * 4 + 0] = cx - w / 2;
        hboxes[i * 4 + 1] = cy - h / 2;
        hboxes[i * 4 + 2] = cx + w / 2;
        hboxes[i * 4 + 3] = cy + h / 2;
        hscores[i] = u01(rng);
        hcls[i]    = (int)(u01(rng) * 80.0f);
    }
    float *d_b, *d_s;
    int *d_c, *d_k1, *d_k2, *d_kc1, *d_kc2;
    cudaMalloc(&d_b, sizeof(float) * K * 4);
    cudaMalloc(&d_s, sizeof(float) * K);
    cudaMalloc(&d_c, sizeof(int)   * K);
    cudaMalloc(&d_k1, sizeof(int) * MAX_OUT);
    cudaMalloc(&d_k2, sizeof(int) * MAX_OUT);
    cudaMalloc(&d_kc1, sizeof(int));
    cudaMalloc(&d_kc2, sizeof(int));
    cudaMemcpy(d_b, hboxes.data(),  sizeof(float) * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, hscores.data(), sizeof(float) * K,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, hcls.data(),    sizeof(int)   * K,     cudaMemcpyHostToDevice);

    launch_nms   (d_b, d_s, d_c, d_k1, d_kc1, K, IOU, MAX_OUT);
    launch_nms_v2(d_b, d_s, d_c, d_k2, d_kc2, K, IOU, MAX_OUT);
    cudaDeviceSynchronize();
    int kc1 = 0, kc2 = 0;
    cudaMemcpy(&kc1, d_kc1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&kc2, d_kc2, sizeof(int), cudaMemcpyDeviceToHost);
    // Outputs should be identical for greedy NMS — same input, same algorithm.
    float diff = (float)std::abs(kc1 - kc2);

    auto v1 = time_kernel([&] { launch_nms   (d_b, d_s, d_c, d_k1, d_kc1, K, IOU, MAX_OUT); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_nms_v2(d_b, d_s, d_c, d_k2, d_kc2, K, IOU, MAX_OUT); }, iters, warmup);
    char shape[32]; snprintf(shape, 32, "K=%d", K);
    record("nms (parallel block)", shape,
           compute_stats(v1), compute_stats(v2), diff, kc1 == kc2);

    cudaFree(d_b); cudaFree(d_s); cudaFree(d_c);
    cudaFree(d_k1); cudaFree(d_k2); cudaFree(d_kc1); cudaFree(d_kc2);
}

static void bench_yolov8_xywh(int iters, int warmup) {
    const int NC = 80, A = 8400;
    int total = (4 + NC) * A;
    std::vector<float> hp(total);
    std::mt19937 rng(10);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    for (auto& v : hp) v = u(rng);
    float *d_p, *d_b1, *d_b2, *d_s1, *d_s2;
    int *d_c1, *d_c2;
    cudaMalloc(&d_p,  sizeof(float) * total);
    cudaMalloc(&d_b1, sizeof(float) * A * 4);
    cudaMalloc(&d_b2, sizeof(float) * A * 4);
    cudaMalloc(&d_s1, sizeof(float) * A);
    cudaMalloc(&d_s2, sizeof(float) * A);
    cudaMalloc(&d_c1, sizeof(int)   * A);
    cudaMalloc(&d_c2, sizeof(int)   * A);
    cudaMemcpy(d_p, hp.data(), sizeof(float) * total, cudaMemcpyHostToDevice);

    launch_yolov8_decode_xywh   (d_p, NC, A, d_b1, d_s1, d_c1);
    launch_yolov8_decode_xywh_v2(d_p, NC, A, d_b2, d_s2, d_c2);
    cudaDeviceSynchronize();
    float diff_b = max_abs_diff_f(d_b1, d_b2, A * 4);

    auto v1 = time_kernel([&] { launch_yolov8_decode_xywh   (d_p, NC, A, d_b1, d_s1, d_c1); }, iters, warmup);
    auto v2 = time_kernel([&] { launch_yolov8_decode_xywh_v2(d_p, NC, A, d_b2, d_s2, d_c2); }, iters, warmup);
    record("yolov8_decode_xywh", "C=80 A=8400",
           compute_stats(v1), compute_stats(v2), diff_b, diff_b < 1e-4f);

    cudaFree(d_p); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_c1); cudaFree(d_c2);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    int iters = 200;
    int warmup = 20;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--iters")  == 0 && i + 1 < argc) iters  = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help")   == 0) {
            printf("Usage: kernel_bench [--iters N] [--warmup N]\n"); return 0;
        }
    }

    print_gpu_info();
    printf("\nKernel micro-benchmark: %d warmup + %d timed iters per kernel.\n",
           warmup, iters);
    print_header();

    // Preprocess
    bench_hwc_to_chw(iters, warmup);
    bench_bgr_to_rgb_chw(iters, warmup);

    // Forward path
    bench_silu(iters, warmup);
    bench_bn_silu(iters, warmup);
    bench_concat_channel(iters, warmup);
    bench_upsample_2x(iters, warmup);
    bench_maxpool_sppf(iters, warmup);

    // Postprocess
    bench_dfl(iters, warmup);
    bench_score_filter(iters, warmup);
    bench_nms(iters, warmup);
    bench_yolov8_xywh(iters, warmup);

    // Summary
    int mismatches = 0;
    for (const auto& r : g_results) if (!r.ok) mismatches++;
    printf("\n=== %zu kernels, %d mismatch%s ===\n",
           g_results.size(), mismatches, mismatches == 1 ? "" : "es");
    return mismatches == 0 ? 0 : 1;
}
