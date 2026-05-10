#include "cuda_utils.h"
#include "yolo_kernels.cuh"
#include "conv2d.h"
#include "yolov8.h"
#include "postprocess.h"
#include "image_io.h"
#include "pipeline.h"

#include <cudnn.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, name)                                                       \
    do {                                                                        \
        if (cond) { printf("[PASS] %s\n", name); passed++; }                    \
        else      { printf("[FAIL] %s\n", name); failed++; }                    \
    } while (0)

static void test_silu() {
    int n = 16;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = (float)(i - 8) * 0.5f;
    float *d;
    cudaMalloc(&d, sizeof(float) * n);
    cudaMemcpy(d, h.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    launch_silu(d, n);
    cudaMemcpy(h.data(), d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaFree(d);
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float x = (float)(i - 8) * 0.5f;
        float expected = x / (1.0f + std::exp(-x));
        if (std::fabs(h[i] - expected) > 1e-4f) { ok = false; break; }
    }
    CHECK(ok, "SiLU activation");
}

static void test_concat_channel() {
    int n = 1, ca = 2, cb = 3, h = 2, w = 2;
    std::vector<float> a(n * ca * h * w);
    std::vector<float> b(n * cb * h * w);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (float)i + 1.0f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = -(float)(i + 1);
    std::vector<float> out(n * (ca + cb) * h * w);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, sizeof(float) * a.size());
    cudaMalloc(&d_b, sizeof(float) * b.size());
    cudaMalloc(&d_out, sizeof(float) * out.size());
    cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(), cudaMemcpyHostToDevice);
    launch_concat_channel(d_a, ca, d_b, cb, d_out, n, h, w);
    cudaMemcpy(out.data(), d_out, sizeof(float) * out.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    // First ca channels should match a, next cb channels should match b.
    bool ok = true;
    int hw = h * w;
    for (int c = 0; c < ca && ok; ++c)
        for (int p = 0; p < hw && ok; ++p)
            if (out[c * hw + p] != a[c * hw + p]) ok = false;
    for (int c = 0; c < cb && ok; ++c)
        for (int p = 0; p < hw && ok; ++p)
            if (out[(ca + c) * hw + p] != b[c * hw + p]) ok = false;
    CHECK(ok, "Concat channel");
}

static void test_upsample_2x() {
    int n = 1, c = 1, h = 2, w = 2;
    float src[4] = {1, 2, 3, 4};
    float expected[16] = {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4
    };
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(src));
    cudaMalloc(&d_out, sizeof(expected));
    cudaMemcpy(d_in, src, sizeof(src), cudaMemcpyHostToDevice);
    launch_upsample_nearest_2x(d_in, d_out, n, c, h, w);
    float h_out[16];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    bool ok = true;
    for (int i = 0; i < 16; ++i) if (h_out[i] != expected[i]) { ok = false; break; }
    CHECK(ok, "Upsample 2x nearest");
}

static void test_maxpool_same() {
    int n = 1, c = 1, h = 3, w = 3, k = 3;
    float src[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(src));
    cudaMalloc(&d_out, sizeof(src));
    cudaMemcpy(d_in, src, sizeof(src), cudaMemcpyHostToDevice);
    launch_maxpool2d_same(d_in, d_out, n, c, h, w, k);
    float h_out[9];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    // center pixel sees full 3x3 -> max=9
    CHECK(h_out[4] == 9.0f, "MaxPool same kernel center");
    CHECK(h_out[0] >= 5.0f, "MaxPool same kernel corner respects pad");
}

static void test_letterbox_params() {
    auto p = compute_letterbox(1920, 1080, 640, 640);
    // Expected: scale = 640/1920 = 0.3333, new_h = 360, pad_y = 140
    CHECK(std::fabs(p.scale - 640.0f / 1920.0f) < 1e-5f, "Letterbox scale");
    CHECK(p.pad_x == 0, "Letterbox pad_x for 16:9");
    CHECK(p.pad_y == (640 - 360) / 2, "Letterbox pad_y for 16:9");
}

static void test_dfl_decode() {
    int n = 1, anchors = 1, reg_max = 16;
    std::vector<float> reg(n * anchors * 4 * reg_max, 0.0f);
    // Make side 0 strongly peaked at bin 5
    reg[5] = 10.0f;
    // Side 1 peaked at bin 0
    reg[reg_max + 0] = 10.0f;
    // Side 2 peaked at bin 15
    reg[2 * reg_max + 15] = 10.0f;
    // Side 3 peaked at bin 8
    reg[3 * reg_max + 8] = 10.0f;
    float *d_reg, *d_ltrb;
    cudaMalloc(&d_reg, sizeof(float) * reg.size());
    cudaMalloc(&d_ltrb, sizeof(float) * 4);
    cudaMemcpy(d_reg, reg.data(), sizeof(float) * reg.size(), cudaMemcpyHostToDevice);
    launch_dfl_decode(d_reg, d_ltrb, n, anchors, reg_max);
    float ltrb[4];
    cudaMemcpy(ltrb, d_ltrb, sizeof(ltrb), cudaMemcpyDeviceToHost);
    cudaFree(d_reg); cudaFree(d_ltrb);
    CHECK(std::fabs(ltrb[0] - 5.0f) < 0.05f, "DFL decode peak at bin 5");
    CHECK(std::fabs(ltrb[1] - 0.0f) < 0.05f, "DFL decode peak at bin 0");
    CHECK(std::fabs(ltrb[2] - 15.0f) < 0.05f, "DFL decode peak at bin 15");
    CHECK(std::fabs(ltrb[3] - 8.0f) < 0.05f, "DFL decode peak at bin 8");
}

static void test_conv2d_smoke() {
    cudnnHandle_t h;
    cudnnCreate(&h);
    Conv2D conv(h, 3, 16, 3, 2, 1, true, 1234);
    int N = 1, H = 64, W = 64;
    int OH = conv.out_h(H);
    int OW = conv.out_w(W);
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(float) * N * 3 * H * W);
    cudaMalloc(&d_out, sizeof(float) * N * 16 * OH * OW);
    cudaMemset(d_in, 0, sizeof(float) * N * 3 * H * W);

    size_t ws = conv.workspace_bytes(N, H, W);
    void* d_ws = nullptr;
    if (ws) cudaMalloc(&d_ws, ws);
    conv.forward(d_in, d_out, N, H, W, d_ws, ws);
    cudaDeviceSynchronize();
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_in); cudaFree(d_out);
    cudnnDestroy(h);
    CHECK(OH == 32 && OW == 32, "Conv2D stride-2 output dims");
}

static void test_full_pipeline_smoke() {
    YOLOv8Pipeline pipe(640, 640, 80, 16, 0.05f, 0.45f, 7);
    auto img = make_synthetic_image(800, 600, 42);
    auto dets = pipe.infer(img);
    printf("       full pipeline produced %zu detections\n", dets.size());
    auto t = pipe.last_timings();
    printf("       timing: pre=%.2fms fwd=%.2fms post=%.2fms total=%.2fms\n",
           t.preprocess, t.forward, t.postprocess, t.total);
    CHECK(t.total > 0.0f, "Pipeline runs end-to-end");
}

int main() {
    print_gpu_info();
    printf("\nRunning YOLOv8-CUDA tests...\n\n");
    test_silu();
    test_concat_channel();
    test_upsample_2x();
    test_maxpool_same();
    test_letterbox_params();
    test_dfl_decode();
    test_conv2d_smoke();
    test_full_pipeline_smoke();
    printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
