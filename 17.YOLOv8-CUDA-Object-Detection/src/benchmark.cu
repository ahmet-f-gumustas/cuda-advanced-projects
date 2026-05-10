#include "pipeline.h"
#include "image_io.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    int iters = 100;
    int warmup = 10;
    int w = 1280, h = 720;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) w = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) h = atoi(argv[++i]);
    }

    print_gpu_info();
    printf("\nYOLOv8-CUDA benchmark — input %dx%d, %d iters (+%d warmup)\n", w, h, iters, warmup);

    auto img = make_synthetic_image(w, h, 42);
    YOLOv8Pipeline pipe(640, 640, 80, 16, 0.25f, 0.45f, 7);

    for (int i = 0; i < warmup; ++i) pipe.infer(img);

    std::vector<float> pre, fwd, post, total;
    pre.reserve(iters); fwd.reserve(iters);
    post.reserve(iters); total.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        pipe.infer(img);
        auto t = pipe.last_timings();
        pre.push_back(t.preprocess);
        fwd.push_back(t.forward);
        post.push_back(t.postprocess);
        total.push_back(t.total);
    }

    auto stats = [&](const std::vector<float>& v) {
        float sum = 0.0f, mn = 1e30f, mx = -1e30f;
        for (float x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        float mean = sum / v.size();
        std::vector<float> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        float p50 = sorted[sorted.size() / 2];
        float p99 = sorted[(size_t)(sorted.size() * 0.99)];
        printf("  mean=%6.2f  min=%6.2f  p50=%6.2f  p99=%6.2f  max=%6.2f ms\n",
               mean, mn, p50, p99, mx);
        return mean;
    };

    printf("\nPreprocess  : "); stats(pre);
    printf("Forward     : "); stats(fwd);
    printf("Postprocess : "); stats(post);
    printf("Total       : "); float mean = stats(total);
    printf("\nThroughput  : %.1f FPS (mean)\n", 1000.0f / mean);
    return 0;
}
