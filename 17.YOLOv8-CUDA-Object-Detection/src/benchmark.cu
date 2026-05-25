#include "pipeline.h"
#include "yolo_dispatch.h"
#include "image_io.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

struct Stats { float mean, p50, p99; };

static Stats compute_stats(std::vector<float> v) {
    Stats s{0.0f, 0.0f, 0.0f};
    if (v.empty()) return s;
    double sum = 0.0;
    for (float x : v) sum += x;
    s.mean = (float)(sum / v.size());
    std::sort(v.begin(), v.end());
    s.p50 = v[v.size() / 2];
    s.p99 = v[(size_t)(v.size() * 0.99)];
    return s;
}

struct Run {
    Stats pre, fwd, post, total;
};

static Run run_pipeline(YOLOv8Pipeline& pipe, const Image& img,
                        int iters, int warmup) {
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
    return { compute_stats(pre), compute_stats(fwd),
             compute_stats(post), compute_stats(total) };
}

static void print_row(const char* label, const Stats& s) {
    printf("  %-12s mean=%6.3f  p50=%6.3f  p99=%6.3f ms\n",
           label, s.mean, s.p50, s.p99);
}

int main(int argc, char** argv) {
    int iters = 100;
    int warmup = 10;
    int w = 1280, h = 720;
    bool compare = false;
    bool v2_only = false;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--iters")   == 0 && i + 1 < argc) iters  = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup")  == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--width")   == 0 && i + 1 < argc) w = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height")  == 0 && i + 1 < argc) h = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--compare") == 0) compare = true;
        else if (std::strcmp(argv[i], "--v2")      == 0) v2_only = true;
        else if (std::strcmp(argv[i], "--help")    == 0) {
            printf("Usage: yolo_benchmark [--iters N] [--warmup N] [--width W] [--height H]\n"
                   "                      [--compare]  run v1 and v2 side by side\n"
                   "                      [--v2]       run v2 path only\n");
            return 0;
        }
    }

    print_gpu_info();
    printf("\nYOLOv8-CUDA benchmark — input %dx%d, %d iters (+%d warmup)\n",
           w, h, iters, warmup);

    auto img = make_synthetic_image(w, h, 42);

    auto run_one = [&](bool v2) {
        set_use_v2_kernels(v2);
        YOLOv8Pipeline pipe(640, 640, 80, 16, 0.25f, 0.45f, 7);
        auto r = run_pipeline(pipe, img, iters, warmup);
        printf("\n[%s]\n", v2 ? "v2 (optimized kernels)" : "v1 (baseline)");
        print_row("Preprocess",  r.pre);
        print_row("Forward",     r.fwd);
        print_row("Postprocess", r.post);
        print_row("Total",       r.total);
        printf("  Throughput  %.1f FPS (mean)\n", 1000.0f / r.total.mean);
        return r;
    };

    if (compare) {
        Run v1 = run_one(false);
        Run v2 = run_one(true);
        printf("\n[speedup v1 / v2]\n");
        printf("  Preprocess   %.2fx\n", v1.pre.mean   / v2.pre.mean);
        printf("  Forward      %.2fx\n", v1.fwd.mean   / v2.fwd.mean);
        printf("  Postprocess  %.2fx\n", v1.post.mean  / v2.post.mean);
        printf("  Total        %.2fx\n", v1.total.mean / v2.total.mean);
    } else {
        run_one(v2_only);
    }

    return 0;
}
