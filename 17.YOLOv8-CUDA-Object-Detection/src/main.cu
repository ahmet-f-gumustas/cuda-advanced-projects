// YOLOv8-CUDA demo: load (or synthesize) image, run detection, draw boxes, save output.

#include "pipeline.h"
#include "image_io.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cstring>
#include <string>

static const char* USAGE =
    "Usage:\n"
    "  yolo_detect [--input <path.ppm>] [--output <path.ppm>]\n"
    "              [--score <thresh>] [--iou <thresh>] [--seed <N>]\n"
    "If no input is given, a synthetic 800x600 image is generated.\n";

int main(int argc, char** argv) {
    std::string in_path, out_path = "detections.ppm";
    float score_thresh = 0.25f, iou_thresh = 0.45f;
    unsigned seed = 42;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)        in_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)  out_path = argv[++i];
        else if (std::strcmp(argv[i], "--score") == 0 && i + 1 < argc)   score_thresh = (float)atof(argv[++i]);
        else if (std::strcmp(argv[i], "--iou") == 0 && i + 1 < argc)     iou_thresh = (float)atof(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)    seed = (unsigned)atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0)                    { printf("%s", USAGE); return 0; }
    }

    print_gpu_info();

    Image img;
    if (!in_path.empty()) {
        img = read_ppm(in_path);
        if (img.width == 0) {
            fprintf(stderr, "Failed to read %s — falling back to synthetic image.\n",
                    in_path.c_str());
            img = make_synthetic_image(800, 600, seed);
        } else {
            printf("Loaded %s (%dx%d).\n", in_path.c_str(), img.width, img.height);
        }
    } else {
        img = make_synthetic_image(800, 600, seed);
        printf("Using synthetic 800x600 image (seed=%u).\n", seed);
    }

    YOLOv8Pipeline pipe(640, 640, 80, 16, score_thresh, iou_thresh, seed);

    // Warm-up
    pipe.infer(img);

    auto dets = pipe.infer(img);
    auto t = pipe.last_timings();
    printf("\nDetections: %zu\n", dets.size());
    printf("Timing (ms): preprocess=%.2f  forward=%.2f  postprocess=%.2f  total=%.2f\n",
           t.preprocess, t.forward, t.postprocess, t.total);
    printf("Throughput: %.1f FPS\n", 1000.0f / t.total);

    int show = std::min((int)dets.size(), 10);
    for (int i = 0; i < show; ++i) {
        const auto& d = dets[i];
        printf("  [%d] cls=%d score=%.3f  box=(%.1f,%.1f)-(%.1f,%.1f)\n",
               i, d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
    }

    draw_detections(img, dets);
    if (write_ppm(out_path, img)) {
        printf("\nSaved annotated image to %s\n", out_path.c_str());
    } else {
        fprintf(stderr, "Failed to write %s\n", out_path.c_str());
    }
    return 0;
}
