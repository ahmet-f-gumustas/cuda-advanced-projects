#ifndef PIPELINE_H
#define PIPELINE_H

#include "yolov8.h"
#include "postprocess.h"
#include "image_io.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

// End-to-end YOLOv8 inference: image -> letterbox -> normalize -> forward -> postprocess.
class YOLOv8Pipeline {
public:
    YOLOv8Pipeline(int in_w = 640, int in_h = 640,
                   int num_classes = 80, int reg_max = 16,
                   float score_thresh = 0.25f, float iou_thresh = 0.45f,
                   unsigned seed = 42);
    ~YOLOv8Pipeline();

    // Run on a host image. Returns detections in original-image coordinates.
    std::vector<Detection> infer(const Image& img);

    // Per-stage timing from last infer() call (milliseconds).
    struct Timings {
        float preprocess = 0.0f;
        float forward = 0.0f;
        float postprocess = 0.0f;
        float total = 0.0f;
    };
    Timings last_timings() const { return last_timings_; }

    int in_w() const { return in_w_; }
    int in_h() const { return in_h_; }

private:
    int in_w_, in_h_;
    float score_thresh_;
    float iou_thresh_;

    std::unique_ptr<YOLOv8> model_;
    std::unique_ptr<PostProcessor> post_;

    // Device buffers for preprocessing
    uint8_t* d_src_image_ = nullptr;
    size_t d_src_capacity_ = 0;
    uint8_t* d_letterboxed_ = nullptr;
    float*   d_input_chw_   = nullptr;

    cudaStream_t stream_ = nullptr;

    Timings last_timings_;
};

#endif // PIPELINE_H
