#ifndef TS_PIPELINE_H
#define TS_PIPELINE_H

#include "postprocess.h"
#include "image_io.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

// Forward declare torch::jit::Module to avoid pulling LibTorch into every TU.
namespace torch { namespace jit { struct Module; } }

// End-to-end inference using a TorchScript-exported ultralytics YOLOv8 model.
//
// Architecture:
//   uint8 HWC frame  (host or device)
//     ↓ upload + letterbox + HWC→CHW normalize   [custom CUDA kernels]
//   float CHW [1, 3, 640, 640]
//     ↓ torch::jit::Module::forward                [LibTorch + CUDA]
//   raw [1, 84, 8400]
//     ↓ decode_xywh → score_filter → NMS → unletterbox  [custom CUDA kernels]
//   detections in original-image coords
class TorchScriptPipeline {
public:
    TorchScriptPipeline(const std::string& model_path,
                        int in_w = 640, int in_h = 640,
                        int num_classes = 80,
                        float score_thresh = 0.25f,
                        float iou_thresh   = 0.45f);
    ~TorchScriptPipeline();

    // Reuses internal device buffers across calls. Frame may be any HxW.
    std::vector<Detection> infer(const Image& img);

    // Convenience: accept raw bytes (HWC RGB uint8) without constructing an Image.
    std::vector<Detection> infer_raw(const uint8_t* bgr_or_rgb_host, int w, int h, bool is_bgr);

    struct Timings {
        float upload      = 0.0f;
        float preprocess  = 0.0f;
        float forward     = 0.0f;
        float postprocess = 0.0f;
        float total       = 0.0f;
    };
    Timings last_timings() const { return last_timings_; }

    int in_w() const { return in_w_; }
    int in_h() const { return in_h_; }

private:
    int in_w_, in_h_;
    int num_classes_;
    int num_anchors_ = 0;
    float score_thresh_;
    float iou_thresh_;

    // LibTorch model (heap-allocated; PIMPL keeps torch headers out of this .h)
    std::unique_ptr<torch::jit::Module> module_;

    // Device buffers
    uint8_t* d_src_image_ = nullptr;
    size_t d_src_capacity_ = 0;
    uint8_t* d_letterboxed_ = nullptr;
    float*   d_input_chw_   = nullptr;

    // Postprocess scratch (re-using PostProcessor)
    std::unique_ptr<PostProcessor> post_;

    // Scratch for raw decode (decode_xywh outputs)
    float* d_boxes_all_ = nullptr;
    float* d_scores_all_ = nullptr;
    int*   d_class_all_ = nullptr;

    cudaStream_t stream_ = nullptr;
    Timings last_timings_;
};

#endif // TS_PIPELINE_H
