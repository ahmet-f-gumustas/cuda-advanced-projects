#include "ts_pipeline.h"
#include "yolo_kernels.cuh"
#include "cuda_utils.h"

#include <torch/script.h>
#include <torch/cuda.h>

#include <cstring>
#include <stdexcept>
#include <vector>

TorchScriptPipeline::TorchScriptPipeline(const std::string& model_path,
                                         int in_w, int in_h, int num_classes,
                                         float score_thresh, float iou_thresh)
    : in_w_(in_w), in_h_(in_h), num_classes_(num_classes),
      score_thresh_(score_thresh), iou_thresh_(iou_thresh) {

    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA not available to LibTorch");
    }

    module_ = std::make_unique<torch::jit::Module>(torch::jit::load(model_path));
    module_->to(torch::kCUDA);
    module_->eval();

    // Probe the model once to discover anchor count (e.g., 8400 for 640x640).
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto probe_in = torch::zeros({1, 3, in_h_, in_w_}, opts);
    torch::NoGradGuard ng;
    auto probe_out = module_->forward({probe_in}).toTensor();
    // Expect shape [1, 4 + num_classes, num_anchors]
    if (probe_out.dim() != 3 || probe_out.size(0) != 1) {
        throw std::runtime_error("Unexpected model output rank");
    }
    int out_c = (int)probe_out.size(1);
    num_anchors_ = (int)probe_out.size(2);
    if (out_c != 4 + num_classes_) {
        // Fall back: use whatever the model declares.
        num_classes_ = out_c - 4;
    }

    CUDA_CHECK(cudaMalloc(&d_letterboxed_, sizeof(uint8_t) * in_w_ * in_h_ * 3));
    CUDA_CHECK(cudaMalloc(&d_input_chw_,   sizeof(float)   * in_w_ * in_h_ * 3));
    CUDA_CHECK(cudaMalloc(&d_boxes_all_,   sizeof(float) * num_anchors_ * 4));
    CUDA_CHECK(cudaMalloc(&d_scores_all_,  sizeof(float) * num_anchors_));
    CUDA_CHECK(cudaMalloc(&d_class_all_,   sizeof(int) * num_anchors_));

    post_ = std::make_unique<PostProcessor>(num_anchors_, num_classes_,
                                            /*reg_max=*/16, /*max_dets=*/300);
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

TorchScriptPipeline::~TorchScriptPipeline() {
    if (d_src_image_)   cudaFree(d_src_image_);
    if (d_letterboxed_) cudaFree(d_letterboxed_);
    if (d_input_chw_)   cudaFree(d_input_chw_);
    if (d_boxes_all_)   cudaFree(d_boxes_all_);
    if (d_scores_all_)  cudaFree(d_scores_all_);
    if (d_class_all_)   cudaFree(d_class_all_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Detection> TorchScriptPipeline::infer(const Image& img) {
    return infer_raw(img.pixels.data(), img.width, img.height, /*is_bgr=*/false);
}

std::vector<Detection> TorchScriptPipeline::infer_raw(const uint8_t* src, int w, int h,
                                                      bool is_bgr) {
    if (!src || w <= 0 || h <= 0) return {};

    size_t bytes = (size_t)w * h * 3;
    if (bytes > d_src_capacity_) {
        if (d_src_image_) cudaFree(d_src_image_);
        CUDA_CHECK(cudaMalloc(&d_src_image_, bytes));
        d_src_capacity_ = bytes;
    }

    CudaTimer t_total, t_up, t_pre, t_fwd, t_post;
    t_total.start(stream_);

    // Upload to GPU.
    t_up.start(stream_);
    if (is_bgr) {
        // Swap BGR→RGB on host first (cheap for camera frames; could be a kernel).
        std::vector<uint8_t> rgb(bytes);
        for (size_t i = 0; i < (size_t)w * h; ++i) {
            rgb[i * 3 + 0] = src[i * 3 + 2];
            rgb[i * 3 + 1] = src[i * 3 + 1];
            rgb[i * 3 + 2] = src[i * 3 + 0];
        }
        CUDA_CHECK(cudaMemcpyAsync(d_src_image_, rgb.data(), bytes,
                                   cudaMemcpyHostToDevice, stream_));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_src_image_, src, bytes,
                                   cudaMemcpyHostToDevice, stream_));
    }
    last_timings_.upload = t_up.stop(stream_);

    // Preprocess on GPU: letterbox + HWC u8 → CHW f32.
    t_pre.start(stream_);
    auto lb = compute_letterbox(w, h, in_w_, in_h_);
    launch_letterbox_uint8(d_src_image_, w, h, d_letterboxed_, in_w_, in_h_,
                           lb, 114, stream_);
    launch_hwc_uint8_to_chw_float(d_letterboxed_, d_input_chw_, in_w_, in_h_, stream_);
    last_timings_.preprocess = t_pre.stop(stream_);

    // Build a torch tensor view of d_input_chw_ (no copy).
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto input = torch::from_blob(d_input_chw_, {1, 3, in_h_, in_w_}, opts);

    // Forward.
    t_fwd.start(stream_);
    torch::NoGradGuard ng;
    auto out = module_->forward({input}).toTensor();
    CUDA_CHECK(cudaStreamSynchronize(stream_));   // sync default stream with our stream
    last_timings_.forward = t_fwd.stop(stream_);

    // Decode raw output [1, 4+C, A] -> per-anchor xyxy + score + class on our stream.
    t_post.start(stream_);
    const float* d_pred = out.data_ptr<float>();
    launch_yolov8_decode_xywh(d_pred, num_classes_, num_anchors_,
                              d_boxes_all_, d_scores_all_, d_class_all_, stream_);

    auto dets = post_->run_decoded(d_boxes_all_, d_scores_all_, d_class_all_,
                                   num_anchors_, score_thresh_, iou_thresh_,
                                   lb.scale, lb.pad_x, lb.pad_y, w, h, stream_);
    last_timings_.postprocess = t_post.stop(stream_);
    last_timings_.total = t_total.stop(stream_);
    return dets;
}
