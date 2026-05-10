#include "pipeline.h"
#include "yolo_kernels.cuh"
#include "cuda_utils.h"

YOLOv8Pipeline::YOLOv8Pipeline(int in_w, int in_h, int num_classes, int reg_max,
                               float score_thresh, float iou_thresh, unsigned seed)
    : in_w_(in_w), in_h_(in_h),
      score_thresh_(score_thresh), iou_thresh_(iou_thresh) {

    model_ = std::make_unique<YOLOv8>(in_w_, in_h_, num_classes, reg_max, seed);
    post_  = std::make_unique<PostProcessor>(model_->total_anchors(),
                                             model_->num_classes(),
                                             model_->reg_max(), 300);

    CUDA_CHECK(cudaMalloc(&d_letterboxed_, sizeof(uint8_t) * in_w_ * in_h_ * 3));
    CUDA_CHECK(cudaMalloc(&d_input_chw_,   sizeof(float)   * in_w_ * in_h_ * 3));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

YOLOv8Pipeline::~YOLOv8Pipeline() {
    if (d_src_image_)  cudaFree(d_src_image_);
    if (d_letterboxed_) cudaFree(d_letterboxed_);
    if (d_input_chw_)   cudaFree(d_input_chw_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Detection> YOLOv8Pipeline::infer(const Image& img) {
    if (img.width <= 0 || img.height <= 0) return {};

    size_t bytes = (size_t)img.width * img.height * 3;
    if (bytes > d_src_capacity_) {
        if (d_src_image_) cudaFree(d_src_image_);
        CUDA_CHECK(cudaMalloc(&d_src_image_, bytes));
        d_src_capacity_ = bytes;
    }

    CudaTimer t_total, t_pre, t_fwd, t_post;
    t_total.start(stream_);

    // Upload source image
    CUDA_CHECK(cudaMemcpyAsync(d_src_image_, img.pixels.data(), bytes,
                               cudaMemcpyHostToDevice, stream_));

    // Preprocess: letterbox + HWC->CHW normalize
    t_pre.start(stream_);
    auto lb = compute_letterbox(img.width, img.height, in_w_, in_h_);
    launch_letterbox_uint8(d_src_image_, img.width, img.height,
                           d_letterboxed_, in_w_, in_h_,
                           lb, 114, stream_);
    launch_hwc_uint8_to_chw_float(d_letterboxed_, d_input_chw_,
                                  in_w_, in_h_, stream_);
    last_timings_.preprocess = t_pre.stop(stream_);

    // Forward
    t_fwd.start(stream_);
    model_->forward(d_input_chw_, stream_);
    last_timings_.forward = t_fwd.stop(stream_);

    // Postprocess
    t_post.start(stream_);
    auto dets = post_->run(model_->cls_flat(), model_->reg_flat(),
                           model_->anchor_xy(), model_->anchor_stride(),
                           score_thresh_, iou_thresh_,
                           lb.scale, lb.pad_x, lb.pad_y,
                           img.width, img.height, stream_);
    last_timings_.postprocess = t_post.stop(stream_);
    last_timings_.total = t_total.stop(stream_);

    return dets;
}
