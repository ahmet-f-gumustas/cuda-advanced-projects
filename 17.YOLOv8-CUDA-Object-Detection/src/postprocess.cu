#include "postprocess.h"
#include "yolo_kernels.cuh"
#include "cuda_utils.h"

#include <algorithm>
#include <vector>

PostProcessor::PostProcessor(int total_anchors, int num_classes, int reg_max, int max_dets)
    : total_anchors_(total_anchors), num_classes_(num_classes),
      reg_max_(reg_max), max_dets_(max_dets) {

    CUDA_CHECK(cudaMalloc(&d_ltrb_,         sizeof(float) * total_anchors_ * 4));
    CUDA_CHECK(cudaMalloc(&d_boxes_all_,    sizeof(float) * total_anchors_ * 4));
    CUDA_CHECK(cudaMalloc(&d_scores_all_,   sizeof(float) * total_anchors_));
    CUDA_CHECK(cudaMalloc(&d_class_all_,    sizeof(int) * total_anchors_));
    CUDA_CHECK(cudaMalloc(&d_boxes_kept_,   sizeof(float) * max_dets_ * 4));
    CUDA_CHECK(cudaMalloc(&d_scores_kept_,  sizeof(float) * max_dets_));
    CUDA_CHECK(cudaMalloc(&d_class_kept_,   sizeof(int) * max_dets_));
    CUDA_CHECK(cudaMalloc(&d_count_,        sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keep_idx_,     sizeof(int) * max_dets_));
    CUDA_CHECK(cudaMalloc(&d_keep_count_,   sizeof(int)));
}

PostProcessor::~PostProcessor() {
    auto del = [](void* p) { if (p) cudaFree(p); };
    del(d_ltrb_); del(d_boxes_all_); del(d_scores_all_); del(d_class_all_);
    del(d_boxes_kept_); del(d_scores_kept_); del(d_class_kept_);
    del(d_count_); del(d_keep_idx_); del(d_keep_count_);
}

std::vector<Detection> PostProcessor::run(
    const float* d_cls_flat, const float* d_reg_flat,
    const float* d_anchor_xy, const float* d_anchor_stride,
    float score_thresh, float iou_thresh,
    float letterbox_scale, int pad_x, int pad_y,
    int orig_w, int orig_h,
    cudaStream_t stream) {

    // 1) DFL decode -> per-anchor (l, t, r, b) in feature-grid units
    launch_dfl_decode(d_reg_flat, d_ltrb_, 1, total_anchors_, reg_max_, stream);

    // 2) Anchor-free decode -> xyxy in net-input pixel coords + scores + class
    launch_decode_predictions(d_ltrb_, d_cls_flat, d_anchor_xy, d_anchor_stride,
                              d_boxes_all_, d_scores_all_, d_class_all_,
                              1, total_anchors_, num_classes_, stream);

    // 3) Score-threshold filter -> compact arrays
    launch_score_filter(d_boxes_all_, d_scores_all_, d_class_all_,
                        d_boxes_kept_, d_scores_kept_, d_class_kept_,
                        d_count_, total_anchors_, score_thresh, max_dets_,
                        stream);

    int h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int kept = std::min(h_count, max_dets_);

    if (kept == 0) return {};

    // 4) NMS on the filtered set
    launch_nms(d_boxes_kept_, d_scores_kept_, d_class_kept_,
               d_keep_idx_, d_keep_count_,
               kept, iou_thresh, max_dets_, stream);

    int h_keep = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_keep, d_keep_count_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_keep == 0) return {};

    // 5) Gather kept indices into final tight arrays, undo letterbox, copy to host.
    // For small h_keep we do this on the host: copy keep indices and the kept arrays.
    std::vector<int>   h_keep_idx(h_keep);
    std::vector<float> h_boxes(kept * 4);
    std::vector<float> h_scores(kept);
    std::vector<int>   h_class(kept);
    CUDA_CHECK(cudaMemcpyAsync(h_keep_idx.data(), d_keep_idx_, sizeof(int) * h_keep,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_boxes.data(),  d_boxes_kept_, sizeof(float) * kept * 4,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_scores.data(), d_scores_kept_, sizeof(float) * kept,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_class.data(),  d_class_kept_,  sizeof(int) * kept,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pack final final boxes (after letterbox inverse) on host.
    float inv_scale = 1.0f / letterbox_scale;
    std::vector<Detection> out;
    out.reserve(h_keep);
    for (int i = 0; i < h_keep; ++i) {
        int idx = h_keep_idx[i];
        float x1 = (h_boxes[idx * 4 + 0] - pad_x) * inv_scale;
        float y1 = (h_boxes[idx * 4 + 1] - pad_y) * inv_scale;
        float x2 = (h_boxes[idx * 4 + 2] - pad_x) * inv_scale;
        float y2 = (h_boxes[idx * 4 + 3] - pad_y) * inv_scale;
        x1 = std::max(0.0f, std::min(x1, (float)(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(orig_h - 1)));
        out.push_back({x1, y1, x2, y2, h_scores[idx], h_class[idx]});
    }
    return out;
}

std::vector<Detection> PostProcessor::run_decoded(
    const float* d_boxes_in, const float* d_scores_in, const int* d_class_in,
    int n_in, float score_thresh, float iou_thresh,
    float letterbox_scale, int pad_x, int pad_y,
    int orig_w, int orig_h, cudaStream_t stream) {

    // Score-threshold filter into the kept arrays.
    launch_score_filter(d_boxes_in, d_scores_in, d_class_in,
                        d_boxes_kept_, d_scores_kept_, d_class_kept_,
                        d_count_, n_in, score_thresh, max_dets_, stream);

    int h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int kept = std::min(h_count, max_dets_);
    if (kept == 0) return {};

    launch_nms(d_boxes_kept_, d_scores_kept_, d_class_kept_,
               d_keep_idx_, d_keep_count_, kept, iou_thresh, max_dets_, stream);

    int h_keep = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_keep, d_keep_count_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_keep == 0) return {};

    std::vector<int>   h_keep_idx(h_keep);
    std::vector<float> h_boxes(kept * 4);
    std::vector<float> h_scores(kept);
    std::vector<int>   h_class(kept);
    CUDA_CHECK(cudaMemcpyAsync(h_keep_idx.data(), d_keep_idx_, sizeof(int) * h_keep,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_boxes.data(),  d_boxes_kept_, sizeof(float) * kept * 4,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_scores.data(), d_scores_kept_, sizeof(float) * kept,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_class.data(),  d_class_kept_,  sizeof(int) * kept,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float inv_scale = 1.0f / letterbox_scale;
    std::vector<Detection> out;
    out.reserve(h_keep);
    for (int i = 0; i < h_keep; ++i) {
        int idx = h_keep_idx[i];
        float x1 = (h_boxes[idx * 4 + 0] - pad_x) * inv_scale;
        float y1 = (h_boxes[idx * 4 + 1] - pad_y) * inv_scale;
        float x2 = (h_boxes[idx * 4 + 2] - pad_x) * inv_scale;
        float y2 = (h_boxes[idx * 4 + 3] - pad_y) * inv_scale;
        x1 = std::max(0.0f, std::min(x1, (float)(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(orig_h - 1)));
        out.push_back({x1, y1, x2, y2, h_scores[idx], h_class[idx]});
    }
    return out;
}
