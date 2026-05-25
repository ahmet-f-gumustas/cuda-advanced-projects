#ifndef YOLO_DISPATCH_H
#define YOLO_DISPATCH_H

// Runtime dispatcher between v1 and v2 kernels.
// Default: v1. Call `set_use_v2_kernels(true)` to switch the whole pipeline
// to optimized variants. The flag is global so it propagates through Conv2D,
// Backbone, Neck, Head, PostProcessor, and pipelines without threading a
// parameter through every constructor.

#include "yolo_kernels.cuh"

bool use_v2_kernels();
void set_use_v2_kernels(bool on);

// Dispatchers. Each routes to v1 or v2 based on the global flag.

inline void dispatch_silu(float* d_data, int n, cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_silu_v2(d_data, n, stream);
    else                  launch_silu(d_data, n, stream);
}

inline void dispatch_hwc_uint8_to_chw_float(const uint8_t* d_src, float* d_dst,
                                            int w, int h, cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_hwc_uint8_to_chw_float_v2(d_src, d_dst, w, h, stream);
    else                  launch_hwc_uint8_to_chw_float(d_src, d_dst, w, h, stream);
}

inline void dispatch_concat_channel(const float* d_a, int ca,
                                    const float* d_b, int cb,
                                    float* d_out, int n, int h, int w,
                                    cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_concat_channel_v2(d_a, ca, d_b, cb, d_out, n, h, w, stream);
    else                  launch_concat_channel(d_a, ca, d_b, cb, d_out, n, h, w, stream);
}

inline void dispatch_upsample_nearest_2x(const float* d_in, float* d_out,
                                         int n, int c, int h, int w,
                                         cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_upsample_nearest_2x_v2(d_in, d_out, n, c, h, w, stream);
    else                  launch_upsample_nearest_2x(d_in, d_out, n, c, h, w, stream);
}

inline void dispatch_maxpool2d_same(const float* d_in, float* d_out,
                                    int n, int c, int h, int w, int k,
                                    cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_maxpool2d_same_v2(d_in, d_out, n, c, h, w, k, stream);
    else                  launch_maxpool2d_same(d_in, d_out, n, c, h, w, k, stream);
}

inline void dispatch_dfl_decode(const float* d_reg, float* d_ltrb,
                                int n, int anchors, int reg_max,
                                cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_dfl_decode_v2(d_reg, d_ltrb, n, anchors, reg_max, stream);
    else                  launch_dfl_decode(d_reg, d_ltrb, n, anchors, reg_max, stream);
}

inline void dispatch_score_filter(const float* d_boxes_in, const float* d_scores_in,
                                  const int* d_class_in,
                                  float* d_boxes_out, float* d_scores_out, int* d_class_out,
                                  int* d_count,
                                  int n_in, float score_thresh, int max_out,
                                  cudaStream_t stream = 0) {
    if (use_v2_kernels()) {
        launch_score_filter_v2(d_boxes_in, d_scores_in, d_class_in,
                               d_boxes_out, d_scores_out, d_class_out,
                               d_count, n_in, score_thresh, max_out, stream);
    } else {
        launch_score_filter(d_boxes_in, d_scores_in, d_class_in,
                            d_boxes_out, d_scores_out, d_class_out,
                            d_count, n_in, score_thresh, max_out, stream);
    }
}

inline void dispatch_nms(const float* d_boxes, const float* d_scores, const int* d_class_id,
                         int* d_keep, int* d_keep_count,
                         int k, float iou_thresh, int max_out,
                         cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_nms_v2(d_boxes, d_scores, d_class_id,
                                        d_keep, d_keep_count, k, iou_thresh, max_out, stream);
    else                  launch_nms(d_boxes, d_scores, d_class_id,
                                     d_keep, d_keep_count, k, iou_thresh, max_out, stream);
}

inline void dispatch_yolov8_decode_xywh(const float* d_pred, int num_classes, int num_anchors,
                                        float* d_boxes, float* d_scores, int* d_class_id,
                                        cudaStream_t stream = 0) {
    if (use_v2_kernels()) launch_yolov8_decode_xywh_v2(d_pred, num_classes, num_anchors,
                                                       d_boxes, d_scores, d_class_id, stream);
    else                  launch_yolov8_decode_xywh(d_pred, num_classes, num_anchors,
                                                    d_boxes, d_scores, d_class_id, stream);
}

#endif // YOLO_DISPATCH_H
