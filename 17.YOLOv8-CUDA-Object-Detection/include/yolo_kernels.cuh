#ifndef YOLO_KERNELS_CUH
#define YOLO_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================
// Preprocessing
// ============================================================

// Letterbox resize: src [src_h, src_w, 3] uint8 HWC -> dst [dst_h, dst_w, 3] uint8 HWC,
// preserves aspect ratio with gray (114) padding. Returns scale + pad offsets via host params.
struct LetterboxParams {
    float scale;
    int pad_x;
    int pad_y;
};

LetterboxParams compute_letterbox(int src_w, int src_h, int dst_w, int dst_h);

void launch_letterbox_uint8(const uint8_t* d_src, int src_w, int src_h,
                            uint8_t* d_dst, int dst_w, int dst_h,
                            const LetterboxParams& p, uint8_t pad_value,
                            cudaStream_t stream = 0);

// Convert uint8 HWC [H, W, 3] -> float32 CHW [3, H, W] normalized to [0, 1].
void launch_hwc_uint8_to_chw_float(const uint8_t* d_src, float* d_dst,
                                   int w, int h, cudaStream_t stream = 0);

// ============================================================
// Activations / fused normalization
// ============================================================

// Element-wise SiLU: x * sigmoid(x)
void launch_silu(float* d_data, int n, cudaStream_t stream = 0);

// Fused BatchNorm + SiLU. Layout: NCHW with channels = c.
// Per-channel: y = silu( (x - mean) * rstd * gamma + beta )
// where rstd = 1 / sqrt(var + eps).
void launch_bn_silu(float* d_data, const float* d_mean, const float* d_rstd,
                    const float* d_gamma, const float* d_beta,
                    int n, int c, int h, int w, cudaStream_t stream = 0);

// ============================================================
// Tensor reshape / topology
// ============================================================

// Concatenate two NCHW tensors along channel dim.
// a [N, ca, H, W] + b [N, cb, H, W] -> out [N, ca+cb, H, W]
void launch_concat_channel(const float* d_a, int ca,
                           const float* d_b, int cb,
                           float* d_out, int n, int h, int w,
                           cudaStream_t stream = 0);

// Slice channels [c0, c1) from NCHW tensor.
void launch_slice_channel(const float* d_in, int c_total,
                          float* d_out, int c0, int c1,
                          int n, int h, int w, cudaStream_t stream = 0);

// 2x nearest-neighbor upsample NCHW [N,C,H,W] -> [N,C,2H,2W]
void launch_upsample_nearest_2x(const float* d_in, float* d_out,
                                int n, int c, int h, int w,
                                cudaStream_t stream = 0);

// Max pool 2D with kernel k, stride 1, padding k/2 (preserves spatial dims).
// Used inside SPPF block.
void launch_maxpool2d_same(const float* d_in, float* d_out,
                           int n, int c, int h, int w, int k,
                           cudaStream_t stream = 0);

// ============================================================
// Postprocess
// ============================================================

// Distribution Focal Loss decode: per anchor, per side (l,t,r,b) we have a 16-bin
// softmax distribution and we compute the expected value.
// d_reg: [N, anchors, 4, 16]  -> d_ltrb: [N, anchors, 4]
void launch_dfl_decode(const float* d_reg, float* d_ltrb,
                       int n, int anchors, int reg_max,
                       cudaStream_t stream = 0);

// Build a per-level anchor grid (center coords in feature-map units).
// out: [num_anchors, 2] (cx, cy) and per-anchor stride array.
void build_anchor_grid(int feat_h, int feat_w, int stride,
                       float* out_xy_cpu, float* out_stride_cpu, int offset);

// Anchor-free decode: take per-anchor ltrb + class logits and produce
//   d_boxes: [N, anchors, 4]  (xyxy in input-image pixel coords)
//   d_scores: [N, anchors]    (max class probability * objectness=1)
//   d_class_id: [N, anchors]  (argmax class index)
// d_cls: [N, anchors, num_classes] raw logits (sigmoid will be applied per-class)
// d_anchor_xy: [anchors, 2] in feature-grid coords; d_anchor_stride: [anchors] strides.
void launch_decode_predictions(const float* d_ltrb, const float* d_cls,
                               const float* d_anchor_xy, const float* d_anchor_stride,
                               float* d_boxes, float* d_scores, int* d_class_id,
                               int n, int anchors, int num_classes,
                               cudaStream_t stream = 0);

// Class-aware NMS. Inputs (after score threshold filtering).
// boxes [K, 4] xyxy, scores [K], class_id [K]; sort by score; suppress IoU > iou_thresh
// within same class. Returns number of kept indices via host_keep_count.
// d_keep_indices preallocated [K] int.
void launch_nms(const float* d_boxes, const float* d_scores, const int* d_class_id,
                int* d_keep, int* d_keep_count,
                int k, float iou_thresh, int max_out,
                cudaStream_t stream = 0);

// Filter detections by score threshold; outputs compact arrays.
// Returns count via host_count_out.
void launch_score_filter(const float* d_boxes_in, const float* d_scores_in,
                         const int* d_class_in,
                         float* d_boxes_out, float* d_scores_out, int* d_class_out,
                         int* d_count,
                         int n_in, float score_thresh, int max_out,
                         cudaStream_t stream = 0);

// Decode ultralytics-style YOLOv8 raw output [1, 4+num_classes, num_anchors]
// into per-anchor xyxy boxes, max-class score, and class id.
// Layout matches torch tensor with shape (1, 84, 8400) — channel-major, anchor-stride.
// Class scores are assumed to already be sigmoid'd (ultralytics convention).
void launch_yolov8_decode_xywh(const float* d_pred, int num_classes, int num_anchors,
                               float* d_boxes, float* d_scores, int* d_class_id,
                               cudaStream_t stream = 0);

// Undo letterbox transform on output boxes (in-place).
// Input boxes are in network-input pixel coords (e.g. 640x640).
// Maps back to original image coords using inverse of LetterboxParams.
void launch_unletterbox_boxes(float* d_boxes, int n,
                              float scale, int pad_x, int pad_y,
                              int orig_w, int orig_h,
                              cudaStream_t stream = 0);

#endif // YOLO_KERNELS_CUH
