#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cuda_runtime.h>
#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

// PostProcessor wraps the GPU postprocess pipeline:
//   DFL decode -> anchor-free decode -> score filter -> NMS -> unletterbox -> host copy
//
// Owns scratch buffers sized for `total_anchors`.
class PostProcessor {
public:
    PostProcessor(int total_anchors, int num_classes, int reg_max, int max_dets = 300);
    ~PostProcessor();

    // Runs full postprocess.
    //   d_cls_flat: [total_anchors, num_classes]   logits
    //   d_reg_flat: [total_anchors, 4 * reg_max]   logits
    //   d_anchor_xy / d_anchor_stride: from DetectHead
    //
    // Returns host-side detection list in original-image coords.
    std::vector<Detection> run(const float* d_cls_flat, const float* d_reg_flat,
                               const float* d_anchor_xy, const float* d_anchor_stride,
                               float score_thresh, float iou_thresh,
                               float letterbox_scale, int pad_x, int pad_y,
                               int orig_w, int orig_h,
                               cudaStream_t stream = 0);

private:
    int total_anchors_;
    int num_classes_;
    int reg_max_;
    int max_dets_;

    float *d_ltrb_ = nullptr;            // [A, 4]
    float *d_boxes_all_ = nullptr;       // [A, 4]
    float *d_scores_all_ = nullptr;      // [A]
    int   *d_class_all_ = nullptr;       // [A]
    float *d_boxes_kept_ = nullptr;      // [max_dets, 4]
    float *d_scores_kept_ = nullptr;     // [max_dets]
    int   *d_class_kept_ = nullptr;      // [max_dets]
    int   *d_count_ = nullptr;           // [1]
    int   *d_keep_idx_ = nullptr;        // [max_dets]
    int   *d_keep_count_ = nullptr;      // [1]
};

#endif // POSTPROCESS_H
