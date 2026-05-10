#ifndef HEAD_H
#define HEAD_H

#include "conv2d.h"
#include <cudnn.h>
#include <memory>

// Decoupled detection head with DFL regression (YOLOv8 style).
// Three branches at strides 8/16/32. For each level:
//   cls_branch: 2x ConvBNSiLU(c,c,3) + Conv1x1(c -> num_classes)
//   reg_branch: 2x ConvBNSiLU(c,c,3) + Conv1x1(c -> 4 * reg_max)
//
// Outputs are concatenated across levels into flat anchor-major layout:
//   d_cls_flat  : [B, total_anchors, num_classes]
//   d_reg_flat  : [B, total_anchors, 4 * reg_max]
class DetectHead {
public:
    DetectHead(cudnnHandle_t handle, int in_w, int in_h,
               int num_classes, int reg_max, unsigned seed);
    ~DetectHead();

    size_t conv_workspace_bytes() const { return conv_ws_bytes_; }

    // Forward: takes the three neck outputs.
    void forward(const float* d_p3, const float* d_p4, const float* d_p5,
                 void* ws, size_t ws_bytes, cudaStream_t stream = 0);

    int total_anchors() const { return total_anchors_; }
    int num_classes() const { return num_classes_; }
    int reg_max() const { return reg_max_; }

    const float* cls_flat() const { return d_cls_flat_; }
    const float* reg_flat() const { return d_reg_flat_; }

    // Anchor grid (xy in feature-grid units) and per-anchor strides, device-side.
    const float* anchor_xy() const { return d_anchor_xy_; }
    const float* anchor_stride() const { return d_anchor_stride_; }

private:
    cudnnHandle_t handle_;
    int in_w_, in_h_;
    int num_classes_;
    int reg_max_;
    int total_anchors_;
    size_t conv_ws_bytes_ = 0;

    // Per-level layer triplets — cls branch
    std::unique_ptr<Conv2D> p3_cls_a_, p3_cls_b_, p3_cls_pred_;
    std::unique_ptr<Conv2D> p4_cls_a_, p4_cls_b_, p4_cls_pred_;
    std::unique_ptr<Conv2D> p5_cls_a_, p5_cls_b_, p5_cls_pred_;
    // reg branch
    std::unique_ptr<Conv2D> p3_reg_a_, p3_reg_b_, p3_reg_pred_;
    std::unique_ptr<Conv2D> p4_reg_a_, p4_reg_b_, p4_reg_pred_;
    std::unique_ptr<Conv2D> p5_reg_a_, p5_reg_b_, p5_reg_pred_;

    // Intermediate buffers (per branch, per level: A,B,out)
    float *d_p3_cls_a_ = nullptr, *d_p3_cls_b_ = nullptr, *d_p3_cls_out_ = nullptr;
    float *d_p3_reg_a_ = nullptr, *d_p3_reg_b_ = nullptr, *d_p3_reg_out_ = nullptr;
    float *d_p4_cls_a_ = nullptr, *d_p4_cls_b_ = nullptr, *d_p4_cls_out_ = nullptr;
    float *d_p4_reg_a_ = nullptr, *d_p4_reg_b_ = nullptr, *d_p4_reg_out_ = nullptr;
    float *d_p5_cls_a_ = nullptr, *d_p5_cls_b_ = nullptr, *d_p5_cls_out_ = nullptr;
    float *d_p5_reg_a_ = nullptr, *d_p5_reg_b_ = nullptr, *d_p5_reg_out_ = nullptr;

    // Flat anchor-major outputs
    float *d_cls_flat_ = nullptr;
    float *d_reg_flat_ = nullptr;

    // Anchor grid (device)
    float *d_anchor_xy_ = nullptr;
    float *d_anchor_stride_ = nullptr;
};

#endif // HEAD_H
