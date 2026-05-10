#ifndef YOLOV8_H
#define YOLOV8_H

#include "backbone.h"
#include "neck.h"
#include "head.h"
#include <cudnn.h>
#include <memory>

// Top-level YOLOv8-like model.
// Manages backbone + neck + head and a shared cuDNN convolution workspace.
class YOLOv8 {
public:
    YOLOv8(int in_w = 640, int in_h = 640,
           int num_classes = 80, int reg_max = 16,
           unsigned seed = 42);
    ~YOLOv8();

    // Run forward on a preprocessed input tensor: d_input [1, 3, in_h, in_w] float CHW [0,1].
    void forward(const float* d_input, cudaStream_t stream = 0);

    // After forward(), these are populated:
    const float* cls_flat() const { return head_->cls_flat(); }
    const float* reg_flat() const { return head_->reg_flat(); }
    const float* anchor_xy() const { return head_->anchor_xy(); }
    const float* anchor_stride() const { return head_->anchor_stride(); }
    int total_anchors() const { return head_->total_anchors(); }
    int num_classes() const { return head_->num_classes(); }
    int reg_max() const { return head_->reg_max(); }
    int in_w() const { return in_w_; }
    int in_h() const { return in_h_; }

private:
    int in_w_, in_h_;
    cudnnHandle_t cudnn_handle_;
    void* d_workspace_ = nullptr;
    size_t workspace_bytes_ = 0;

    std::unique_ptr<Backbone> backbone_;
    std::unique_ptr<Neck> neck_;
    std::unique_ptr<DetectHead> head_;
};

#endif // YOLOV8_H
