#include "yolov8.h"
#include "cuda_utils.h"

#include <algorithm>

YOLOv8::YOLOv8(int in_w, int in_h, int num_classes, int reg_max, unsigned seed)
    : in_w_(in_w), in_h_(in_h) {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));

    backbone_ = std::make_unique<Backbone>(cudnn_handle_, in_w_, in_h_, seed);
    neck_     = std::make_unique<Neck>(cudnn_handle_, in_w_, in_h_, seed);
    head_     = std::make_unique<DetectHead>(cudnn_handle_, in_w_, in_h_,
                                             num_classes, reg_max, seed);

    workspace_bytes_ = std::max({
        backbone_->conv_workspace_bytes(),
        neck_->conv_workspace_bytes(),
        head_->conv_workspace_bytes()
    });
    if (workspace_bytes_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_bytes_));
    }
}

YOLOv8::~YOLOv8() {
    if (d_workspace_) cudaFree(d_workspace_);
    if (cudnn_handle_) cudnnDestroy(cudnn_handle_);
}

void YOLOv8::forward(const float* d_input, cudaStream_t stream) {
    backbone_->forward(d_input, d_workspace_, workspace_bytes_, stream);
    neck_->forward(backbone_->p3(), backbone_->p4(), backbone_->p5(),
                   d_workspace_, workspace_bytes_, stream);
    head_->forward(neck_->p3_out(), neck_->p4_out(), neck_->p5_out(),
                   d_workspace_, workspace_bytes_, stream);
}
