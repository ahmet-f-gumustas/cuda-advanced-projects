#ifndef NECK_H
#define NECK_H

#include "conv2d.h"
#include <cudnn.h>
#include <memory>

// PAN-FPN style neck for YOLOv8.
// Inputs:  P3 [B, 64, 80, 80], P4 [B, 128, 40, 40], P5 [B, 256, 20, 20]
// Outputs: same shapes (P3_out, P4_out, P5_out)
class Neck {
public:
    Neck(cudnnHandle_t handle, int in_w, int in_h, unsigned seed);
    ~Neck();

    size_t conv_workspace_bytes() const { return conv_ws_bytes_; }

    void forward(const float* d_p3, const float* d_p4, const float* d_p5,
                 void* ws, size_t ws_bytes, cudaStream_t stream = 0);

    const float* p3_out() const { return d_p3_out_; }
    const float* p4_out() const { return d_p4_out_; }
    const float* p5_out() const { return d_p5_out_; }

private:
    cudnnHandle_t handle_;
    int in_w_, in_h_;
    size_t conv_ws_bytes_ = 0;

    // Layers
    std::unique_ptr<Conv2D> conv_p4n_;     // 384 -> 128
    std::unique_ptr<Conv2D> conv_p3o_;     // 192 -> 64
    std::unique_ptr<Conv2D> conv_p3_down_; // 64 -> 64 s2
    std::unique_ptr<Conv2D> conv_p4o_;     // 192 -> 128
    std::unique_ptr<Conv2D> conv_p4_down_; // 128 -> 128 s2
    std::unique_ptr<Conv2D> conv_p5o_;     // 384 -> 256

    // Buffers
    float *d_up_p5_ = nullptr;
    float *d_cat_top1_ = nullptr;
    float *d_p4_n_ = nullptr;
    float *d_up_p4n_ = nullptr;
    float *d_cat_top2_ = nullptr;
    float *d_p3_out_ = nullptr;
    float *d_p3_down_ = nullptr;
    float *d_cat_bot1_ = nullptr;
    float *d_p4_out_ = nullptr;
    float *d_p4_down_ = nullptr;
    float *d_cat_bot2_ = nullptr;
    float *d_p5_out_ = nullptr;
};

#endif // NECK_H
