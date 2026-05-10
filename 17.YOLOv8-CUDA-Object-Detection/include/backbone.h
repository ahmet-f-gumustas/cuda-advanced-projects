#ifndef BACKBONE_H
#define BACKBONE_H

#include "conv2d.h"
#include <cudnn.h>
#include <memory>

// Simplified CSPDarknet-style backbone for YOLOv8.
// Input:  [B, 3, 640, 640]
// Output: P3 [B, 64, 80, 80], P4 [B, 128, 40, 40], P5 [B, 256, 20, 20]
//
// Stages: stem(s2) → down1(s2) → bot1 → down2(s2) → bot2(=P3)
//                → down3(s2) → bot3(=P4) → down4(s2) → bot4 → SPPF(=P5)
class Backbone {
public:
    Backbone(cudnnHandle_t handle, int in_w, int in_h, unsigned seed);
    ~Backbone();

    // Returns required workspace bytes for cuDNN convs (max across all convs).
    size_t conv_workspace_bytes() const { return conv_ws_bytes_; }

    void forward(const float* d_input, void* workspace, size_t ws_bytes,
                 cudaStream_t stream = 0);

    // Output pointers after forward() — valid until next forward / destruction.
    const float* p3() const { return d_p3_; }
    const float* p4() const { return d_p4_; }
    const float* p5() const { return d_p5_; }

    int p3_c() const { return 64; }
    int p4_c() const { return 128; }
    int p5_c() const { return 256; }
    int p3_h() const { return in_h_ / 8; }
    int p3_w() const { return in_w_ / 8; }
    int p4_h() const { return in_h_ / 16; }
    int p4_w() const { return in_w_ / 16; }
    int p5_h() const { return in_h_ / 32; }
    int p5_w() const { return in_w_ / 32; }

private:
    cudnnHandle_t handle_;
    int in_w_, in_h_;
    size_t conv_ws_bytes_ = 0;

    // Convolution layers
    std::unique_ptr<Conv2D> stem_;     // 3 -> 16, s2
    std::unique_ptr<Conv2D> down1_;    // 16 -> 32, s2
    std::unique_ptr<Conv2D> bot1_a_;   // 32 -> 32, 3x3
    std::unique_ptr<Conv2D> bot1_b_;   // 32 -> 32, 3x3
    std::unique_ptr<Conv2D> down2_;    // 32 -> 64, s2
    std::unique_ptr<Conv2D> bot2_a_;   // 64 -> 64
    std::unique_ptr<Conv2D> bot2_b_;
    std::unique_ptr<Conv2D> down3_;    // 64 -> 128, s2
    std::unique_ptr<Conv2D> bot3_a_;
    std::unique_ptr<Conv2D> bot3_b_;
    std::unique_ptr<Conv2D> down4_;    // 128 -> 256, s2
    std::unique_ptr<Conv2D> bot4_a_;
    std::unique_ptr<Conv2D> bot4_b_;
    std::unique_ptr<Conv2D> sppf_in_;  // 256 -> 128, 1x1
    std::unique_ptr<Conv2D> sppf_out_; // 512 -> 256, 1x1

    // Intermediate buffers (device)
    float *d_stem_ = nullptr;
    float *d_d1_ = nullptr, *d_b1a_ = nullptr, *d_b1b_ = nullptr;
    float *d_d2_ = nullptr, *d_b2a_ = nullptr, *d_b2b_ = nullptr;
    float *d_p3_ = nullptr;
    float *d_d3_ = nullptr, *d_b3a_ = nullptr, *d_b3b_ = nullptr;
    float *d_p4_ = nullptr;
    float *d_d4_ = nullptr, *d_b4a_ = nullptr, *d_b4b_ = nullptr;
    float *d_sppf_in_ = nullptr;
    float *d_sppf_y1_ = nullptr, *d_sppf_y2_ = nullptr, *d_sppf_y3_ = nullptr;
    float *d_sppf_cat_ = nullptr;
    float *d_sppf_cat2_ = nullptr;
    float *d_p5_ = nullptr;
};

#endif // BACKBONE_H
