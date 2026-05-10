#ifndef CONV2D_H
#define CONV2D_H

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstddef>

// Conv2D module with optional fused bias add and SiLU activation.
//
// Weights are held on-device and randomly initialized (this project is
// architecture-focused; trained weights aren't required to demonstrate the
// pipeline). BN folding is handled at construction: a "ConvBNSiLU" block is
// just a Conv2D with bias and SiLU enabled.
class Conv2D {
public:
    Conv2D(cudnnHandle_t handle, int in_c, int out_c,
           int kernel, int stride, int pad,
           bool with_silu, unsigned seed);
    ~Conv2D();

    // Compute output spatial dimensions for a given input spatial size.
    int out_h(int in_h) const { return (in_h + 2 * pad_ - kernel_) / stride_ + 1; }
    int out_w(int in_w) const { return out_h(in_w); }  // symmetric

    // Query cuDNN-required workspace size for this (N, H, W).
    size_t workspace_bytes(int n, int in_h, int in_w);

    // Forward: d_in [N, in_c, H, W] -> d_out [N, out_c, out_h, out_w]
    void forward(const float* d_in, float* d_out,
                 int n, int in_h, int in_w,
                 void* workspace, size_t workspace_bytes,
                 cudaStream_t stream = 0);

    int in_c() const { return in_c_; }
    int out_c() const { return out_c_; }
    int kernel() const { return kernel_; }
    int stride() const { return stride_; }

private:
    cudnnHandle_t handle_;
    int in_c_, out_c_, kernel_, stride_, pad_;
    bool with_silu_;

    cudnnFilterDescriptor_t w_desc_ = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnTensorDescriptor_t b_desc_ = nullptr;
    cudnnActivationDescriptor_t act_desc_ = nullptr;

    float* d_w_ = nullptr;
    float* d_b_ = nullptr;
};

#endif // CONV2D_H
