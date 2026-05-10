#include "conv2d.h"
#include "cuda_utils.h"
#include "yolo_kernels.cuh"

#include <random>
#include <vector>
#include <cmath>

Conv2D::Conv2D(cudnnHandle_t handle, int in_c, int out_c,
               int kernel, int stride, int pad,
               bool with_silu, unsigned seed)
    : handle_(handle), in_c_(in_c), out_c_(out_c),
      kernel_(kernel), stride_(stride), pad_(pad),
      with_silu_(with_silu) {

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc_));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           out_c_, in_c_, kernel_, kernel_));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc_, pad_, pad_, stride_, stride_, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(b_desc_, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, out_c_, 1, 1));

    if (with_silu_) {
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
        // cuDNN has no native SiLU activation; we'll call our custom kernel
        // post-conv instead of binding here. Descriptor kept for symmetry.
    }

    // Random init (Kaiming-ish): N(0, sqrt(2 / fan_in))
    int w_count = out_c_ * in_c_ * kernel_ * kernel_;
    std::mt19937 rng(seed);
    float fan_in = (float)(in_c_ * kernel_ * kernel_);
    float std = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std);

    std::vector<float> h_w(w_count);
    for (auto& v : h_w) v = dist(rng);
    std::vector<float> h_b(out_c_, 0.0f);

    CUDA_CHECK(cudaMalloc(&d_w_, sizeof(float) * w_count));
    CUDA_CHECK(cudaMalloc(&d_b_, sizeof(float) * out_c_));
    CUDA_CHECK(cudaMemcpy(d_w_, h_w.data(), sizeof(float) * w_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, h_b.data(), sizeof(float) * out_c_,
                          cudaMemcpyHostToDevice));
}

Conv2D::~Conv2D() {
    if (d_w_) cudaFree(d_w_);
    if (d_b_) cudaFree(d_b_);
    if (w_desc_) cudnnDestroyFilterDescriptor(w_desc_);
    if (conv_desc_) cudnnDestroyConvolutionDescriptor(conv_desc_);
    if (b_desc_) cudnnDestroyTensorDescriptor(b_desc_);
    if (act_desc_) cudnnDestroyActivationDescriptor(act_desc_);
}

size_t Conv2D::workspace_bytes(int n, int in_h, int in_w) {
    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               n, in_c_, in_h, in_w);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               n, out_c_, out_h(in_h), out_w(in_w));
    size_t ws = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle_, in_desc, w_desc_, conv_desc_, out_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &ws);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    return ws;
}

void Conv2D::forward(const float* d_in, float* d_out,
                     int n, int in_h, int in_w,
                     void* workspace, size_t workspace_bytes_avail,
                     cudaStream_t stream) {
    cudnnTensorDescriptor_t in_desc, out_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           n, in_c_, in_h, in_w));
    int oh = out_h(in_h);
    int ow = out_w(in_w);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           n, out_c_, oh, ow));

    CUDNN_CHECK(cudnnSetStream(handle_, stream));

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle_, &alpha, in_desc, d_in, w_desc_, d_w_, conv_desc_,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        workspace, workspace_bytes_avail, &beta, out_desc, d_out));

    CUDNN_CHECK(cudnnAddTensor(handle_, &alpha, b_desc_, d_b_,
                               &alpha, out_desc, d_out));

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);

    if (with_silu_) {
        launch_silu(d_out, n * out_c_ * oh * ow, stream);
    }
}
