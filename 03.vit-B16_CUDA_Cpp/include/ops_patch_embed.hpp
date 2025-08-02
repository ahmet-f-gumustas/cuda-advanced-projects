#pragma once
#include "tensor.hpp"
#include <cuda_fp16.h>

void patch_embed_forward(
    const Tensor& images,     // [B, 3, H, W]
    const Tensor& weight,     // [D, 3, P, P]
    const Tensor& bias,       // [D]
    Tensor& output,          // [B, N, D] where N = (H/P) * (W/P)
    int patch_size,
    cudaStream_t stream);

// Using unfold + GEMM approach
void patch_embed_unfold_gemm(
    const Tensor& images,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output,
    Tensor& workspace,
    int patch_size,
    cublasLtHandle_t cublaslt_handle,
    cudaStream_t stream);