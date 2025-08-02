#pragma once
#include "tensor.hpp"
#include <cuda_fp16.h>

void add_positional_embedding(
    Tensor& patches,              // [B, N, D] - modified in-place
    const Tensor& pos_embed,      // [1, N, D]
    cudaStream_t stream);

void add_class_token(
    const Tensor& patches,        // [B, N-1, D]
    const Tensor& class_token,    // [1, 1, D]
    Tensor& output,              // [B, N, D]
    cudaStream_t stream);