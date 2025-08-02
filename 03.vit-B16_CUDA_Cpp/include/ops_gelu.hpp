#pragma once
#include "tensor.hpp"
#include <cuda_fp16.h>

void gelu_forward(
    const Tensor& input,    // [B, S, D]
    Tensor& output,         // [B, S, D]
    cudaStream_t stream);

// Fast approximation using tanh
void gelu_fast_forward(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream);