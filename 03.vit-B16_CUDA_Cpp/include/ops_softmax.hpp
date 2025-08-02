#pragma once
#include "tensor.hpp"
#include <cuda_fp16.h>

void softmax_forward(
    const Tensor& input,    // [B, H, S, S]
    Tensor& output,         // [B, H, S, S]
    cudaStream_t stream);

// CPU reference for validation
void softmax_cpu_reference(
    const float* input,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len);