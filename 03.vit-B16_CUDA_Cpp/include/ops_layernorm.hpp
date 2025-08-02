#pragma once
#include "tensor.hpp"
#include <cuda_fp16.h>

void layernorm_forward(
    const Tensor& input,    // [B, S, D]
    const Tensor& gamma,    // [D]
    const Tensor& beta,     // [D]
    Tensor& output,         // [B, S, D]
    float eps,
    cudaStream_t stream);

// CPU reference for validation
void layernorm_cpu_reference(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps);