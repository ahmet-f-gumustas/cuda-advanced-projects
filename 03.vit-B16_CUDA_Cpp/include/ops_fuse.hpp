#pragma once
#include "tensor.hpp"
#include <cuda_fp16.h>

// Fused bias + residual + dropout
void fused_bias_residual_dropout(
    const Tensor& input,      // [B, S, D]
    const Tensor& bias,       // [D]
    const Tensor& residual,   // [B, S, D]
    Tensor& output,           // [B, S, D]
    float dropout_prob,
    bool training,
    cudaStream_t stream);

// Simplified version without dropout for inference
void fused_bias_residual(
    const Tensor& input,      // [B, S, D]
    const Tensor& bias,       // [D]
    const Tensor& residual,   // [B, S, D]
    Tensor& output,           // [B, S, D]
    cudaStream_t stream);