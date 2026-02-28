#include "../include/transformer_kernels.cuh"
#include "../include/cuda_utils.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <iostream>

// ============================================================================
// Host-side quantization helpers
// ============================================================================

// Quantize a FP16 device matrix to INT8 in-place (allocates INT8 output)
// Returns: d_int8 output and d_scales per-row
// Caller is responsible for freeing d_int8 and d_scales
void quantize_matrix(const half* d_fp16,
                     int8_t**    d_int8_out,
                     float**     d_scales_out,
                     int         rows,
                     int         cols)
{
    CUDA_CHECK(cudaMalloc(d_int8_out,   (size_t)rows * cols * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(d_scales_out, rows * sizeof(float)));

    // One block per row, up to 512 threads
    int block = (cols < 512) ? cols : 512;
    int smem  = ((block + 31) / 32) * sizeof(float);  // warp count * float

    quantize_fp16_to_int8_kernel<<<rows, block, smem>>>(
        d_fp16, *d_int8_out, *d_scales_out, rows, cols);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Dequantize an INT8 device matrix back to FP16 (allocates FP16 output)
void dequantize_matrix(const int8_t* d_int8,
                       const float*  d_scales,
                       half**        d_fp16_out,
                       int           rows,
                       int           cols)
{
    CUDA_CHECK(cudaMalloc(d_fp16_out, (size_t)rows * cols * sizeof(half)));

    int block = (cols < 512) ? cols : 512;
    dequantize_int8_to_fp16_kernel<<<rows, block>>>(
        d_int8, *d_fp16_out, d_scales, rows, cols);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Print quantization statistics (host-side, copies a small sample)
void print_quant_stats(const half* d_fp16,
                       const int8_t* d_int8,
                       const float* d_scales,
                       int rows, int cols,
                       const char* name)
{
    // Copy first row of scales and a few values for inspection
    float h_scales[4] = {};
    int   n_check = (rows < 4) ? rows : 4;
    CUDA_CHECK(cudaMemcpy(h_scales, d_scales, n_check * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "[Quant] " << name << "  rows=" << rows << " cols=" << cols << "\n";
    for (int r = 0; r < n_check; ++r)
        std::cout << "  row" << r << " scale=" << h_scales[r] << "\n";
}
