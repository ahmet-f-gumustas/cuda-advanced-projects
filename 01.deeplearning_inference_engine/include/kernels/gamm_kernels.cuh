#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "../core/types.h"

namespace deep_engine {
namespace kernels {

// Basic GEMM kernel
template<typename T>
__global__ void gemm_kernel(const T* A, const T* B, T* C,
                           int M, int N, int K,
                           T alpha, T beta,
                           bool transA, bool transB);

// Optimized GEMM with shared memory tiling
template<typename T, int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_tiled_kernel(const T* A, const T* B, T* C,
                                 int M, int N, int K,
                                 T alpha, T beta,
                                 bool transA, bool transB);

// GEMM with bias and activation fusion
template<typename T>
__global__ void gemm_bias_relu_kernel(const T* A, const T* B, const T* bias, T* C,
                                     int M, int N, int K,
                                     T alpha, T beta,
                                     bool transA, bool transB);

// INT8 GEMM kernel
__global__ void int8_gemm_kernel(const int8_t* A, const int8_t* B, int32_t* C,
                                int M, int N, int K,
                                bool transA, bool transB);

// Mixed precision GEMM (FP16 compute, FP32 accumulate)
__global__ void mixed_precision_gemm_kernel(const __half* A, const __half* B, float* C,
                                           int M, int N, int K,
                                           float alpha, float beta,
                                           bool transA, bool transB);

// Tensor Core GEMM for Ampere and newer
#if __CUDA_ARCH__ >= 800
__global__ void tensor_core_gemm_kernel(const __half* A, const __half* B, __half* C,
                                       int M, int N, int K,
                                       __half alpha, __half beta);

__global__ void tensor_core_int8_gemm_kernel(const int8_t* A, const int8_t* B, int32_t* C,
                                            int M, int N, int K);
#endif

// Batched GEMM
template<typename T>
__global__ void batched_gemm_kernel(const T** A_array, const T** B_array, T** C_array,
                                   int M, int N, int K,
                                   T alpha, T beta,
                                   bool transA, bool transB,
                                   int batch_count);

// Strided batched GEMM
template<typename T>
__global__ void strided_batched_gemm_kernel(const T* A, const T* B, T* C,
                                           int M, int N, int K,
                                           T alpha, T beta,
                                           bool transA, bool transB,
                                           int strideA, int strideB, int strideC,
                                           int batch_count);

// GEMM launcher functions
template<typename T>
void launch_gemm(const T* A, const T* B, T* C,
                int M, int N, int K,
                T alpha, T beta,
                bool transA, bool transB,
                cudaStream_t stream);

template<typename T>
void launch_gemm_optimized(const T* A, const T* B, T* C,
                          int M, int N, int K,
                          T alpha, T beta,
                          bool transA, bool transB,
                          cudaStream_t stream);

// cuBLAS wrappers
class CublasGemmWrapper {
public:
    CublasGemmWrapper();
    ~CublasGemmWrapper();
    
    // Standard GEMM
    void gemm(const float* A, const float* B, float* C,
             int M, int N, int K,
             float alpha, float beta,
             bool transA, bool transB,
             cudaStream_t stream);
    
    void gemm(const __half* A, const __half* B, __half* C,
             int M, int N, int K,
             __half alpha, __half beta,
             bool transA, bool transB,
             cudaStream_t stream);
    
    // Batched GEMM
    void batched_gemm(const float** A_array, const float** B_array, float** C_array,
                     int M, int N, int K,
                     float alpha, float beta,
                     bool transA, bool transB,
                     int batch_count,
                     cudaStream_t stream);
    
    // Strided batched GEMM
    void strided_batched_gemm(const float* A, const float* B, float* C,
                             int M, int N, int K,
                             float alpha, float beta,
                             bool transA, bool transB,
                             long long strideA, long long strideB, long long strideC,
                             int batch_count,
                             cudaStream_t stream);
    
private:
    cublasHandle_t handle_;
    cublasLtHandle_t lt_handle_;
};

} // namespace kernels
} // namespace deep_engine