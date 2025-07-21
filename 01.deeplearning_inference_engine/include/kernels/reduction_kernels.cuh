#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "../core/types.h"

namespace deep_engine {
namespace kernels {

// Basic reduction operations
template<typename T>
__global__ void reduce_sum_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void reduce_mean_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void reduce_max_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void reduce_min_kernel(const T* input, T* output, int size);

// Axis-specific reductions
template<typename T>
__global__ void reduce_sum_axis_kernel(const T* input, T* output,
                                      int* input_shape, int* output_shape,
                                      int ndim, int axis);

template<typename T>
__global__ void reduce_mean_axis_kernel(const T* input, T* output,
                                       int* input_shape, int* output_shape,
                                       int ndim, int axis);

// Variance and standard deviation
template<typename T>
__global__ void variance_kernel(const T* input, const T* mean, T* output,
                               int size, bool unbiased);

template<typename T>
__global__ void std_kernel(const T* input, const T* mean, T* output,
                          int size, bool unbiased);

// L1 and L2 norms
template<typename T>
__global__ void l1_norm_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void l2_norm_kernel(const T* input, T* output, int size);

template<typename T>
__global__ void lp_norm_kernel(const T* input, T* output, float p, int size);

// Argmax and argmin
template<typename T>
__global__ void argmax_kernel(const T* input, int* output, int size);

template<typename T>
__global__ void argmin_kernel(const T* input, int* output, int size);

template<typename T>
__global__ void argmax_axis_kernel(const T* input, int* output,
                                  int* input_shape, int ndim, int axis);

// Top-k operations
template<typename T>
__global__ void topk_kernel(const T* input, T* values, int* indices,
                           int batch_size, int input_size, int k);

// Segmented reductions
template<typename T>
__global__ void segment_reduce_sum_kernel(const T* input, T* output,
                                         const int* segment_ids,
                                         int size, int num_segments);

// Block-wise reduction utilities
template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val);

template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_max(T val);

template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_min(T val);

// Warp-level primitives
template<typename T>
__device__ T warp_reduce_sum(T val);

template<typename T>
__device__ T warp_reduce_max(T val);

template<typename T>
__device__ T warp_reduce_min(T val);

// CUB-based efficient reductions
template<typename T>
class CubReductionWrapper {
public:
    CubReductionWrapper();
    ~CubReductionWrapper();
    
    void reduce_sum(const T* input, T* output, int size, cudaStream_t stream);
    void reduce_max(const T* input, T* output, int size, cudaStream_t stream);
    void reduce_min(const T* input, T* output, int size, cudaStream_t stream);
    void reduce_argmax(const T* input, int* output, int size, cudaStream_t stream);
    void reduce_argmin(const T* input, int* output, int size, cudaStream_t stream);
    
private:
    void* d_temp_storage_;
    size_t temp_storage_bytes_;
};

// Launcher functions
template<typename T>
void launch_reduce_sum(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_reduce_mean(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_reduce_max(const T* input, T* output, int size, cudaStream_t stream);

template<typename T>
void launch_reduce_sum_axis(const T* input, T* output,
                           const std::vector<int>& input_shape,
                           int axis, cudaStream_t stream);

template<typename T>
void launch_variance(const T* input, const T* mean, T* output,
                    int size, bool unbiased, cudaStream_t stream);

template<typename T>
void launch_topk(const T* input, T* values, int* indices,
                int batch_size, int input_size, int k,
                cudaStream_t stream);

} // namespace kernels
} // namespace deep_engine