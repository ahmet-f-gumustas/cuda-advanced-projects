#ifndef MAMBA_SELECTIVE_SCAN_CUH
#define MAMBA_SELECTIVE_SCAN_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

struct SelectiveScanConfig {
    int batch = 1;
    int seq_len = 256;
    int dim = 64;
    int state_size = 16;
    float delta_bias = 0.0f;

    size_t token_elements() const {
        return static_cast<size_t>(batch) * seq_len * dim;
    }

    size_t parameter_elements() const {
        return static_cast<size_t>(batch) * seq_len * state_size;
    }

    size_t state_elements() const {
        return token_elements() * state_size;
    }
};

enum class ScanAlgorithm {
    Naive,
    Parallel,
    Cub,
    Fused
};

// Layouts:
// u, delta, y: [batch, seq_len, dim]
// A:           [dim, state_size]
// B, C:        [batch, seq_len, state_size]
// D:           [dim]
// state:       [batch, seq_len, dim, state_size]
void selective_scan_forward(const float* d_u,
                            const float* d_delta,
                            const float* d_A,
                            const float* d_B,
                            const float* d_C,
                            const float* d_D,
                            float* d_y,
                            float* d_state,
                            const SelectiveScanConfig& config,
                            ScanAlgorithm algorithm,
                            cudaStream_t stream = 0);

// FP16 storage with FP32 recurrence and reduction. This path is fused and does
// not allocate the [batch, seq_len, dim, state_size] state tensor.
void selective_scan_forward_fp16(const __half* d_u,
                                 const __half* d_delta,
                                 const __half* d_A,
                                 const __half* d_B,
                                 const __half* d_C,
                                 const __half* d_D,
                                 __half* d_y,
                                 const SelectiveScanConfig& config,
                                 cudaStream_t stream = 0);

size_t selective_scan_workspace_bytes(const SelectiveScanConfig& config,
                                      ScanAlgorithm algorithm);

#endif
