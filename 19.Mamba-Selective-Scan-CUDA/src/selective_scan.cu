#include "selective_scan.cuh"

#include "cuda_utils.h"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr int kScanThreads = 256;
constexpr int kFusedThreads = 128;

struct ScanPair {
    float a;
    float b;
};

// Returns right(left(x)). This order makes an inclusive prefix scan match the
// time order of the recurrence h[t] = a[t] * h[t - 1] + b[t].
struct Compose {
    __host__ __device__ ScanPair operator()(const ScanPair& left,
                                            const ScanPair& right) const {
        return {right.a * left.a, right.a * left.b + right.b};
    }
};

__device__ __forceinline__ ScanPair warp_inclusive_scan(ScanPair value) {
    const int lane = threadIdx.x & 31;
    const Compose compose{};
    for (int offset = 1; offset < 32; offset <<= 1) {
        const ScanPair previous{
            __shfl_up_sync(0xffffffffu, value.a, offset),
            __shfl_up_sync(0xffffffffu, value.b, offset)};
        if (lane >= offset) {
            value = compose(previous, value);
        }
    }
    return value;
}

__device__ __forceinline__ float softplus(float value) {
    if (value > 20.0f) {
        return value;
    }
    if (value < -20.0f) {
        return expf(value);
    }
    return log1pf(expf(value));
}

__device__ __forceinline__ ScanPair recurrence_pair(const float* u,
                                                     const float* delta,
                                                     const float* A,
                                                     const float* B,
                                                     int batch,
                                                     int token,
                                                     int channel,
                                                     int state,
                                                     const SelectiveScanConfig& config) {
    const size_t token_index =
        (static_cast<size_t>(batch) * config.seq_len + token) * config.dim + channel;
    const size_t parameter_index =
        (static_cast<size_t>(batch) * config.seq_len + token) * config.state_size + state;
    const size_t matrix_index = static_cast<size_t>(channel) * config.state_size + state;
    const float step = softplus(delta[token_index] + config.delta_bias);
    return {expf(step * A[matrix_index]), step * B[parameter_index] * u[token_index]};
}

__global__ void naive_state_kernel(const float* u,
                                   const float* delta,
                                   const float* A,
                                   const float* B,
                                   float* states,
                                   SelectiveScanConfig config) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = config.batch * config.dim * config.state_size;
    if (row >= rows) {
        return;
    }

    const int state = row % config.state_size;
    const int channel = (row / config.state_size) % config.dim;
    const int batch = row / (config.state_size * config.dim);
    float hidden = 0.0f;

    for (int token = 0; token < config.seq_len; ++token) {
        const ScanPair pair = recurrence_pair(u, delta, A, B, batch, token,
                                              channel, state, config);
        hidden = pair.a * hidden + pair.b;
        const size_t output_index =
            ((static_cast<size_t>(batch) * config.seq_len + token) * config.dim + channel) *
                config.state_size + state;
        states[output_index] = hidden;
    }
}

__global__ void parallel_state_kernel(const float* u,
                                      const float* delta,
                                      const float* A,
                                      const float* B,
                                      float* states,
                                      SelectiveScanConfig config) {
    __shared__ ScanPair warp_prefix[kScanThreads / 32];
    __shared__ ScanPair carry;

    const int row = blockIdx.x;
    const int state = row % config.state_size;
    const int channel = (row / config.state_size) % config.dim;
    const int batch = row / (config.state_size * config.dim);
    const int lane = threadIdx.x;

    if (lane == 0) {
        carry = {1.0f, 0.0f};
    }
    __syncthreads();

    const Compose compose{};
    for (int tile = 0; tile < config.seq_len; tile += kScanThreads) {
        const int token = tile + lane;
        const int valid = min(kScanThreads, config.seq_len - tile);
        ScanPair value{1.0f, 0.0f};
        if (lane < valid) {
            value = recurrence_pair(u, delta, A, B, batch, token, channel, state, config);
        }
        if (lane == 0) {
            value = compose(carry, value);
        }

        ScanPair prefix = warp_inclusive_scan(value);
        const int warp = lane / 32;
        const int warp_lane = lane & 31;
        if (warp_lane == 31) {
            warp_prefix[warp] = prefix;
        }
        __syncthreads();

        if (warp == 0) {
            ScanPair warp_value = warp_lane < kScanThreads / 32
                                      ? warp_prefix[warp_lane]
                                      : ScanPair{1.0f, 0.0f};
            warp_value = warp_inclusive_scan(warp_value);
            if (warp_lane < kScanThreads / 32) {
                warp_prefix[warp_lane] = warp_value;
            }
        }
        __syncthreads();

        if (warp > 0) {
            prefix = compose(warp_prefix[warp - 1], prefix);
        }
        if (lane < valid) {
            const size_t output_index =
                ((static_cast<size_t>(batch) * config.seq_len + token) * config.dim + channel) *
                    config.state_size + state;
            states[output_index] = prefix.b;
            if (lane == valid - 1) {
                carry = prefix;
            }
        }
        __syncthreads();
    }
}

template <int BlockThreads>
__global__ void cub_state_kernel(const float* u,
                                 const float* delta,
                                 const float* A,
                                 const float* B,
                                 float* states,
                                 SelectiveScanConfig config) {
    using BlockScan = cub::BlockScan<ScanPair, BlockThreads>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ ScanPair carry;

    const int row = blockIdx.x;
    const int state = row % config.state_size;
    const int channel = (row / config.state_size) % config.dim;
    const int batch = row / (config.state_size * config.dim);
    const int lane = threadIdx.x;

    if (lane == 0) {
        carry = {1.0f, 0.0f};
    }
    __syncthreads();

    const Compose compose{};
    for (int tile = 0; tile < config.seq_len; tile += BlockThreads) {
        const int token = tile + lane;
        const int valid = min(BlockThreads, config.seq_len - tile);
        ScanPair value{1.0f, 0.0f};
        if (lane < valid) {
            value = recurrence_pair(u, delta, A, B, batch, token, channel, state, config);
        }
        if (lane == 0) {
            value = compose(carry, value);
        }

        ScanPair prefix{};
        BlockScan(scan_storage).InclusiveScan(value, prefix, compose);
        if (lane < valid) {
            const size_t output_index =
                ((static_cast<size_t>(batch) * config.seq_len + token) * config.dim + channel) *
                    config.state_size + state;
            states[output_index] = prefix.b;
            if (lane == valid - 1) {
                carry = prefix;
            }
        }
        __syncthreads();
    }
}

__global__ void state_output_kernel(const float* u,
                                    const float* C,
                                    const float* D,
                                    const float* states,
                                    float* y,
                                    SelectiveScanConfig config) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = config.batch * config.seq_len * config.dim;
    if (index >= elements) {
        return;
    }

    const int channel = index % config.dim;
    const int token_row = index / config.dim;
    float output = D[channel] * u[index];
    const size_t state_base = static_cast<size_t>(index) * config.state_size;
    const size_t parameter_base = static_cast<size_t>(token_row) * config.state_size;
    for (int state = 0; state < config.state_size; ++state) {
        output += C[parameter_base + state] * states[state_base + state];
    }
    y[index] = output;
}

__global__ void fused_scan_kernel(const float* u,
                                  const float* delta,
                                  const float* A,
                                  const float* B,
                                  const float* C,
                                  const float* D,
                                  float* y,
                                  SelectiveScanConfig config) {
    __shared__ float reduction[kFusedThreads];
    const int row = blockIdx.x;
    const int channel = row % config.dim;
    const int batch = row / config.dim;
    const int state = threadIdx.x;
    float hidden = 0.0f;

    for (int token = 0; token < config.seq_len; ++token) {
        const size_t token_index =
            (static_cast<size_t>(batch) * config.seq_len + token) * config.dim + channel;
        float contribution = 0.0f;
        if (state < config.state_size) {
            const ScanPair pair = recurrence_pair(u, delta, A, B, batch, token,
                                                  channel, state, config);
            hidden = pair.a * hidden + pair.b;
            const size_t parameter_index =
                (static_cast<size_t>(batch) * config.seq_len + token) * config.state_size + state;
            contribution = C[parameter_index] * hidden;
        }
        reduction[state] = contribution;
        __syncthreads();

        for (int stride = kFusedThreads / 2; stride > 0; stride >>= 1) {
            if (state < stride) {
                reduction[state] += reduction[state + stride];
            }
            __syncthreads();
        }
        if (state == 0) {
            y[token_index] = reduction[0] + D[channel] * u[token_index];
        }
        __syncthreads();
    }
}

__global__ void fused_scan_fp16_kernel(const __half* u,
                                       const __half* delta,
                                       const __half* A,
                                       const __half* B,
                                       const __half* C,
                                       const __half* D,
                                       __half* y,
                                       SelectiveScanConfig config) {
    __shared__ float reduction[kFusedThreads];
    const int row = blockIdx.x;
    const int channel = row % config.dim;
    const int batch = row / config.dim;
    const int state = threadIdx.x;
    float hidden = 0.0f;

    for (int token = 0; token < config.seq_len; ++token) {
        const size_t token_index =
            (static_cast<size_t>(batch) * config.seq_len + token) * config.dim + channel;
        float contribution = 0.0f;
        if (state < config.state_size) {
            const size_t parameter_index =
                (static_cast<size_t>(batch) * config.seq_len + token) * config.state_size + state;
            const size_t matrix_index = static_cast<size_t>(channel) * config.state_size + state;
            const float step = softplus(__half2float(delta[token_index]) + config.delta_bias);
            const float transition = expf(step * __half2float(A[matrix_index]));
            const float input = step * __half2float(B[parameter_index]) *
                                __half2float(u[token_index]);
            hidden = transition * hidden + input;
            contribution = __half2float(C[parameter_index]) * hidden;
        }
        reduction[state] = contribution;
        __syncthreads();

        for (int stride = kFusedThreads / 2; stride > 0; stride >>= 1) {
            if (state < stride) {
                reduction[state] += reduction[state + stride];
            }
            __syncthreads();
        }
        if (state == 0) {
            const float output = reduction[0] + __half2float(D[channel]) *
                                                __half2float(u[token_index]);
            y[token_index] = __float2half(output);
        }
        __syncthreads();
    }
}

void validate(const SelectiveScanConfig& config) {
    if (config.batch <= 0 || config.seq_len <= 0 || config.dim <= 0 ||
        config.state_size <= 0 || config.state_size > kFusedThreads) {
        throw std::invalid_argument(
            "selective scan dimensions must be positive and state_size must be <= 128");
    }
}

}  // namespace

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
                            cudaStream_t stream) {
    validate(config);
    const int state_rows = config.batch * config.dim * config.state_size;

    if (algorithm == ScanAlgorithm::Fused) {
        fused_scan_kernel<<<config.batch * config.dim, kFusedThreads, 0, stream>>>(
            d_u, d_delta, d_A, d_B, d_C, d_D, d_y, config);
        CUDA_CHECK_LAST();
        return;
    }
    if (d_state == nullptr) {
        throw std::invalid_argument("non-fused selective scan requires a state workspace");
    }

    if (algorithm == ScanAlgorithm::Naive) {
        constexpr int threads = 256;
        const int blocks = (state_rows + threads - 1) / threads;
        naive_state_kernel<<<blocks, threads, 0, stream>>>(d_u, d_delta, d_A, d_B,
                                                           d_state, config);
    } else if (algorithm == ScanAlgorithm::Parallel) {
        parallel_state_kernel<<<state_rows, kScanThreads, 0, stream>>>(
            d_u, d_delta, d_A, d_B, d_state, config);
    } else if (algorithm == ScanAlgorithm::Cub) {
        cub_state_kernel<kScanThreads><<<state_rows, kScanThreads, 0, stream>>>(
            d_u, d_delta, d_A, d_B, d_state, config);
    } else {
        throw std::invalid_argument("unknown selective scan algorithm");
    }
    CUDA_CHECK_LAST();

    constexpr int threads = 256;
    const int elements = static_cast<int>(config.token_elements());
    const int blocks = (elements + threads - 1) / threads;
    state_output_kernel<<<blocks, threads, 0, stream>>>(d_u, d_C, d_D, d_state, d_y,
                                                        config);
    CUDA_CHECK_LAST();
}

void selective_scan_forward_fp16(const __half* d_u,
                                 const __half* d_delta,
                                 const __half* d_A,
                                 const __half* d_B,
                                 const __half* d_C,
                                 const __half* d_D,
                                 __half* d_y,
                                 const SelectiveScanConfig& config,
                                 cudaStream_t stream) {
    validate(config);
    fused_scan_fp16_kernel<<<config.batch * config.dim, kFusedThreads, 0, stream>>>(
        d_u, d_delta, d_A, d_B, d_C, d_D, d_y, config);
    CUDA_CHECK_LAST();
}

size_t selective_scan_workspace_bytes(const SelectiveScanConfig& config,
                                      ScanAlgorithm algorithm) {
    return algorithm == ScanAlgorithm::Fused ? 0 : config.state_elements() * sizeof(float);
}
