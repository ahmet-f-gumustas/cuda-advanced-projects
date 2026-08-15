#include "mamba_block.h"

#include "cuda_utils.h"
#include "selective_scan.cuh"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

__global__ void rms_norm_kernel(const float* input,
                                const float* weight,
                                float* output,
                                int rows,
                                int width) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[static_cast<size_t>(row) * width + column];
        sum += value * value;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(scratch[0] / width + 1.0e-5f);
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * width + column;
        output[index] = input[index] * inverse_rms * weight[column];
    }
}

// Weight layout is [output_width, input_width], row major.
__global__ void linear_kernel(const float* input,
                              const float* weight,
                              float* output,
                              int rows,
                              int input_width,
                              int output_width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * output_width;
    if (index >= elements) {
        return;
    }
    const int row = index / output_width;
    const int output_column = index % output_width;
    float accumulator = 0.0f;
    for (int column = 0; column < input_width; ++column) {
        accumulator += input[static_cast<size_t>(row) * input_width + column] *
                       weight[static_cast<size_t>(output_column) * input_width + column];
    }
    output[index] = accumulator;
}

__global__ void split_projection_kernel(const float* projected,
                                        float* x,
                                        float* z,
                                        int elements,
                                        int inner_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / inner_dim;
    const int column = index % inner_dim;
    const size_t source = static_cast<size_t>(row) * (2 * inner_dim) + column;
    x[index] = projected[source];
    z[index] = projected[source + inner_dim];
}

__global__ void causal_conv1d_kernel(const float* input,
                                     const float* weight,
                                     const float* bias,
                                     float* output,
                                     int batch,
                                     int seq_len,
                                     int channels,
                                     int kernel_width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = batch * seq_len * channels;
    if (index >= elements) {
        return;
    }
    const int channel = index % channels;
    const int token = (index / channels) % seq_len;
    const int batch_index = index / (channels * seq_len);
    float accumulator = bias[channel];
    for (int tap = 0; tap < kernel_width; ++tap) {
        const int source_token = token - tap;
        if (source_token >= 0) {
            const size_t source =
                (static_cast<size_t>(batch_index) * seq_len + source_token) * channels + channel;
            accumulator += input[source] *
                           weight[static_cast<size_t>(channel) * kernel_width + tap];
        }
    }
    output[index] = silu(accumulator);
}

__global__ void split_parameters_kernel(const float* parameters,
                                        float* B,
                                        float* C,
                                        int rows,
                                        int dt_rank,
                                        int state_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * state_size;
    if (index >= elements) {
        return;
    }
    const int row = index / state_size;
    const int state = index % state_size;
    const int width = dt_rank + 2 * state_size;
    B[index] = parameters[static_cast<size_t>(row) * width + dt_rank + state];
    C[index] = parameters[static_cast<size_t>(row) * width + dt_rank + state_size + state];
}

__global__ void delta_projection_kernel(const float* parameters,
                                        const float* weight,
                                        const float* bias,
                                        float* delta,
                                        int rows,
                                        int inner_dim,
                                        int dt_rank,
                                        int parameter_width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * inner_dim;
    if (index >= elements) {
        return;
    }
    const int row = index / inner_dim;
    const int channel = index % inner_dim;
    float accumulator = bias[channel];
    for (int rank = 0; rank < dt_rank; ++rank) {
        accumulator += parameters[static_cast<size_t>(row) * parameter_width + rank] *
                       weight[static_cast<size_t>(channel) * dt_rank + rank];
    }
    delta[index] = accumulator;
}

__global__ void gate_kernel(float* values, const float* gate, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        values[index] *= silu(gate[index]);
    }
}

__global__ void residual_kernel(const float* residual, float* output, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] += residual[index];
    }
}

int blocks_for(size_t elements) {
    return static_cast<int>((elements + kThreads - 1) / kThreads);
}

void require_size(const std::vector<float>& values, size_t expected, const char* name) {
    if (values.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has an invalid size");
    }
}

void free_device(float*& pointer) {
    if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
    }
}

}  // namespace

MambaWeights make_mamba_weights(const MambaConfig& config, unsigned seed) {
    if (config.model_dim <= 0 || config.inner_dim <= 0 || config.state_size <= 0 ||
        config.state_size > 128 || config.dt_rank <= 0 || config.conv_width <= 0) {
        throw std::invalid_argument("invalid Mamba configuration");
    }

    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto random_matrix = [&](size_t count, int fan_in) {
        std::vector<float> values(count);
        const float scale = 1.0f / std::sqrt(static_cast<float>(fan_in));
        for (float& value : values) {
            value = normal(generator) * scale;
        }
        return values;
    };

    MambaWeights weights;
    weights.norm.assign(config.model_dim, 1.0f);
    weights.in_proj = random_matrix(static_cast<size_t>(2 * config.inner_dim) * config.model_dim,
                                    config.model_dim);
    weights.conv_weight = random_matrix(
        static_cast<size_t>(config.inner_dim) * config.conv_width, config.conv_width);
    weights.conv_bias.assign(config.inner_dim, 0.0f);
    weights.x_proj = random_matrix(
        static_cast<size_t>(config.dt_rank + 2 * config.state_size) * config.inner_dim,
        config.inner_dim);
    weights.dt_proj = random_matrix(static_cast<size_t>(config.inner_dim) * config.dt_rank,
                                    config.dt_rank);
    weights.dt_bias.assign(config.inner_dim, -2.0f);
    weights.A.resize(static_cast<size_t>(config.inner_dim) * config.state_size);
    for (int channel = 0; channel < config.inner_dim; ++channel) {
        for (int state = 0; state < config.state_size; ++state) {
            weights.A[static_cast<size_t>(channel) * config.state_size + state] =
                -static_cast<float>(state + 1);
        }
    }
    weights.D.assign(config.inner_dim, 1.0f);
    weights.out_proj = random_matrix(
        static_cast<size_t>(config.model_dim) * config.inner_dim, config.inner_dim);
    return weights;
}

MambaBlock::MambaBlock(const MambaConfig& config) : config_(config) {
    if (config.batch <= 0 || config.seq_len <= 0 || config.model_dim <= 0 ||
        config.inner_dim <= 0 || config.state_size <= 0 || config.state_size > 128 ||
        config.dt_rank <= 0 || config.conv_width <= 0) {
        throw std::invalid_argument("invalid Mamba configuration");
    }
    allocate();
    load_weights(make_mamba_weights(config));
}

MambaBlock::~MambaBlock() {
    release();
}

void MambaBlock::allocate() {
    const size_t rows = static_cast<size_t>(config_.batch) * config_.seq_len;
    const size_t model_elements = rows * config_.model_dim;
    const size_t inner_elements = rows * config_.inner_dim;
    const size_t parameter_width = config_.dt_rank + 2 * config_.state_size;

    d_norm_ = device_alloc<float>(config_.model_dim);
    d_in_proj_ = device_alloc<float>(static_cast<size_t>(2 * config_.inner_dim) *
                                     config_.model_dim);
    d_conv_weight_ = device_alloc<float>(static_cast<size_t>(config_.inner_dim) *
                                         config_.conv_width);
    d_conv_bias_ = device_alloc<float>(config_.inner_dim);
    d_x_proj_ = device_alloc<float>(parameter_width * config_.inner_dim);
    d_dt_proj_ = device_alloc<float>(static_cast<size_t>(config_.inner_dim) * config_.dt_rank);
    d_dt_bias_ = device_alloc<float>(config_.inner_dim);
    d_A_ = device_alloc<float>(static_cast<size_t>(config_.inner_dim) * config_.state_size);
    d_D_ = device_alloc<float>(config_.inner_dim);
    d_out_proj_ = device_alloc<float>(static_cast<size_t>(config_.model_dim) * config_.inner_dim);

    d_normalized_ = device_alloc<float>(model_elements);
    d_projected_ = device_alloc<float>(2 * inner_elements);
    d_x_ = device_alloc<float>(inner_elements);
    d_z_ = device_alloc<float>(inner_elements);
    d_convolved_ = device_alloc<float>(inner_elements);
    d_parameters_ = device_alloc<float>(rows * parameter_width);
    d_delta_ = device_alloc<float>(inner_elements);
    d_B_ = device_alloc<float>(rows * config_.state_size);
    d_C_ = device_alloc<float>(rows * config_.state_size);
    d_scan_output_ = device_alloc<float>(inner_elements);
}

void MambaBlock::release() {
    free_device(d_norm_);
    free_device(d_in_proj_);
    free_device(d_conv_weight_);
    free_device(d_conv_bias_);
    free_device(d_x_proj_);
    free_device(d_dt_proj_);
    free_device(d_dt_bias_);
    free_device(d_A_);
    free_device(d_D_);
    free_device(d_out_proj_);
    free_device(d_normalized_);
    free_device(d_projected_);
    free_device(d_x_);
    free_device(d_z_);
    free_device(d_convolved_);
    free_device(d_parameters_);
    free_device(d_delta_);
    free_device(d_B_);
    free_device(d_C_);
    free_device(d_scan_output_);
}

void MambaBlock::load_weights(const MambaWeights& weights) {
    const size_t parameter_width = config_.dt_rank + 2 * config_.state_size;
    require_size(weights.norm, config_.model_dim, "norm");
    require_size(weights.in_proj,
                 static_cast<size_t>(2 * config_.inner_dim) * config_.model_dim, "in_proj");
    require_size(weights.conv_weight,
                 static_cast<size_t>(config_.inner_dim) * config_.conv_width, "conv_weight");
    require_size(weights.conv_bias, config_.inner_dim, "conv_bias");
    require_size(weights.x_proj, parameter_width * config_.inner_dim, "x_proj");
    require_size(weights.dt_proj,
                 static_cast<size_t>(config_.inner_dim) * config_.dt_rank, "dt_proj");
    require_size(weights.dt_bias, config_.inner_dim, "dt_bias");
    require_size(weights.A,
                 static_cast<size_t>(config_.inner_dim) * config_.state_size, "A");
    require_size(weights.D, config_.inner_dim, "D");
    require_size(weights.out_proj,
                 static_cast<size_t>(config_.model_dim) * config_.inner_dim, "out_proj");

    copy_to_device(d_norm_, weights.norm.data(), weights.norm.size());
    copy_to_device(d_in_proj_, weights.in_proj.data(), weights.in_proj.size());
    copy_to_device(d_conv_weight_, weights.conv_weight.data(), weights.conv_weight.size());
    copy_to_device(d_conv_bias_, weights.conv_bias.data(), weights.conv_bias.size());
    copy_to_device(d_x_proj_, weights.x_proj.data(), weights.x_proj.size());
    copy_to_device(d_dt_proj_, weights.dt_proj.data(), weights.dt_proj.size());
    copy_to_device(d_dt_bias_, weights.dt_bias.data(), weights.dt_bias.size());
    copy_to_device(d_A_, weights.A.data(), weights.A.size());
    copy_to_device(d_D_, weights.D.data(), weights.D.size());
    copy_to_device(d_out_proj_, weights.out_proj.data(), weights.out_proj.size());
}

void MambaBlock::forward(const float* d_input, float* d_output, cudaStream_t stream) {
    const int rows = config_.batch * config_.seq_len;
    const int model_elements = rows * config_.model_dim;
    const int inner_elements = rows * config_.inner_dim;
    const int parameter_width = config_.dt_rank + 2 * config_.state_size;

    rms_norm_kernel<<<rows, kThreads, kThreads * sizeof(float), stream>>>(
        d_input, d_norm_, d_normalized_, rows, config_.model_dim);
    linear_kernel<<<blocks_for(static_cast<size_t>(rows) * 2 * config_.inner_dim),
                    kThreads, 0, stream>>>(
        d_normalized_, d_in_proj_, d_projected_, rows, config_.model_dim,
        2 * config_.inner_dim);
    split_projection_kernel<<<blocks_for(inner_elements), kThreads, 0, stream>>>(
        d_projected_, d_x_, d_z_, inner_elements, config_.inner_dim);
    causal_conv1d_kernel<<<blocks_for(inner_elements), kThreads, 0, stream>>>(
        d_x_, d_conv_weight_, d_conv_bias_, d_convolved_, config_.batch,
        config_.seq_len, config_.inner_dim, config_.conv_width);
    linear_kernel<<<blocks_for(static_cast<size_t>(rows) * parameter_width),
                    kThreads, 0, stream>>>(
        d_convolved_, d_x_proj_, d_parameters_, rows, config_.inner_dim,
        parameter_width);
    split_parameters_kernel<<<blocks_for(static_cast<size_t>(rows) * config_.state_size),
                              kThreads, 0, stream>>>(
        d_parameters_, d_B_, d_C_, rows, config_.dt_rank, config_.state_size);
    delta_projection_kernel<<<blocks_for(inner_elements), kThreads, 0, stream>>>(
        d_parameters_, d_dt_proj_, d_dt_bias_, d_delta_, rows, config_.inner_dim,
        config_.dt_rank, parameter_width);

    SelectiveScanConfig scan_config;
    scan_config.batch = config_.batch;
    scan_config.seq_len = config_.seq_len;
    scan_config.dim = config_.inner_dim;
    scan_config.state_size = config_.state_size;
    selective_scan_forward(d_convolved_, d_delta_, d_A_, d_B_, d_C_, d_D_,
                           d_scan_output_, nullptr, scan_config, ScanAlgorithm::Fused, stream);

    gate_kernel<<<blocks_for(inner_elements), kThreads, 0, stream>>>(
        d_scan_output_, d_z_, inner_elements);
    linear_kernel<<<blocks_for(model_elements), kThreads, 0, stream>>>(
        d_scan_output_, d_out_proj_, d_output, rows, config_.inner_dim,
        config_.model_dim);
    residual_kernel<<<blocks_for(model_elements), kThreads, 0, stream>>>(
        d_input, d_output, model_elements);
    CUDA_CHECK_LAST();
}
