#include "lstm_kernels.cuh"
#include <cmath>

// Activation functions
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh_activation(float x) {
    return tanhf(x);
}

__device__ float sigmoid_grad(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__device__ float tanh_grad(float x) {
    float t = tanh_activation(x);
    return 1.0f - t * t;
}

// LSTM forward pass kernel
__global__ void lstm_forward_kernel(
    const float* x,
    const float* h_prev,
    const float* c_prev,
    const float* W_i,
    const float* W_f,
    const float* W_g,
    const float* W_o,
    const float* b_i,
    const float* b_f,
    const float* b_g,
    const float* b_o,
    float* i_gate,
    float* f_gate,
    float* g_gate,
    float* o_gate,
    float* c_next,
    float* h_next,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;

    int concat_size = input_size + hidden_size;

    // Compute gates for this hidden unit
    float i_val = b_i[hidden_idx];
    float f_val = b_f[hidden_idx];
    float g_val = b_g[hidden_idx];
    float o_val = b_o[hidden_idx];

    // Input contribution
    for (int i = 0; i < input_size; i++) {
        float x_val = x[batch_idx * input_size + i];
        i_val += W_i[hidden_idx * concat_size + i] * x_val;
        f_val += W_f[hidden_idx * concat_size + i] * x_val;
        g_val += W_g[hidden_idx * concat_size + i] * x_val;
        o_val += W_o[hidden_idx * concat_size + i] * x_val;
    }

    // Hidden state contribution
    for (int i = 0; i < hidden_size; i++) {
        float h_val = h_prev[batch_idx * hidden_size + i];
        i_val += W_i[hidden_idx * concat_size + input_size + i] * h_val;
        f_val += W_f[hidden_idx * concat_size + input_size + i] * h_val;
        g_val += W_g[hidden_idx * concat_size + input_size + i] * h_val;
        o_val += W_o[hidden_idx * concat_size + input_size + i] * h_val;
    }

    // Apply activations
    i_val = sigmoid(i_val);
    f_val = sigmoid(f_val);
    g_val = tanh_activation(g_val);
    o_val = sigmoid(o_val);

    // Store gate values
    int idx = batch_idx * hidden_size + hidden_idx;
    i_gate[idx] = i_val;
    f_gate[idx] = f_val;
    g_gate[idx] = g_val;
    o_gate[idx] = o_val;

    // Update cell state
    float c_val = f_val * c_prev[idx] + i_val * g_val;
    c_next[idx] = c_val;

    // Update hidden state
    h_next[idx] = o_val * tanh_activation(c_val);
}

// LSTM backward pass kernel
__global__ void lstm_backward_kernel(
    const float* dh_next,
    const float* dc_next,
    const float* c_prev,
    const float* c_next,
    const float* i_gate,
    const float* f_gate,
    const float* g_gate,
    const float* o_gate,
    const float* h_prev,
    const float* x,
    const float* W_i,
    const float* W_f,
    const float* W_g,
    const float* W_o,
    float* dh_prev,
    float* dc_prev,
    float* dx,
    float* dW_i,
    float* dW_f,
    float* dW_g,
    float* dW_o,
    float* db_i,
    float* db_f,
    float* db_g,
    float* db_o,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;

    int idx = batch_idx * hidden_size + hidden_idx;
    int concat_size = input_size + hidden_size;

    // Gradient through hidden state
    float dh = dh_next[idx];

    // Gradient through cell state
    float dc = dc_next[idx];
    dc += dh * o_gate[idx] * tanh_grad(c_next[idx]);

    // Gate gradients
    float di = dc * g_gate[idx] * sigmoid_grad(i_gate[idx]);
    float df = dc * c_prev[idx] * sigmoid_grad(f_gate[idx]);
    float dg = dc * i_gate[idx] * tanh_grad(g_gate[idx]);
    float do_gate = dh * tanh_activation(c_next[idx]) * sigmoid_grad(o_gate[idx]);

    // Gradient for previous cell state
    dc_prev[idx] = dc * f_gate[idx];

    // Accumulate bias gradients
    atomicAdd(&db_i[hidden_idx], di);
    atomicAdd(&db_f[hidden_idx], df);
    atomicAdd(&db_g[hidden_idx], dg);
    atomicAdd(&db_o[hidden_idx], do_gate);

    // Accumulate weight gradients and compute input/hidden gradients
    for (int i = 0; i < input_size; i++) {
        float x_val = x[batch_idx * input_size + i];

        atomicAdd(&dW_i[hidden_idx * concat_size + i], di * x_val);
        atomicAdd(&dW_f[hidden_idx * concat_size + i], df * x_val);
        atomicAdd(&dW_g[hidden_idx * concat_size + i], dg * x_val);
        atomicAdd(&dW_o[hidden_idx * concat_size + i], do_gate * x_val);

        float dx_val = 0.0f;
        dx_val += W_i[hidden_idx * concat_size + i] * di;
        dx_val += W_f[hidden_idx * concat_size + i] * df;
        dx_val += W_g[hidden_idx * concat_size + i] * dg;
        dx_val += W_o[hidden_idx * concat_size + i] * do_gate;

        atomicAdd(&dx[batch_idx * input_size + i], dx_val);
    }

    for (int i = 0; i < hidden_size; i++) {
        float h_val = h_prev[batch_idx * hidden_size + i];

        atomicAdd(&dW_i[hidden_idx * concat_size + input_size + i], di * h_val);
        atomicAdd(&dW_f[hidden_idx * concat_size + input_size + i], df * h_val);
        atomicAdd(&dW_g[hidden_idx * concat_size + input_size + i], dg * h_val);
        atomicAdd(&dW_o[hidden_idx * concat_size + input_size + i], do_gate * h_val);

        float dh_val = 0.0f;
        dh_val += W_i[hidden_idx * concat_size + input_size + i] * di;
        dh_val += W_f[hidden_idx * concat_size + input_size + i] * df;
        dh_val += W_g[hidden_idx * concat_size + input_size + i] * dg;
        dh_val += W_o[hidden_idx * concat_size + input_size + i] * do_gate;

        atomicAdd(&dh_prev[batch_idx * hidden_size + i], dh_val);
    }
}

// Matrix multiplication kernel
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Element-wise operations
__global__ void elementwise_add_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

__global__ void elementwise_mul_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }
}

__global__ void apply_sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = sigmoid(data[idx]);
    }
}

__global__ void apply_tanh_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanh_activation(data[idx]);
    }
}
