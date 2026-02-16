#ifndef LSTM_KERNELS_CUH
#define LSTM_KERNELS_CUH

#include <cuda_runtime.h>

// Activation functions
__device__ float sigmoid(float x);
__device__ float tanh_activation(float x);
__device__ float sigmoid_grad(float x);
__device__ float tanh_grad(float x);

// LSTM forward pass kernel
__global__ void lstm_forward_kernel(
    const float* x,           // Input: [batch_size, input_size]
    const float* h_prev,      // Previous hidden state: [batch_size, hidden_size]
    const float* c_prev,      // Previous cell state: [batch_size, hidden_size]
    const float* W_i,         // Input gate weights: [hidden_size, input_size + hidden_size]
    const float* W_f,         // Forget gate weights: [hidden_size, input_size + hidden_size]
    const float* W_g,         // Cell gate weights: [hidden_size, input_size + hidden_size]
    const float* W_o,         // Output gate weights: [hidden_size, input_size + hidden_size]
    const float* b_i,         // Input gate bias: [hidden_size]
    const float* b_f,         // Forget gate bias: [hidden_size]
    const float* b_g,         // Cell gate bias: [hidden_size]
    const float* b_o,         // Output gate bias: [hidden_size]
    float* i_gate,            // Output: input gate activations
    float* f_gate,            // Output: forget gate activations
    float* g_gate,            // Output: cell gate activations
    float* o_gate,            // Output: output gate activations
    float* c_next,            // Output: next cell state
    float* h_next,            // Output: next hidden state
    int batch_size,
    int input_size,
    int hidden_size
);

// LSTM backward pass kernel
__global__ void lstm_backward_kernel(
    const float* dh_next,     // Gradient from next layer
    const float* dc_next,     // Gradient of cell state
    const float* c_prev,
    const float* c_next,
    const float* i_gate,
    const float* f_gate,
    const float* g_gate,
    const float* o_gate,
    const float* h_prev,
    const float* x,
    const float* W_i,         // Weight matrices (needed for backprop)
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
);

// Matrix multiplication kernel
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

// Element-wise operations
__global__ void elementwise_add_kernel(float* a, const float* b, int size);
__global__ void elementwise_mul_kernel(float* a, const float* b, int size);
__global__ void apply_sigmoid_kernel(float* data, int size);
__global__ void apply_tanh_kernel(float* data, int size);

#endif // LSTM_KERNELS_CUH
