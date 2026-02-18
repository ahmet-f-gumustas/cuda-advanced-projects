#ifndef DQN_KERNELS_CUH
#define DQN_KERNELS_CUH

#include <cuda_runtime.h>

// ============================================================
// DQN CUDA Kernels - Deep Q-Network for Reinforcement Learning
// ============================================================

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// -----------------------------------------------------------
// Forward Pass Kernels
// -----------------------------------------------------------

// Dense (fully connected) layer forward: output = input * W^T + bias
// Each thread computes one output neuron for one sample in the batch
__global__ void dense_forward_kernel(
    const float* input,     // [batch_size, in_features]
    const float* weights,   // [out_features, in_features]
    const float* bias,      // [out_features]
    float* output,          // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features
);

// ReLU activation forward
__global__ void relu_forward_kernel(
    const float* input,     // [n]
    float* output,          // [n]
    int n
);

// Argmax kernel - find action with highest Q-value per sample
__global__ void argmax_kernel(
    const float* q_values,  // [batch_size, num_actions]
    int* actions,           // [batch_size]
    int batch_size,
    int num_actions
);

// Max Q-value kernel - find max Q-value per sample (for target computation)
__global__ void max_qvalue_kernel(
    const float* q_values,  // [batch_size, num_actions]
    float* max_q,           // [batch_size]
    int batch_size,
    int num_actions
);

// -----------------------------------------------------------
// Loss and Target Computation Kernels
// -----------------------------------------------------------

// Compute TD target: target = reward + gamma * max_q_next * (1 - done)
__global__ void compute_td_target_kernel(
    const float* rewards,       // [batch_size]
    const float* max_q_next,    // [batch_size]
    const float* dones,         // [batch_size] (1.0 if done, 0.0 otherwise)
    float* td_targets,          // [batch_size]
    float gamma,
    int batch_size
);

// Gather Q-values for taken actions: q_taken[i] = q_values[i, actions[i]]
__global__ void gather_q_values_kernel(
    const float* q_values,      // [batch_size, num_actions]
    const int* actions,         // [batch_size]
    float* q_taken,             // [batch_size]
    int batch_size,
    int num_actions
);

// Compute MSE loss gradient: dq = 2 * (q_taken - td_target) / batch_size
__global__ void compute_loss_gradient_kernel(
    const float* q_taken,       // [batch_size]
    const float* td_targets,    // [batch_size]
    float* dq,                  // [batch_size]
    float* loss_out,            // [1] accumulated loss
    int batch_size
);

// Scatter gradient back to Q-value positions
__global__ void scatter_gradient_kernel(
    const float* dq,            // [batch_size]
    const int* actions,         // [batch_size]
    float* dq_values,           // [batch_size, num_actions] (zeroed)
    int batch_size,
    int num_actions
);

// -----------------------------------------------------------
// Backward Pass Kernels
// -----------------------------------------------------------

// ReLU backward
__global__ void relu_backward_kernel(
    const float* grad_output,   // [n]
    const float* input,         // [n] (pre-activation)
    float* grad_input,          // [n]
    int n
);

// Dense layer backward - compute gradients for weights, bias, and input
// Weight gradient: dW = grad_output^T * input
__global__ void dense_weight_grad_kernel(
    const float* grad_output,   // [batch_size, out_features]
    const float* input,         // [batch_size, in_features]
    float* grad_weights,        // [out_features, in_features]
    int batch_size,
    int in_features,
    int out_features
);

// Bias gradient: db = sum(grad_output, axis=0)
__global__ void dense_bias_grad_kernel(
    const float* grad_output,   // [batch_size, out_features]
    float* grad_bias,           // [out_features]
    int batch_size,
    int out_features
);

// Input gradient: dx = grad_output * W
__global__ void dense_input_grad_kernel(
    const float* grad_output,   // [batch_size, out_features]
    const float* weights,       // [out_features, in_features]
    float* grad_input,          // [batch_size, in_features]
    int batch_size,
    int in_features,
    int out_features
);

// -----------------------------------------------------------
// Optimizer Kernels
// -----------------------------------------------------------

// Adam optimizer update step
__global__ void adam_update_kernel(
    float* params,              // Parameters to update
    const float* grads,         // Gradients
    float* m,                   // First moment estimate
    float* v,                   // Second moment estimate
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float beta1_t,              // beta1^t (precomputed)
    float beta2_t,              // beta2^t (precomputed)
    int n
);

// SGD with momentum update
__global__ void sgd_momentum_update_kernel(
    float* params,
    const float* grads,
    float* velocity,
    float lr,
    float momentum,
    int n
);

// -----------------------------------------------------------
// Utility Kernels
// -----------------------------------------------------------

// Copy weights from online network to target network
__global__ void copy_weights_kernel(
    const float* src,
    float* dst,
    int n
);

// Soft update: target = tau * online + (1 - tau) * target
__global__ void soft_update_kernel(
    const float* online_params,
    float* target_params,
    float tau,
    int n
);

// Zero out a buffer
__global__ void zero_buffer_kernel(
    float* buffer,
    int n
);

// Compute MSE loss (reduction)
__global__ void mse_loss_kernel(
    const float* predictions,
    const float* targets,
    float* loss,
    int n
);

#endif // DQN_KERNELS_CUH
