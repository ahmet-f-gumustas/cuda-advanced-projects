#include "../include/dqn_kernels.cuh"
#include <stdio.h>

// ============================================================
// Forward Pass Kernels
// ============================================================

__global__ void dense_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || out_idx >= out_features) return;

    float sum = bias[out_idx];
    for (int i = 0; i < in_features; i++) {
        sum += input[batch_idx * in_features + i] * weights[out_idx * in_features + i];
    }
    output[batch_idx * out_features + out_idx] = sum;
}

__global__ void relu_forward_kernel(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void argmax_kernel(
    const float* q_values,
    int* actions,
    int batch_size,
    int num_actions
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    float max_val = q_values[batch_idx * num_actions];
    int max_idx = 0;
    for (int a = 1; a < num_actions; a++) {
        float val = q_values[batch_idx * num_actions + a];
        if (val > max_val) {
            max_val = val;
            max_idx = a;
        }
    }
    actions[batch_idx] = max_idx;
}

__global__ void max_qvalue_kernel(
    const float* q_values,
    float* max_q,
    int batch_size,
    int num_actions
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    float max_val = q_values[batch_idx * num_actions];
    for (int a = 1; a < num_actions; a++) {
        float val = q_values[batch_idx * num_actions + a];
        if (val > max_val) {
            max_val = val;
        }
    }
    max_q[batch_idx] = max_val;
}

// ============================================================
// Loss and Target Computation Kernels
// ============================================================

__global__ void compute_td_target_kernel(
    const float* rewards,
    const float* max_q_next,
    const float* dones,
    float* td_targets,
    float gamma,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    td_targets[idx] = rewards[idx] + gamma * max_q_next[idx] * (1.0f - dones[idx]);
}

__global__ void gather_q_values_kernel(
    const float* q_values,
    const int* actions,
    float* q_taken,
    int batch_size,
    int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    q_taken[idx] = q_values[idx * num_actions + actions[idx]];
}

__global__ void compute_loss_gradient_kernel(
    const float* q_taken,
    const float* td_targets,
    float* dq,
    float* loss_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float diff = q_taken[idx] - td_targets[idx];
    // Huber loss gradient (clip to [-1, 1] for stability)
    float grad = fmaxf(-1.0f, fminf(1.0f, diff));
    dq[idx] = grad / (float)batch_size;

    // Accumulate Huber loss for monitoring
    float abs_diff = fabsf(diff);
    float loss = (abs_diff <= 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    atomicAdd(loss_out, loss / (float)batch_size);
}

__global__ void scatter_gradient_kernel(
    const float* dq,
    const int* actions,
    float* dq_values,
    int batch_size,
    int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // dq_values is already zeroed
    dq_values[idx * num_actions + actions[idx]] = dq[idx];
}

// ============================================================
// Backward Pass Kernels
// ============================================================

__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
}

__global__ void dense_weight_grad_kernel(
    const float* grad_output,
    const float* input,
    float* grad_weights,
    int batch_size,
    int in_features,
    int out_features
) {
    int out_idx = blockIdx.x;
    int in_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (out_idx >= out_features || in_idx >= in_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += grad_output[b * out_features + out_idx] * input[b * in_features + in_idx];
    }
    grad_weights[out_idx * in_features + in_idx] = sum;
}

__global__ void dense_bias_grad_kernel(
    const float* grad_output,
    float* grad_bias,
    int batch_size,
    int out_features
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += grad_output[b * out_features + out_idx];
    }
    grad_bias[out_idx] = sum;
}

__global__ void dense_input_grad_kernel(
    const float* grad_output,
    const float* weights,
    float* grad_input,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int in_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || in_idx >= in_features) return;

    float sum = 0.0f;
    for (int o = 0; o < out_features; o++) {
        sum += grad_output[batch_idx * out_features + o] * weights[o * in_features + in_idx];
    }
    grad_input[batch_idx * in_features + in_idx] = sum;
}

// ============================================================
// Optimizer Kernels
// ============================================================

__global__ void adam_update_kernel(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float beta1_t,
    float beta2_t,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grads[idx];

    // Update biased first moment estimate
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    // Update biased second moment estimate
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    // Compute bias-corrected estimates
    float m_hat = m[idx] / (1.0f - beta1_t);
    float v_hat = v[idx] / (1.0f - beta2_t);

    // Update parameters
    params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

__global__ void sgd_momentum_update_kernel(
    float* params,
    const float* grads,
    float* velocity,
    float lr,
    float momentum,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    velocity[idx] = momentum * velocity[idx] + grads[idx];
    params[idx] -= lr * velocity[idx];
}

// ============================================================
// Utility Kernels
// ============================================================

__global__ void copy_weights_kernel(
    const float* src,
    float* dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx];
}

__global__ void soft_update_kernel(
    const float* online_params,
    float* target_params,
    float tau,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    target_params[idx] = tau * online_params[idx] + (1.0f - tau) * target_params[idx];
}

__global__ void zero_buffer_kernel(
    float* buffer,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buffer[idx] = 0.0f;
}

__global__ void mse_loss_kernel(
    const float* predictions,
    const float* targets,
    float* loss,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float diff = predictions[idx] - targets[idx];
    atomicAdd(loss, diff * diff / (float)n);
}
