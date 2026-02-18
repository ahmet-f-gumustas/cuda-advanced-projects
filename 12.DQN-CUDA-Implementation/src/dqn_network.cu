#include "../include/dqn_network.h"
#include "../include/dqn_kernels.cuh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

// ============================================================
// DQN Network Implementation
// ============================================================

DQNNetwork::DQNNetwork(int state_size, int num_actions, int batch_size,
                       float learning_rate, float beta1, float beta2, float epsilon)
    : state_size_(state_size), num_actions_(num_actions), batch_size_(batch_size),
      learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      timestep_(0) {

    rng_.seed(42);

    // Initialize layers
    init_layer(fc1_, state_size, 128);
    init_layer(fc2_, 128, 128);
    init_layer(fc3_, 128, num_actions);

    // Allocate gradient buffers for ReLU backward
    CUDA_CHECK(cudaMalloc(&d_grad_relu1_, batch_size * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu2_, batch_size * 128 * sizeof(float)));
}

DQNNetwork::~DQNNetwork() {
    free_layer(fc1_);
    free_layer(fc2_);
    free_layer(fc3_);
    cudaFree(d_grad_relu1_);
    cudaFree(d_grad_relu2_);
}

void DQNNetwork::init_layer(DenseLayer& layer, int in_features, int out_features) {
    layer.in_features = in_features;
    layer.out_features = out_features;

    int weight_size = out_features * in_features;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&layer.d_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_grad_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_grad_bias, out_features * sizeof(float)));

    // Adam state
    CUDA_CHECK(cudaMalloc(&layer.d_m_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_v_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_m_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_v_bias, out_features * sizeof(float)));

    // Activations
    CUDA_CHECK(cudaMalloc(&layer.d_input, batch_size_ * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_pre_activation, batch_size_ * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_output, batch_size_ * out_features * sizeof(float)));

    // Initialize weights with He initialization
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features));
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(out_features, 0.0f);

    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = dist(rng_);
    }

    CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer.d_bias, h_bias.data(), out_features * sizeof(float), cudaMemcpyHostToDevice));

    // Zero Adam state
    CUDA_CHECK(cudaMemset(layer.d_m_weights, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer.d_v_weights, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer.d_m_bias, 0, out_features * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer.d_v_bias, 0, out_features * sizeof(float)));
}

void DQNNetwork::free_layer(DenseLayer& layer) {
    cudaFree(layer.d_weights);
    cudaFree(layer.d_bias);
    cudaFree(layer.d_grad_weights);
    cudaFree(layer.d_grad_bias);
    cudaFree(layer.d_m_weights);
    cudaFree(layer.d_v_weights);
    cudaFree(layer.d_m_bias);
    cudaFree(layer.d_v_bias);
    cudaFree(layer.d_input);
    cudaFree(layer.d_pre_activation);
    cudaFree(layer.d_output);
}

void DQNNetwork::zero_gradients() {
    int block = 256;

    auto zero = [&](float* buf, int n) {
        int grid = (n + block - 1) / block;
        zero_buffer_kernel<<<grid, block>>>(buf, n);
    };

    zero(fc1_.d_grad_weights, fc1_.out_features * fc1_.in_features);
    zero(fc1_.d_grad_bias, fc1_.out_features);
    zero(fc2_.d_grad_weights, fc2_.out_features * fc2_.in_features);
    zero(fc2_.d_grad_bias, fc2_.out_features);
    zero(fc3_.d_grad_weights, fc3_.out_features * fc3_.in_features);
    zero(fc3_.d_grad_bias, fc3_.out_features);
}

void DQNNetwork::forward(const float* d_states, float* d_q_values) {
    int block_size = 128;

    // Layer 1: FC1
    // Cache input for backward pass
    CUDA_CHECK(cudaMemcpy(fc1_.d_input, d_states, batch_size_ * state_size_ * sizeof(float), cudaMemcpyDeviceToDevice));

    {
        dim3 grid(batch_size_, (fc1_.out_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_forward_kernel<<<grid, block>>>(
            d_states, fc1_.d_weights, fc1_.d_bias, fc1_.d_pre_activation,
            batch_size_, fc1_.in_features, fc1_.out_features
        );
    }

    // ReLU 1
    {
        int n = batch_size_ * fc1_.out_features;
        int grid = (n + 255) / 256;
        relu_forward_kernel<<<grid, 256>>>(fc1_.d_pre_activation, fc1_.d_output, n);
    }

    // Layer 2: FC2
    CUDA_CHECK(cudaMemcpy(fc2_.d_input, fc1_.d_output, batch_size_ * 128 * sizeof(float), cudaMemcpyDeviceToDevice));

    {
        dim3 grid(batch_size_, (fc2_.out_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_forward_kernel<<<grid, block>>>(
            fc1_.d_output, fc2_.d_weights, fc2_.d_bias, fc2_.d_pre_activation,
            batch_size_, fc2_.in_features, fc2_.out_features
        );
    }

    // ReLU 2
    {
        int n = batch_size_ * fc2_.out_features;
        int grid = (n + 255) / 256;
        relu_forward_kernel<<<grid, 256>>>(fc2_.d_pre_activation, fc2_.d_output, n);
    }

    // Layer 3: FC3 (output, no activation)
    CUDA_CHECK(cudaMemcpy(fc3_.d_input, fc2_.d_output, batch_size_ * 128 * sizeof(float), cudaMemcpyDeviceToDevice));

    {
        dim3 grid(batch_size_, (fc3_.out_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_forward_kernel<<<grid, block>>>(
            fc2_.d_output, fc3_.d_weights, fc3_.d_bias, d_q_values,
            batch_size_, fc3_.in_features, fc3_.out_features
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void DQNNetwork::backward(const float* d_loss_grad) {
    // d_loss_grad: [batch_size, num_actions]
    zero_gradients();

    int block_size = 128;

    // ---- Layer 3 (output) backward ----
    // Weight gradient
    {
        dim3 grid(fc3_.out_features, (fc3_.in_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_weight_grad_kernel<<<grid, block>>>(
            d_loss_grad, fc3_.d_input, fc3_.d_grad_weights,
            batch_size_, fc3_.in_features, fc3_.out_features
        );
    }
    // Bias gradient
    {
        int grid = (fc3_.out_features + 255) / 256;
        dense_bias_grad_kernel<<<grid, 256>>>(
            d_loss_grad, fc3_.d_grad_bias, batch_size_, fc3_.out_features
        );
    }
    // Input gradient -> d_grad_relu2_
    {
        dim3 grid(batch_size_, (fc3_.in_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_input_grad_kernel<<<grid, block>>>(
            d_loss_grad, fc3_.d_weights, d_grad_relu2_,
            batch_size_, fc3_.in_features, fc3_.out_features
        );
    }

    // ---- ReLU 2 backward ----
    {
        int n = batch_size_ * fc2_.out_features;
        int grid = (n + 255) / 256;
        relu_backward_kernel<<<grid, 256>>>(
            d_grad_relu2_, fc2_.d_pre_activation, d_grad_relu2_, n
        );
    }

    // ---- Layer 2 backward ----
    {
        dim3 grid(fc2_.out_features, (fc2_.in_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_weight_grad_kernel<<<grid, block>>>(
            d_grad_relu2_, fc2_.d_input, fc2_.d_grad_weights,
            batch_size_, fc2_.in_features, fc2_.out_features
        );
    }
    {
        int grid = (fc2_.out_features + 255) / 256;
        dense_bias_grad_kernel<<<grid, 256>>>(
            d_grad_relu2_, fc2_.d_grad_bias, batch_size_, fc2_.out_features
        );
    }
    {
        dim3 grid(batch_size_, (fc2_.in_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_input_grad_kernel<<<grid, block>>>(
            d_grad_relu2_, fc2_.d_weights, d_grad_relu1_,
            batch_size_, fc2_.in_features, fc2_.out_features
        );
    }

    // ---- ReLU 1 backward ----
    {
        int n = batch_size_ * fc1_.out_features;
        int grid = (n + 255) / 256;
        relu_backward_kernel<<<grid, 256>>>(
            d_grad_relu1_, fc1_.d_pre_activation, d_grad_relu1_, n
        );
    }

    // ---- Layer 1 backward ----
    {
        dim3 grid(fc1_.out_features, (fc1_.in_features + block_size - 1) / block_size);
        dim3 block(block_size);
        dense_weight_grad_kernel<<<grid, block>>>(
            d_grad_relu1_, fc1_.d_input, fc1_.d_grad_weights,
            batch_size_, fc1_.in_features, fc1_.out_features
        );
    }
    {
        int grid = (fc1_.out_features + 255) / 256;
        dense_bias_grad_kernel<<<grid, 256>>>(
            d_grad_relu1_, fc1_.d_grad_bias, batch_size_, fc1_.out_features
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Adam update ----
    timestep_++;
    float beta1_t = powf(beta1_, (float)timestep_);
    float beta2_t = powf(beta2_, (float)timestep_);

    auto adam_update = [&](DenseLayer& layer) {
        int w_size = layer.out_features * layer.in_features;
        int b_size = layer.out_features;
        int block = 256;

        // Update weights
        {
            int grid = (w_size + block - 1) / block;
            adam_update_kernel<<<grid, block>>>(
                layer.d_weights, layer.d_grad_weights,
                layer.d_m_weights, layer.d_v_weights,
                learning_rate_, beta1_, beta2_, epsilon_,
                beta1_t, beta2_t, w_size
            );
        }
        // Update bias
        {
            int grid = (b_size + block - 1) / block;
            adam_update_kernel<<<grid, block>>>(
                layer.d_bias, layer.d_grad_bias,
                layer.d_m_bias, layer.d_v_bias,
                learning_rate_, beta1_, beta2_, epsilon_,
                beta1_t, beta2_t, b_size
            );
        }
    };

    adam_update(fc1_);
    adam_update(fc2_);
    adam_update(fc3_);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void DQNNetwork::get_actions(const float* d_q_values, int* d_actions) {
    int grid = (batch_size_ + 255) / 256;
    argmax_kernel<<<grid, 256>>>(d_q_values, d_actions, batch_size_, num_actions_);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DQNNetwork::get_max_q(const float* d_q_values, float* d_max_q) {
    int grid = (batch_size_ + 255) / 256;
    max_qvalue_kernel<<<grid, 256>>>(d_q_values, d_max_q, batch_size_, num_actions_);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DQNNetwork::copy_weights_to(DQNNetwork* target) {
    int block = 256;

    auto copy_layer = [&](DenseLayer& src, DenseLayer& dst) {
        int w_size = src.out_features * src.in_features;
        int b_size = src.out_features;

        int grid_w = (w_size + block - 1) / block;
        copy_weights_kernel<<<grid_w, block>>>(src.d_weights, dst.d_weights, w_size);

        int grid_b = (b_size + block - 1) / block;
        copy_weights_kernel<<<grid_b, block>>>(src.d_bias, dst.d_bias, b_size);
    };

    copy_layer(fc1_, target->fc1_);
    copy_layer(fc2_, target->fc2_);
    copy_layer(fc3_, target->fc3_);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void DQNNetwork::soft_update_to(DQNNetwork* target, float tau) {
    int block = 256;

    auto update_layer = [&](DenseLayer& online, DenseLayer& tgt) {
        int w_size = online.out_features * online.in_features;
        int b_size = online.out_features;

        int grid_w = (w_size + block - 1) / block;
        soft_update_kernel<<<grid_w, block>>>(online.d_weights, tgt.d_weights, tau, w_size);

        int grid_b = (b_size + block - 1) / block;
        soft_update_kernel<<<grid_b, block>>>(online.d_bias, tgt.d_bias, tau, b_size);
    };

    update_layer(fc1_, target->fc1_);
    update_layer(fc2_, target->fc2_);
    update_layer(fc3_, target->fc3_);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void DQNNetwork::save_weights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&state_size_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_actions_), sizeof(int));

    auto save_layer = [&](DenseLayer& layer) {
        int w_size = layer.out_features * layer.in_features;
        int b_size = layer.out_features;

        std::vector<float> h_weights(w_size);
        std::vector<float> h_bias(b_size);

        CUDA_CHECK(cudaMemcpy(h_weights.data(), layer.d_weights, w_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bias.data(), layer.d_bias, b_size * sizeof(float), cudaMemcpyDeviceToHost));

        file.write(reinterpret_cast<const char*>(h_weights.data()), w_size * sizeof(float));
        file.write(reinterpret_cast<const char*>(h_bias.data()), b_size * sizeof(float));
    };

    save_layer(fc1_);
    save_layer(fc2_);
    save_layer(fc3_);

    file.close();
    std::cout << "Model saved to: " << filename << std::endl;
}

void DQNNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
        return;
    }

    int saved_state_size, saved_num_actions;
    file.read(reinterpret_cast<char*>(&saved_state_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_num_actions), sizeof(int));

    if (saved_state_size != state_size_ || saved_num_actions != num_actions_) {
        std::cerr << "Error: Model architecture mismatch!" << std::endl;
        std::cerr << "  Expected: state_size=" << state_size_ << ", num_actions=" << num_actions_ << std::endl;
        std::cerr << "  Got: state_size=" << saved_state_size << ", num_actions=" << saved_num_actions << std::endl;
        return;
    }

    auto load_layer = [&](DenseLayer& layer) {
        int w_size = layer.out_features * layer.in_features;
        int b_size = layer.out_features;

        std::vector<float> h_weights(w_size);
        std::vector<float> h_bias(b_size);

        file.read(reinterpret_cast<char*>(h_weights.data()), w_size * sizeof(float));
        file.read(reinterpret_cast<char*>(h_bias.data()), b_size * sizeof(float));

        CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layer.d_bias, h_bias.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));
    };

    load_layer(fc1_);
    load_layer(fc2_);
    load_layer(fc3_);

    file.close();
    std::cout << "Model loaded from: " << filename << std::endl;
}

// ============================================================
// Replay Buffer Implementation
// ============================================================

ReplayBuffer::ReplayBuffer(int capacity, int state_size)
    : capacity_(capacity), state_size_(state_size), current_size_(0), write_pos_(0) {

    rng_.seed(42);

    states_.resize(capacity * state_size);
    actions_.resize(capacity);
    rewards_.resize(capacity);
    next_states_.resize(capacity * state_size);
    dones_.resize(capacity);
}

void ReplayBuffer::add(const std::vector<float>& state, int action, float reward,
                       const std::vector<float>& next_state, bool done) {
    int offset = write_pos_ * state_size_;

    std::memcpy(&states_[offset], state.data(), state_size_ * sizeof(float));
    actions_[write_pos_] = action;
    rewards_[write_pos_] = reward;
    std::memcpy(&next_states_[offset], next_state.data(), state_size_ * sizeof(float));
    dones_[write_pos_] = done ? 1.0f : 0.0f;

    write_pos_ = (write_pos_ + 1) % capacity_;
    if (current_size_ < capacity_) current_size_++;
}

void ReplayBuffer::sample_batch(int batch_size,
                                float* d_states,
                                int* d_actions,
                                float* d_rewards,
                                float* d_next_states,
                                float* d_dones) {
    // Generate random indices
    std::vector<int> indices(batch_size);
    std::uniform_int_distribution<int> dist(0, current_size_ - 1);
    for (int i = 0; i < batch_size; i++) {
        indices[i] = dist(rng_);
    }

    // Gather batch data on host
    std::vector<float> h_states(batch_size * state_size_);
    std::vector<int> h_actions(batch_size);
    std::vector<float> h_rewards(batch_size);
    std::vector<float> h_next_states(batch_size * state_size_);
    std::vector<float> h_dones(batch_size);

    for (int i = 0; i < batch_size; i++) {
        int idx = indices[i];
        std::memcpy(&h_states[i * state_size_], &states_[idx * state_size_], state_size_ * sizeof(float));
        h_actions[i] = actions_[idx];
        h_rewards[i] = rewards_[idx];
        std::memcpy(&h_next_states[i * state_size_], &next_states_[idx * state_size_], state_size_ * sizeof(float));
        h_dones[i] = dones_[idx];
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), batch_size * state_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_actions, h_actions.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rewards, h_rewards.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_states, h_next_states.data(), batch_size * state_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dones, h_dones.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));
}
