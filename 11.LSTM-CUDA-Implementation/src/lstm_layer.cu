#include "lstm_layer.h"
#include "lstm_kernels.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

LSTMLayer::LSTMLayer(int input_size, int hidden_size, int batch_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      batch_size_(batch_size),
      concat_size_(input_size + hidden_size) {
    allocate_memory();
    initialize_weights();
}

LSTMLayer::~LSTMLayer() {
    free_memory();
}

void LSTMLayer::allocate_memory() {
    int weight_size = hidden_size_ * concat_size_;
    int bias_size = hidden_size_;
    int state_size = batch_size_ * hidden_size_;
    int input_size = batch_size_ * input_size_;

    // Allocate parameters
    CUDA_CHECK(cudaMalloc(&d_W_i_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_f_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_g_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_o_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_i_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_f_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_g_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_o_, bias_size * sizeof(float)));

    // Allocate states
    CUDA_CHECK(cudaMalloc(&d_h_prev_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_next_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c_prev_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c_next_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_, input_size * sizeof(float)));

    // Allocate gates
    CUDA_CHECK(cudaMalloc(&d_i_gate_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f_gate_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_gate_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o_gate_, state_size * sizeof(float)));

    // Allocate gradients
    CUDA_CHECK(cudaMalloc(&d_dW_i_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_f_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_g_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_o_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_i_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_f_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_g_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_o_, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dh_prev_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc_prev_, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_, input_size * sizeof(float)));

    // Initialize states and gradients to zero
    CUDA_CHECK(cudaMemset(d_h_prev_, 0, state_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_h_next_, 0, state_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_c_prev_, 0, state_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_c_next_, 0, state_size * sizeof(float)));
}

void LSTMLayer::free_memory() {
    // Free parameters
    cudaFree(d_W_i_); cudaFree(d_W_f_); cudaFree(d_W_g_); cudaFree(d_W_o_);
    cudaFree(d_b_i_); cudaFree(d_b_f_); cudaFree(d_b_g_); cudaFree(d_b_o_);

    // Free states
    cudaFree(d_h_prev_); cudaFree(d_h_next_);
    cudaFree(d_c_prev_); cudaFree(d_c_next_);
    cudaFree(d_x_);

    // Free gates
    cudaFree(d_i_gate_); cudaFree(d_f_gate_);
    cudaFree(d_g_gate_); cudaFree(d_o_gate_);

    // Free gradients
    cudaFree(d_dW_i_); cudaFree(d_dW_f_); cudaFree(d_dW_g_); cudaFree(d_dW_o_);
    cudaFree(d_db_i_); cudaFree(d_db_f_); cudaFree(d_db_g_); cudaFree(d_db_o_);
    cudaFree(d_dh_prev_); cudaFree(d_dc_prev_); cudaFree(d_dx_);
}

void LSTMLayer::initialize_weights() {
    int weight_size = hidden_size_ * concat_size_;
    int bias_size = hidden_size_;

    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / (concat_size_ + hidden_size_));
    std::normal_distribution<float> dist(0.0f, std_dev);

    // Initialize weights on host
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(bias_size);

    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = dist(gen);
    }

    for (int i = 0; i < bias_size; i++) {
        h_bias[i] = 0.0f;
    }

    // Forget gate bias initialized to 1 (common practice)
    for (int i = 0; i < bias_size; i++) {
        h_bias[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_b_f_, h_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_W_i_, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_f_, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_g_, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_o_, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Reset other biases to 0
    for (int i = 0; i < bias_size; i++) {
        h_bias[i] = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_b_i_, h_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_g_, h_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_o_, h_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));
}

void LSTMLayer::forward(const float* x) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_x_, x, batch_size_ * input_size_ * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(batch_size_);
    dim3 block(hidden_size_);

    lstm_forward_kernel<<<grid, block>>>(
        d_x_, d_h_prev_, d_c_prev_,
        d_W_i_, d_W_f_, d_W_g_, d_W_o_,
        d_b_i_, d_b_f_, d_b_g_, d_b_o_,
        d_i_gate_, d_f_gate_, d_g_gate_, d_o_gate_,
        d_c_next_, d_h_next_,
        batch_size_, input_size_, hidden_size_
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update states for next timestep
    CUDA_CHECK(cudaMemcpy(d_h_prev_, d_h_next_, batch_size_ * hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_prev_, d_c_next_, batch_size_ * hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice));
}

void LSTMLayer::backward(const float* dh_next, const float* dc_next) {
    // Zero out gradients
    int weight_size = hidden_size_ * concat_size_;
    int bias_size = hidden_size_;
    int input_size = batch_size_ * input_size_;

    CUDA_CHECK(cudaMemset(d_dW_i_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW_f_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW_g_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW_o_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_i_, 0, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_f_, 0, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_g_, 0, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_o_, 0, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dh_prev_, 0, batch_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dc_prev_, 0, batch_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dx_, 0, input_size * sizeof(float)));

    // Copy gradients to device
    float* d_dh_next_tmp;
    float* d_dc_next_tmp;
    CUDA_CHECK(cudaMalloc(&d_dh_next_tmp, batch_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc_next_tmp, batch_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_dh_next_tmp, dh_next, batch_size_ * hidden_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dc_next_tmp, dc_next, batch_size_ * hidden_size_ * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(batch_size_);
    dim3 block(hidden_size_);

    lstm_backward_kernel<<<grid, block>>>(
        d_dh_next_tmp, d_dc_next_tmp,
        d_c_prev_, d_c_next_,
        d_i_gate_, d_f_gate_, d_g_gate_, d_o_gate_,
        d_h_prev_, d_x_,
        d_W_i_, d_W_f_, d_W_g_, d_W_o_,
        d_dh_prev_, d_dc_prev_, d_dx_,
        d_dW_i_, d_dW_f_, d_dW_g_, d_dW_o_,
        d_db_i_, d_db_f_, d_db_g_, d_db_o_,
        batch_size_, input_size_, hidden_size_
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_dh_next_tmp);
    cudaFree(d_dc_next_tmp);
}

void LSTMLayer::update_weights(float learning_rate) {
    int weight_size = hidden_size_ * concat_size_;
    int bias_size = hidden_size_;

    // Simple SGD update on host
    std::vector<float> h_W(weight_size);
    std::vector<float> h_dW(weight_size);
    std::vector<float> h_b(bias_size);
    std::vector<float> h_db(bias_size);

    // Update input gate weights
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W_i_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dW.data(), d_dW_i_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < weight_size; i++) {
        h_W[i] -= learning_rate * h_dW[i];
    }
    CUDA_CHECK(cudaMemcpy(d_W_i_, h_W.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Update forget gate weights
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W_f_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dW.data(), d_dW_f_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < weight_size; i++) {
        h_W[i] -= learning_rate * h_dW[i];
    }
    CUDA_CHECK(cudaMemcpy(d_W_f_, h_W.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Update cell gate weights
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W_g_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dW.data(), d_dW_g_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < weight_size; i++) {
        h_W[i] -= learning_rate * h_dW[i];
    }
    CUDA_CHECK(cudaMemcpy(d_W_g_, h_W.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Update output gate weights
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W_o_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dW.data(), d_dW_o_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < weight_size; i++) {
        h_W[i] -= learning_rate * h_dW[i];
    }
    CUDA_CHECK(cudaMemcpy(d_W_o_, h_W.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Update biases similarly
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b_i_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db.data(), d_db_i_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bias_size; i++) h_b[i] -= learning_rate * h_db[i];
    CUDA_CHECK(cudaMemcpy(d_b_i_, h_b.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b_f_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db.data(), d_db_f_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bias_size; i++) h_b[i] -= learning_rate * h_db[i];
    CUDA_CHECK(cudaMemcpy(d_b_f_, h_b.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b_g_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db.data(), d_db_g_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bias_size; i++) h_b[i] -= learning_rate * h_db[i];
    CUDA_CHECK(cudaMemcpy(d_b_g_, h_b.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b_o_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db.data(), d_db_o_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bias_size; i++) h_b[i] -= learning_rate * h_db[i];
    CUDA_CHECK(cudaMemcpy(d_b_o_, h_b.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));
}

void LSTMLayer::reset_states() {
    int state_size = batch_size_ * hidden_size_;
    CUDA_CHECK(cudaMemset(d_h_prev_, 0, state_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_h_next_, 0, state_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_c_prev_, 0, state_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_c_next_, 0, state_size * sizeof(float)));
}

void LSTMLayer::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    int weight_size = hidden_size_ * concat_size_;
    int bias_size = hidden_size_;

    std::vector<float> buffer(weight_size);

    // Save weights
    CUDA_CHECK(cudaMemcpy(buffer.data(), d_W_i_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(buffer.data(), d_W_f_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(buffer.data(), d_W_g_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(buffer.data(), d_W_o_, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));

    // Save biases
    buffer.resize(bias_size);
    CUDA_CHECK(cudaMemcpy(buffer.data(), d_b_i_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(buffer.data(), d_b_f_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(buffer.data(), d_b_g_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(buffer.data(), d_b_o_, bias_size * sizeof(float), cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));

    file.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

void LSTMLayer::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    int weight_size = hidden_size_ * concat_size_;
    int bias_size = hidden_size_;

    std::vector<float> buffer(weight_size);

    // Load weights
    file.read(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_W_i_, buffer.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_W_f_, buffer.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_W_g_, buffer.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.data()), weight_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_W_o_, buffer.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Load biases
    buffer.resize(bias_size);
    file.read(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_b_i_, buffer.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_b_f_, buffer.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_b_g_, buffer.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.data()), bias_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_b_o_, buffer.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
}
