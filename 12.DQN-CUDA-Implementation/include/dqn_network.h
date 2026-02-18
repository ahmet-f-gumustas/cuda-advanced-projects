#ifndef DQN_NETWORK_H
#define DQN_NETWORK_H

#include <string>
#include <vector>
#include <random>

// ============================================================
// DQN Network - Deep Q-Network with Target Network
// ============================================================
// Architecture: state_size -> 128 -> 128 -> num_actions
// Activation: ReLU (hidden layers), Linear (output)
// Optimizer: Adam
// ============================================================

struct DenseLayer {
    // Device pointers - weights and biases
    float* d_weights;       // [out_features, in_features]
    float* d_bias;          // [out_features]

    // Device pointers - gradients
    float* d_grad_weights;  // [out_features, in_features]
    float* d_grad_bias;     // [out_features]

    // Adam optimizer state
    float* d_m_weights;     // First moment for weights
    float* d_v_weights;     // Second moment for weights
    float* d_m_bias;        // First moment for bias
    float* d_v_bias;        // Second moment for bias

    // Activations (cached for backward pass)
    float* d_input;         // [batch_size, in_features]
    float* d_pre_activation;// [batch_size, out_features] (before ReLU)
    float* d_output;        // [batch_size, out_features] (after ReLU)

    int in_features;
    int out_features;
};

class DQNNetwork {
public:
    DQNNetwork(int state_size, int num_actions, int batch_size,
               float learning_rate = 0.001f, float beta1 = 0.9f,
               float beta2 = 0.999f, float epsilon = 1e-8f);
    ~DQNNetwork();

    // Forward pass - returns Q-values for all actions
    // q_values: [batch_size, num_actions]
    void forward(const float* d_states, float* d_q_values);

    // Backward pass and parameter update
    void backward(const float* d_loss_grad);

    // Get best actions (argmax of Q-values)
    void get_actions(const float* d_q_values, int* d_actions);

    // Get max Q-values
    void get_max_q(const float* d_q_values, float* d_max_q);

    // Copy weights to target network
    void copy_weights_to(DQNNetwork* target);

    // Soft update: target = tau * this + (1-tau) * target
    void soft_update_to(DQNNetwork* target, float tau);

    // Save/Load model
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);

    // Getters
    int get_state_size() const { return state_size_; }
    int get_num_actions() const { return num_actions_; }
    int get_batch_size() const { return batch_size_; }

private:
    void init_layer(DenseLayer& layer, int in_features, int out_features);
    void free_layer(DenseLayer& layer);
    void zero_gradients();

    int state_size_;
    int num_actions_;
    int batch_size_;

    // Hyperparameters
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int timestep_;          // For Adam bias correction

    // Network layers: input -> fc1 -> relu -> fc2 -> relu -> fc3 (output)
    DenseLayer fc1_;        // state_size -> 128
    DenseLayer fc2_;        // 128 -> 128
    DenseLayer fc3_;        // 128 -> num_actions

    // Temporary buffers
    float* d_grad_relu1_;   // Gradient after relu1
    float* d_grad_relu2_;   // Gradient after relu2

    // Random number generator
    std::mt19937 rng_;
};

// ============================================================
// Experience Replay Buffer
// ============================================================

struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class ReplayBuffer {
public:
    ReplayBuffer(int capacity, int state_size);

    void add(const std::vector<float>& state, int action, float reward,
             const std::vector<float>& next_state, bool done);

    // Sample a batch and copy directly to device memory
    void sample_batch(int batch_size,
                      float* d_states,      // [batch_size, state_size]
                      int* d_actions,        // [batch_size]
                      float* d_rewards,      // [batch_size]
                      float* d_next_states,  // [batch_size, state_size]
                      float* d_dones);       // [batch_size]

    int size() const { return current_size_; }

private:
    int capacity_;
    int state_size_;
    int current_size_;
    int write_pos_;

    // Stored as flat arrays for efficient batch sampling
    std::vector<float> states_;         // [capacity, state_size]
    std::vector<int> actions_;          // [capacity]
    std::vector<float> rewards_;        // [capacity]
    std::vector<float> next_states_;    // [capacity, state_size]
    std::vector<float> dones_;          // [capacity]

    std::mt19937 rng_;
};

#endif // DQN_NETWORK_H
