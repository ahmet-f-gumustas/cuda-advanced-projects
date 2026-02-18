#include "../include/dqn_network.h"
#include "../include/dqn_kernels.cuh"
#include "../include/cartpole_env.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <numeric>
#include <deque>

// ============================================================
// DQN Training for CartPole - CUDA Implementation
// ============================================================
// Algorithm: Deep Q-Network (DQN) with:
//   - Experience Replay
//   - Target Network (hard update)
//   - Epsilon-greedy exploration with decay
//   - Huber loss (clipped gradient)
//   - Adam optimizer
// ============================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  DQN Training - CartPole (CUDA)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // ---- Hyperparameters ----
    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;     // 4
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;   // 2
    const int BATCH_SIZE = 64;
    const int REPLAY_BUFFER_SIZE = 50000;
    const int MIN_REPLAY_SIZE = 1000;       // Start training after this many experiences
    const int TARGET_UPDATE_FREQ = 500;     // Hard update target network every N steps
    const int NUM_EPISODES = 3000;
    const float GAMMA = 0.99f;              // Discount factor
    const float LEARNING_RATE = 0.001f;
    const float EPSILON_START = 1.0f;
    const float EPSILON_END = 0.01f;
    const float EPSILON_DECAY = 0.997f;
    const int SOLVE_THRESHOLD = 475;        // CartPole is "solved" at avg reward >= 475
    const int SOLVE_WINDOW = 100;           // Over last 100 episodes

    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  Batch Size:         " << BATCH_SIZE << std::endl;
    std::cout << "  Replay Buffer:      " << REPLAY_BUFFER_SIZE << std::endl;
    std::cout << "  Target Update Freq: " << TARGET_UPDATE_FREQ << " steps" << std::endl;
    std::cout << "  Gamma:              " << GAMMA << std::endl;
    std::cout << "  Learning Rate:      " << LEARNING_RATE << std::endl;
    std::cout << "  Epsilon:            " << EPSILON_START << " -> " << EPSILON_END << std::endl;
    std::cout << "  Episodes:           " << NUM_EPISODES << std::endl;
    std::cout << std::endl;

    // ---- Initialize ----
    DQNNetwork online_net(STATE_SIZE, NUM_ACTIONS, BATCH_SIZE, LEARNING_RATE);
    DQNNetwork target_net(STATE_SIZE, NUM_ACTIONS, BATCH_SIZE, LEARNING_RATE);

    // Copy online weights to target
    online_net.copy_weights_to(&target_net);

    ReplayBuffer replay_buffer(REPLAY_BUFFER_SIZE, STATE_SIZE);
    CartPoleEnv env(42);

    // Random number generator for epsilon-greedy
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::uniform_int_distribution<int> action_dist(0, NUM_ACTIONS - 1);

    // Allocate device buffers for training
    float *d_states, *d_next_states, *d_rewards, *d_dones;
    int *d_actions;
    float *d_q_values, *d_q_next_values;
    float *d_max_q_next, *d_td_targets, *d_q_taken, *d_dq, *d_loss;
    float *d_dq_values;  // scattered gradient

    CUDA_CHECK(cudaMalloc(&d_states, BATCH_SIZE * STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_states, BATCH_SIZE * STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rewards, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dones, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_actions, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_q_values, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_next_values, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_q_next, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_td_targets, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_taken, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dq, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dq_values, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));

    // For single state inference (action selection)
    float *d_single_state, *d_single_q;
    // We use batch_size=1 network for inference would be wasteful,
    // so we'll do action selection on CPU using a small host buffer
    DQNNetwork inference_net(STATE_SIZE, NUM_ACTIONS, 1, LEARNING_RATE);

    CUDA_CHECK(cudaMalloc(&d_single_state, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_single_q, NUM_ACTIONS * sizeof(float)));

    // ---- Training Loop ----
    float epsilon = EPSILON_START;
    int total_steps = 0;
    std::deque<float> recent_rewards;
    float best_avg_reward = 0.0f;
    bool solved = false;

    auto train_start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting training..." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        auto ep_start = std::chrono::high_resolution_clock::now();

        std::vector<float> state = env.reset();
        float episode_reward = 0.0f;
        float episode_loss = 0.0f;
        int episode_updates = 0;
        bool done = false;

        // Copy current online weights to inference network
        online_net.copy_weights_to(&inference_net);

        while (!done) {
            // ---- Select Action (epsilon-greedy) ----
            int action;
            if (uniform(rng) < epsilon) {
                action = action_dist(rng);
            } else {
                // Use GPU to compute Q-values
                CUDA_CHECK(cudaMemcpy(d_single_state, state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                inference_net.forward(d_single_state, d_single_q);

                float h_q[NUM_ACTIONS];
                CUDA_CHECK(cudaMemcpy(h_q, d_single_q, NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));

                action = (h_q[0] > h_q[1]) ? 0 : 1;
            }

            // ---- Take Step ----
            auto result = env.step(action);
            done = result.done;
            episode_reward += result.reward;

            // Store experience
            replay_buffer.add(state, action, result.reward, result.state, done);
            state = result.state;
            total_steps++;

            // ---- Train (if enough experiences) ----
            if (replay_buffer.size() >= MIN_REPLAY_SIZE) {
                // Sample batch
                replay_buffer.sample_batch(BATCH_SIZE, d_states, d_actions, d_rewards, d_next_states, d_dones);

                // Compute Q-values for current states
                online_net.forward(d_states, d_q_values);

                // Compute Q-values for next states using TARGET network
                target_net.forward(d_next_states, d_q_next_values);

                // Get max Q-value for next states
                target_net.get_max_q(d_q_next_values, d_max_q_next);

                // Compute TD targets: target = reward + gamma * max_q_next * (1 - done)
                {
                    int grid = (BATCH_SIZE + 255) / 256;
                    compute_td_target_kernel<<<grid, 256>>>(
                        d_rewards, d_max_q_next, d_dones, d_td_targets, GAMMA, BATCH_SIZE
                    );
                }

                // Gather Q-values for taken actions
                {
                    int grid = (BATCH_SIZE + 255) / 256;
                    gather_q_values_kernel<<<grid, 256>>>(
                        d_q_values, d_actions, d_q_taken, BATCH_SIZE, NUM_ACTIONS
                    );
                }

                // Compute loss and gradient
                CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
                {
                    int grid = (BATCH_SIZE + 255) / 256;
                    compute_loss_gradient_kernel<<<grid, 256>>>(
                        d_q_taken, d_td_targets, d_dq, d_loss, BATCH_SIZE
                    );
                }

                // Scatter gradient to Q-value shape
                CUDA_CHECK(cudaMemset(d_dq_values, 0, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));
                {
                    int grid = (BATCH_SIZE + 255) / 256;
                    scatter_gradient_kernel<<<grid, 256>>>(
                        d_dq, d_actions, d_dq_values, BATCH_SIZE, NUM_ACTIONS
                    );
                }

                // Backward pass and parameter update
                online_net.backward(d_dq_values);

                // Get loss value
                float h_loss;
                CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
                episode_loss += h_loss;
                episode_updates++;

                // Hard update target network
                if (total_steps % TARGET_UPDATE_FREQ == 0) {
                    online_net.copy_weights_to(&target_net);
                }
            }
        }

        // Decay epsilon
        epsilon = fmaxf(EPSILON_END, epsilon * EPSILON_DECAY);

        // Track rewards
        recent_rewards.push_back(episode_reward);
        if ((int)recent_rewards.size() > SOLVE_WINDOW) {
            recent_rewards.pop_front();
        }

        float avg_reward = 0.0f;
        if (!recent_rewards.empty()) {
            avg_reward = std::accumulate(recent_rewards.begin(), recent_rewards.end(), 0.0f) / recent_rewards.size();
        }

        if (avg_reward > best_avg_reward) {
            best_avg_reward = avg_reward;
            // Save best model checkpoint
            if ((int)recent_rewards.size() >= SOLVE_WINDOW && best_avg_reward > 50.0f) {
                online_net.save_weights("dqn_model_best.bin");
            }
        }

        auto ep_end = std::chrono::high_resolution_clock::now();
        float ep_ms = std::chrono::duration<float, std::milli>(ep_end - ep_start).count();

        // Print progress
        if (episode % 10 == 0 || avg_reward >= SOLVE_THRESHOLD) {
            float avg_loss = (episode_updates > 0) ? episode_loss / episode_updates : 0.0f;
            std::cout << "Episode " << std::setw(4) << episode
                      << " | Reward: " << std::setw(6) << std::fixed << std::setprecision(1) << episode_reward
                      << " | Avg(100): " << std::setw(6) << std::setprecision(1) << avg_reward
                      << " | Loss: " << std::setw(8) << std::setprecision(4) << avg_loss
                      << " | Eps: " << std::setw(5) << std::setprecision(3) << epsilon
                      << " | Steps: " << std::setw(6) << total_steps
                      << " | Time: " << std::setw(6) << std::setprecision(1) << ep_ms << "ms"
                      << std::endl;
        }

        // Check if solved
        if ((int)recent_rewards.size() >= SOLVE_WINDOW && avg_reward >= SOLVE_THRESHOLD && !solved) {
            solved = true;
            std::cout << std::endl;
            std::cout << "*** SOLVED at episode " << episode << "! ***" << std::endl;
            std::cout << "*** Average reward over last 100 episodes: " << avg_reward << " ***" << std::endl;
            std::cout << std::endl;

            // Save the solved model
            online_net.save_weights("dqn_model_solved.bin");
        }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(train_end - train_start).count();

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "  Total Time:       " << std::fixed << std::setprecision(2) << total_time << "s" << std::endl;
    std::cout << "  Total Steps:      " << total_steps << std::endl;
    std::cout << "  Best Avg Reward:  " << std::setprecision(1) << best_avg_reward << std::endl;
    std::cout << "  Final Epsilon:    " << std::setprecision(4) << epsilon << std::endl;
    std::cout << "  Solved:           " << (solved ? "YES" : "NO") << std::endl;

    // Save final model
    online_net.save_weights("dqn_model.bin");
    std::cout << std::endl;

    // ---- Cleanup ----
    cudaFree(d_states);
    cudaFree(d_next_states);
    cudaFree(d_rewards);
    cudaFree(d_dones);
    cudaFree(d_actions);
    cudaFree(d_q_values);
    cudaFree(d_q_next_values);
    cudaFree(d_max_q_next);
    cudaFree(d_td_targets);
    cudaFree(d_q_taken);
    cudaFree(d_dq);
    cudaFree(d_loss);
    cudaFree(d_dq_values);
    cudaFree(d_single_state);
    cudaFree(d_single_q);

    return 0;
}
