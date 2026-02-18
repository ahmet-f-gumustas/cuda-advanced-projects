#include "../include/dqn_network.h"
#include "../include/dqn_kernels.cuh"
#include "../include/cartpole_env.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <vector>

// ============================================================
// DQN Prediction / Inference - CartPole (CUDA)
// ============================================================
// Loads a trained DQN model and runs inference on CartPole.
// All Q-value computations happen on GPU via CUDA kernels.
// ============================================================

void print_state(const std::vector<float>& state) {
    std::cout << "[pos=" << std::fixed << std::setprecision(4) << state[0]
              << ", vel=" << std::setprecision(4) << state[1]
              << ", ang=" << std::setprecision(4) << state[2]
              << ", ang_vel=" << std::setprecision(4) << state[3] << "]";
}

void print_q_values(const float* q_values, int num_actions) {
    std::cout << "Q[";
    for (int i = 0; i < num_actions; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << (i == 0 ? "Left" : "Right") << "=" << std::fixed << std::setprecision(4) << q_values[i];
    }
    std::cout << "]";
}

// ============================================================
// Test 1: Run episodes and measure performance
// ============================================================
void test_performance(DQNNetwork& net, int num_episodes) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Test 1: Performance Evaluation" << std::endl;
    std::cout << "============================================" << std::endl;

    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;

    float *d_state, *d_q_values;
    CUDA_CHECK(cudaMalloc(&d_state, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_values, NUM_ACTIONS * sizeof(float)));

    std::vector<float> episode_rewards;
    std::vector<float> episode_times;

    for (int ep = 0; ep < num_episodes; ep++) {
        CartPoleEnv env(ep + 100);  // Different seed per episode
        auto state = env.reset();
        float total_reward = 0.0f;
        bool done = false;

        auto start = std::chrono::high_resolution_clock::now();

        while (!done) {
            // GPU inference: compute Q-values
            CUDA_CHECK(cudaMemcpy(d_state, state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            net.forward(d_state, d_q_values);

            float h_q[NUM_ACTIONS];
            CUDA_CHECK(cudaMemcpy(h_q, d_q_values, NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));

            // Greedy action selection
            int action = (h_q[0] > h_q[1]) ? 0 : 1;

            auto result = env.step(action);
            state = result.state;
            done = result.done;
            total_reward += result.reward;
        }

        auto end = std::chrono::high_resolution_clock::now();
        float ep_time = std::chrono::duration<float, std::milli>(end - start).count();

        episode_rewards.push_back(total_reward);
        episode_times.push_back(ep_time);

        std::cout << "  Episode " << std::setw(3) << ep
                  << " | Reward: " << std::setw(5) << std::fixed << std::setprecision(0) << total_reward
                  << " | Time: " << std::setw(8) << std::setprecision(2) << ep_time << " ms"
                  << std::endl;
    }

    // Statistics
    float avg_reward = std::accumulate(episode_rewards.begin(), episode_rewards.end(), 0.0f) / num_episodes;
    float min_reward = *std::min_element(episode_rewards.begin(), episode_rewards.end());
    float max_reward = *std::max_element(episode_rewards.begin(), episode_rewards.end());
    float avg_time = std::accumulate(episode_times.begin(), episode_times.end(), 0.0f) / num_episodes;

    std::cout << std::endl;
    std::cout << "  Results (" << num_episodes << " episodes):" << std::endl;
    std::cout << "    Average Reward: " << std::fixed << std::setprecision(1) << avg_reward << std::endl;
    std::cout << "    Min Reward:     " << std::setprecision(1) << min_reward << std::endl;
    std::cout << "    Max Reward:     " << std::setprecision(1) << max_reward << std::endl;
    std::cout << "    Avg Time/Ep:    " << std::setprecision(2) << avg_time << " ms" << std::endl;
    std::cout << "    Solved (>=475): " << (avg_reward >= 475.0f ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    cudaFree(d_state);
    cudaFree(d_q_values);
}

// ============================================================
// Test 2: Step-by-step visualization of a single episode
// ============================================================
void test_visualization(DQNNetwork& net) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Test 2: Step-by-Step Visualization" << std::endl;
    std::cout << "============================================" << std::endl;

    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;

    float *d_state, *d_q_values;
    CUDA_CHECK(cudaMalloc(&d_state, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_values, NUM_ACTIONS * sizeof(float)));

    CartPoleEnv env(123);
    auto state = env.reset();
    bool done = false;
    int step = 0;
    float total_reward = 0.0f;

    std::cout << "  Initial state: ";
    print_state(state);
    std::cout << std::endl << std::endl;

    // Show first 20 steps and last 10 steps
    int max_display_start = 20;

    while (!done) {
        CUDA_CHECK(cudaMemcpy(d_state, state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        net.forward(d_state, d_q_values);

        float h_q[NUM_ACTIONS];
        CUDA_CHECK(cudaMemcpy(h_q, d_q_values, NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));

        int action = (h_q[0] > h_q[1]) ? 0 : 1;

        if (step < max_display_start) {
            std::cout << "  Step " << std::setw(3) << step << ": State=";
            print_state(state);
            std::cout << " | ";
            print_q_values(h_q, NUM_ACTIONS);
            std::cout << " -> " << (action == 0 ? "LEFT" : "RIGHT") << std::endl;
        } else if (step == max_display_start) {
            std::cout << "  ..." << std::endl;
        }

        auto result = env.step(action);
        state = result.state;
        done = result.done;
        total_reward += result.reward;
        step++;

        // Save last 10 steps info for display
        if (done && step > max_display_start + 10) {
            std::cout << "  (showing last few steps)" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "  Episode ended at step " << step << std::endl;
    std::cout << "  Total reward: " << total_reward << std::endl;

    // Show final state
    std::cout << "  Final state: ";
    print_state(state);
    std::cout << std::endl << std::endl;

    cudaFree(d_state);
    cudaFree(d_q_values);
}

// ============================================================
// Test 3: Q-value analysis for specific states
// ============================================================
void test_q_analysis(DQNNetwork& net) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Test 3: Q-Value Analysis" << std::endl;
    std::cout << "============================================" << std::endl;

    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;

    float *d_state, *d_q_values;
    CUDA_CHECK(cudaMalloc(&d_state, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_values, NUM_ACTIONS * sizeof(float)));

    // Test specific interesting states
    struct TestCase {
        std::string name;
        std::vector<float> state;
    };

    std::vector<TestCase> test_cases = {
        {"Centered & balanced",      {0.0f, 0.0f, 0.0f, 0.0f}},
        {"Pole tilting right",       {0.0f, 0.0f, 0.05f, 0.0f}},
        {"Pole tilting left",        {0.0f, 0.0f, -0.05f, 0.0f}},
        {"Pole falling right fast",  {0.0f, 0.0f, 0.15f, 0.5f}},
        {"Pole falling left fast",   {0.0f, 0.0f, -0.15f, -0.5f}},
        {"Cart drifting right",      {1.5f, 0.5f, 0.0f, 0.0f}},
        {"Cart drifting left",       {-1.5f, -0.5f, 0.0f, 0.0f}},
        {"Critical: near right edge", {2.0f, 0.3f, 0.1f, 0.2f}},
        {"Critical: near left edge",  {-2.0f, -0.3f, -0.1f, -0.2f}},
    };

    for (const auto& tc : test_cases) {
        CUDA_CHECK(cudaMemcpy(d_state, tc.state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        net.forward(d_state, d_q_values);

        float h_q[NUM_ACTIONS];
        CUDA_CHECK(cudaMemcpy(h_q, d_q_values, NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));

        int best_action = (h_q[0] > h_q[1]) ? 0 : 1;

        std::cout << "  " << std::left << std::setw(28) << tc.name << " | ";
        print_q_values(h_q, NUM_ACTIONS);
        std::cout << " -> " << (best_action == 0 ? "LEFT" : "RIGHT") << std::endl;
    }
    std::cout << std::endl;

    cudaFree(d_state);
    cudaFree(d_q_values);
}

// ============================================================
// Test 4: GPU Inference Latency Benchmark
// ============================================================
void test_latency(DQNNetwork& net) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Test 4: GPU Inference Latency Benchmark" << std::endl;
    std::cout << "============================================" << std::endl;

    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;
    const int NUM_ITERATIONS = 10000;

    float *d_state, *d_q_values;
    CUDA_CHECK(cudaMalloc(&d_state, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_values, NUM_ACTIONS * sizeof(float)));

    // Warm up
    std::vector<float> dummy_state = {0.0f, 0.0f, 0.0f, 0.0f};
    CUDA_CHECK(cudaMemcpy(d_state, dummy_state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < 100; i++) {
        net.forward(d_state, d_q_values);
    }

    // Benchmark: GPU forward pass only
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        net.forward(d_state, d_q_values);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float total_us = std::chrono::duration<float, std::micro>(end - start).count();
    float avg_us = total_us / NUM_ITERATIONS;

    std::cout << "  Forward pass (GPU only):" << std::endl;
    std::cout << "    Iterations:   " << NUM_ITERATIONS << std::endl;
    std::cout << "    Total time:   " << std::fixed << std::setprecision(2) << total_us / 1000.0f << " ms" << std::endl;
    std::cout << "    Avg latency:  " << std::setprecision(2) << avg_us << " us" << std::endl;
    std::cout << "    Throughput:   " << std::setprecision(0) << (1e6f / avg_us) << " inferences/sec" << std::endl;
    std::cout << std::endl;

    // Benchmark: Full pipeline (H2D + forward + D2H)
    float h_q[NUM_ACTIONS];

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpy(d_state, dummy_state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        net.forward(d_state, d_q_values);
        CUDA_CHECK(cudaMemcpy(h_q, d_q_values, NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));
    }
    end = std::chrono::high_resolution_clock::now();

    total_us = std::chrono::duration<float, std::micro>(end - start).count();
    avg_us = total_us / NUM_ITERATIONS;

    std::cout << "  Full pipeline (H2D + Forward + D2H):" << std::endl;
    std::cout << "    Iterations:   " << NUM_ITERATIONS << std::endl;
    std::cout << "    Total time:   " << std::setprecision(2) << total_us / 1000.0f << " ms" << std::endl;
    std::cout << "    Avg latency:  " << std::setprecision(2) << avg_us << " us" << std::endl;
    std::cout << "    Throughput:   " << std::setprecision(0) << (1e6f / avg_us) << " inferences/sec" << std::endl;
    std::cout << std::endl;

    cudaFree(d_state);
    cudaFree(d_q_values);
}

// ============================================================
// Test 5: Robustness test with perturbed initial states
// ============================================================
void test_robustness(DQNNetwork& net) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Test 5: Robustness - Perturbed States" << std::endl;
    std::cout << "============================================" << std::endl;

    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;
    const int NUM_TESTS = 50;

    float *d_state, *d_q_values;
    CUDA_CHECK(cudaMalloc(&d_state, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_values, NUM_ACTIONS * sizeof(float)));

    std::mt19937 rng(777);

    // Test with different perturbation levels
    std::vector<float> perturbation_levels = {0.05f, 0.10f, 0.15f, 0.20f};

    for (float perturbation : perturbation_levels) {
        std::uniform_real_distribution<float> dist(-perturbation, perturbation);

        std::vector<float> rewards;
        for (int t = 0; t < NUM_TESTS; t++) {
            CartPoleEnv env(t + 200);
            auto state = env.reset();

            // Add perturbation to initial state
            for (int i = 0; i < STATE_SIZE; i++) {
                state[i] += dist(rng);
            }

            bool done = false;
            float total_reward = 0.0f;

            while (!done) {
                CUDA_CHECK(cudaMemcpy(d_state, state.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                net.forward(d_state, d_q_values);

                float h_q[NUM_ACTIONS];
                CUDA_CHECK(cudaMemcpy(h_q, d_q_values, NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));

                int action = (h_q[0] > h_q[1]) ? 0 : 1;

                auto result = env.step(action);
                state = result.state;
                done = result.done;
                total_reward += result.reward;
            }
            rewards.push_back(total_reward);
        }

        float avg = std::accumulate(rewards.begin(), rewards.end(), 0.0f) / NUM_TESTS;
        float min_r = *std::min_element(rewards.begin(), rewards.end());
        float max_r = *std::max_element(rewards.begin(), rewards.end());

        std::cout << "  Perturbation +/- " << std::fixed << std::setprecision(2) << perturbation
                  << " | Avg: " << std::setw(6) << std::setprecision(1) << avg
                  << " | Min: " << std::setw(5) << std::setprecision(0) << min_r
                  << " | Max: " << std::setw(5) << std::setprecision(0) << max_r
                  << std::endl;
    }
    std::cout << std::endl;

    cudaFree(d_state);
    cudaFree(d_q_values);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "  DQN Prediction - CartPole (CUDA)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Check CUDA device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << std::endl;

    // Determine model file
    std::string model_file = "dqn_model.bin";
    if (argc > 1) {
        model_file = argv[1];
    }

    std::cout << "Loading model: " << model_file << std::endl;

    // Create network for inference (batch_size=1)
    const int STATE_SIZE = CartPoleEnv::STATE_SIZE;
    const int NUM_ACTIONS = CartPoleEnv::NUM_ACTIONS;
    DQNNetwork net(STATE_SIZE, NUM_ACTIONS, 1);

    net.load_weights(model_file);
    std::cout << std::endl;

    // Run all tests
    test_performance(net, 20);
    test_visualization(net);
    test_q_analysis(net);
    test_latency(net);
    test_robustness(net);

    std::cout << "============================================" << std::endl;
    std::cout << "  All tests completed!" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
