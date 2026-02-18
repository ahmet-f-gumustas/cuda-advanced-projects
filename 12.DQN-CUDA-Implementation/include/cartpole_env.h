#ifndef CARTPOLE_ENV_H
#define CARTPOLE_ENV_H

#include <vector>
#include <random>
#include <cmath>

// ============================================================
// CartPole Environment - Classic RL Benchmark
// ============================================================
// State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
// Actions: 0 = push left, 1 = push right
// Reward: +1 for every step the pole remains upright
// Done: pole angle > 12 degrees OR cart position > 2.4
// ============================================================

class CartPoleEnv {
public:
    static const int STATE_SIZE = 4;
    static const int NUM_ACTIONS = 2;
    static constexpr int MAX_STEPS = 500;

    CartPoleEnv(unsigned int seed = 42) : rng_(seed), steps_(0) {
        // Physics constants
        gravity_ = 9.8f;
        cart_mass_ = 1.0f;
        pole_mass_ = 0.1f;
        total_mass_ = cart_mass_ + pole_mass_;
        pole_half_length_ = 0.5f;
        force_magnitude_ = 10.0f;
        dt_ = 0.02f;

        // Thresholds
        x_threshold_ = 2.4f;
        theta_threshold_ = 12.0f * M_PI / 180.0f;  // 12 degrees in radians

        reset();
    }

    // Reset environment to initial state
    std::vector<float> reset() {
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        state_.resize(STATE_SIZE);
        state_[0] = dist(rng_);  // cart position
        state_[1] = dist(rng_);  // cart velocity
        state_[2] = dist(rng_);  // pole angle
        state_[3] = dist(rng_);  // pole angular velocity
        steps_ = 0;
        return state_;
    }

    // Take a step in the environment
    // Returns: (next_state, reward, done, terminated)
    struct StepResult {
        std::vector<float> state;
        float reward;
        bool done;
        bool terminated;    // True if pole fell, false if max_steps reached (truncated)
    };

    StepResult step(int action) {
        float x = state_[0];
        float x_dot = state_[1];
        float theta = state_[2];
        float theta_dot = state_[3];

        // Apply force
        float force = (action == 1) ? force_magnitude_ : -force_magnitude_;

        // Physics simulation (Euler integration)
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float temp = (force + pole_mass_ * pole_half_length_ * theta_dot * theta_dot * sin_theta) / total_mass_;
        float theta_acc = (gravity_ * sin_theta - cos_theta * temp) /
                         (pole_half_length_ * (4.0f / 3.0f - pole_mass_ * cos_theta * cos_theta / total_mass_));
        float x_acc = temp - pole_mass_ * pole_half_length_ * theta_acc * cos_theta / total_mass_;

        // Update state
        state_[0] = x + dt_ * x_dot;
        state_[1] = x_dot + dt_ * x_acc;
        state_[2] = theta + dt_ * theta_dot;
        state_[3] = theta_dot + dt_ * theta_acc;

        steps_++;

        // Separate termination (failure) from truncation (max steps)
        bool terminated = (state_[0] < -x_threshold_ || state_[0] > x_threshold_ ||
                          state_[2] < -theta_threshold_ || state_[2] > theta_threshold_);
        bool truncated = (steps_ >= MAX_STEPS);
        bool done = terminated || truncated;

        // +1 reward for every non-terminal step
        float reward = terminated ? 0.0f : 1.0f;

        return {state_, reward, done, terminated};
    }

    std::vector<float> get_state() const { return state_; }
    int get_steps() const { return steps_; }

private:
    std::vector<float> state_;
    int steps_;

    // Physics parameters
    float gravity_;
    float cart_mass_;
    float pole_mass_;
    float total_mass_;
    float pole_half_length_;
    float force_magnitude_;
    float dt_;

    // Thresholds
    float x_threshold_;
    float theta_threshold_;

    std::mt19937 rng_;
};

#endif // CARTPOLE_ENV_H
