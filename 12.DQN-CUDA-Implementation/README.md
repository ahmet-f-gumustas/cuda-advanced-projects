# 12. Deep Q-Network (DQN) - CUDA Implementation

A complete CUDA implementation of the Deep Q-Network (DQN) reinforcement learning algorithm, applied to the classic CartPole balancing problem. All neural network computations (forward pass, backward pass, optimization) run on the GPU via custom CUDA kernels.

## Overview

### What is DQN?

Deep Q-Network (DQN) is a reinforcement learning algorithm introduced by DeepMind that combines Q-Learning with deep neural networks. The agent learns to map states to action values (Q-values) and selects actions that maximize expected cumulative reward.

### Key DQN Components

1. **Q-Network**: Neural network that approximates Q(s, a) — the expected future reward for taking action `a` in state `s`
2. **Target Network**: A delayed copy of the Q-network used to compute stable TD targets
3. **Experience Replay**: Buffer that stores past transitions and samples random mini-batches for training
4. **Epsilon-Greedy**: Exploration strategy that balances random exploration with learned policy exploitation

### CartPole Environment

| Property | Value |
|----------|-------|
| State Space | 4D continuous: [cart_pos, cart_vel, pole_angle, pole_angular_vel] |
| Action Space | 2 discrete: push left (0), push right (1) |
| Reward | +1 per timestep the pole remains upright |
| Termination | pole angle > 12° OR cart position > 2.4 OR 500 steps |
| Solved | Average reward >= 475 over 100 consecutive episodes |

## Architecture

### Neural Network (Q-Network)

```
Input (state_size=4)
  │
  ▼
Dense Layer 1 (4 → 128) + ReLU
  │
  ▼
Dense Layer 2 (128 → 128) + ReLU
  │
  ▼
Dense Layer 3 (128 → 2) [Linear - Q-values]
  │
  ▼
Output: Q(s, left), Q(s, right)
```

### CUDA Kernel Organization

| Kernel | Purpose | Grid/Block Config |
|--------|---------|-------------------|
| `dense_forward_kernel` | Fully connected forward pass | (batch, out/block) × block |
| `relu_forward_kernel` | ReLU activation | 1D grid |
| `relu_backward_kernel` | ReLU gradient | 1D grid |
| `dense_weight_grad_kernel` | dL/dW computation | (out, in/block) × block |
| `dense_bias_grad_kernel` | dL/db computation | 1D grid |
| `dense_input_grad_kernel` | dL/dx computation | (batch, in/block) × block |
| `adam_update_kernel` | Adam optimizer step | 1D grid |
| `compute_td_target_kernel` | TD target: r + γ·max(Q') | 1D grid |
| `gather_q_values_kernel` | Q(s,a) for taken actions | 1D grid |
| `compute_loss_gradient_kernel` | Huber loss gradient | 1D grid |
| `scatter_gradient_kernel` | Scatter grad to Q-shape | 1D grid |
| `soft_update_kernel` | τ·online + (1-τ)·target | 1D grid |
| `argmax_kernel` | Best action selection | 1D grid |
| `max_qvalue_kernel` | Max Q-value per sample | 1D grid |

## Mathematical Formulation

### Bellman Equation (Q-Learning Target)

```
Q*(s, a) = E[r + γ · max_a' Q*(s', a')]
```

### TD Target

```
y_i = r_i + γ · max_a' Q_target(s'_i, a')    if not terminal
y_i = r_i                                       if terminal
```

### Loss Function (Huber Loss)

```
L = (1/N) Σ huber(Q_online(s_i, a_i) - y_i)

huber(x) = 0.5·x²           if |x| <= 1
            |x| - 0.5        otherwise
```

### Adam Optimizer Update

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

## Project Structure

```
12.DQN-CUDA-Implementation/
├── include/
│   ├── dqn_kernels.cuh     # CUDA kernel declarations
│   ├── dqn_network.h       # DQNNetwork class + ReplayBuffer
│   └── cartpole_env.h      # CartPole environment (header-only)
├── src/
│   ├── dqn_kernels.cu      # CUDA kernel implementations
│   ├── dqn_network.cu      # Network + ReplayBuffer implementation
│   ├── train.cu            # Training executable
│   └── predict.cu          # Inference/test executable
├── build/                   # Build output directory
├── CMakeLists.txt           # CMake build configuration
├── .gitignore
└── README.md
```

## Building

### Prerequisites

- CUDA Toolkit >= 11.0
- CMake >= 3.18
- C++17 compatible compiler
- NVIDIA GPU (Compute Capability >= 7.5)

### Build Commands

```bash
cd 12.DQN-CUDA-Implementation
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Training

```bash
cd build
./train
```

Training output:
```
============================================
  DQN Training - CartPole (CUDA)
============================================
GPU: NVIDIA GeForce RTX 4070
Compute Capability: 8.9

Hyperparameters:
  Batch Size:         64
  Replay Buffer:      50000
  Target Update Freq: 500 steps
  Gamma:              0.99
  Learning Rate:      0.001
  Epsilon:            1.0 -> 0.01
  Episodes:           500

Starting training...
--------------------------------------------
Episode    0 | Reward:   12.0 | Avg(100):   12.0 | Loss:   0.0142 | Eps: 0.995 | ...
Episode   10 | Reward:   23.0 | Avg(100):   18.5 | Loss:   0.0089 | Eps: 0.951 | ...
...
*** SOLVED at episode 285! ***
*** Average reward over last 100 episodes: 478.3 ***
```

The training produces:
- `dqn_model.bin` — final model weights
- `dqn_model_solved.bin` — weights when the environment was first solved

### Prediction / Inference

```bash
cd build
./predict                        # Uses dqn_model.bin
./predict dqn_model_solved.bin   # Use specific model
```

The prediction program runs 5 tests:
1. **Performance Evaluation**: 20 episodes with reward statistics
2. **Step-by-Step Visualization**: Detailed single episode walkthrough
3. **Q-Value Analysis**: Q-values for specific interesting states
4. **Inference Latency Benchmark**: GPU throughput measurement
5. **Robustness Test**: Performance under state perturbations

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 64 | Mini-batch size for training |
| `REPLAY_BUFFER_SIZE` | 50,000 | Maximum replay buffer capacity |
| `MIN_REPLAY_SIZE` | 1,000 | Minimum experiences before training starts |
| `TARGET_UPDATE_FREQ` | 500 | Steps between hard target network updates |
| `GAMMA` | 0.99 | Discount factor |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.01 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.995 | Multiplicative epsilon decay per episode |
| `BETA1` | 0.9 | Adam first moment decay |
| `BETA2` | 0.999 | Adam second moment decay |

## DQN Algorithm Pseudocode

```
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ
Initialize replay buffer D

for episode = 1 to N:
    state = env.reset()
    for t = 1 to T:
        # Epsilon-greedy action selection
        if random() < ε:
            action = random_action()
        else:
            action = argmax_a Q(state, a; θ)    ← GPU forward pass

        # Take step
        next_state, reward, done = env.step(action)
        D.add(state, action, reward, next_state, done)

        # Training step
        if |D| >= min_replay_size:
            batch = D.sample(batch_size)         ← Random mini-batch

            # Compute TD targets (on GPU)
            Q_next = Q(next_states; θ⁻)          ← Target network
            targets = rewards + γ · max(Q_next) · (1 - dones)

            # Compute loss and update (on GPU)
            Q_pred = Q(states; θ)
            loss = huber(Q_pred[actions] - targets)
            θ ← Adam(θ, ∇loss)

        # Update target network
        if steps % target_freq == 0:
            θ⁻ ← θ
```

## Memory Layout

### Weight Matrices (Row-Major)
```
weights[out_features, in_features]
  FC1: [128, 4]     = 512 floats  (2 KB)
  FC2: [128, 128]   = 16384 floats (64 KB)
  FC3: [2, 128]     = 256 floats  (1 KB)
  Total weights:      17152 floats (~67 KB)
```

### GPU Memory Usage
```
Parameters (online + target):  ~134 KB
Adam state (m, v):             ~134 KB
Activations (batch=64):       ~100 KB
Replay batch buffers:          ~10 KB
Total:                         ~378 KB
```

## References

- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- Mnih et al., "Human-level control through deep reinforcement learning" (2015, Nature)
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
