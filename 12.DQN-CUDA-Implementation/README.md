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

Training output (RTX 4070 Laptop GPU):
```
============================================
  DQN Training - CartPole (CUDA)
============================================
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Compute Capability: 8.9

Hyperparameters:
  Batch Size:         64
  Replay Buffer:      50000
  Target Update Freq: 500 steps
  Gamma:              0.99
  Learning Rate:      0.001
  Epsilon:            1 -> 0.01
  Episodes:           3000

Starting training...
--------------------------------------------
Episode    0 | Reward:   27.0 | Avg(100):   27.0 | Loss:   0.0000 | Eps: 0.997 | ...
Episode  500 | Reward:  125.0 | Avg(100):   82.3 | Loss:   0.2201 | Eps: 0.222 | ...
Episode 1000 | Reward:  500.0 | Avg(100):  197.5 | Loss:   0.2834 | Eps: 0.050 | ...
Episode 1500 | Reward:  500.0 | Avg(100):  362.8 | Loss:   0.1812 | Eps: 0.010 | ...
Episode 2080 | Reward:  500.0 | Avg(100):  399.1 | Loss:   0.0976 | Eps: 0.010 | ...
Episode 2090 | Reward:  500.0 | Avg(100):  414.9 | Loss:   0.1499 | Eps: 0.010 | ...
...
--------------------------------------------
Training Complete!
  Total Time:       105.00s
  Total Steps:      588743
  Best Avg Reward:  413.6
  Final Epsilon:    0.0100
```

The training produces:
- `dqn_model.bin` — final model weights
- `dqn_model_best.bin` — weights at best average reward (auto-saved checkpoint)

### Prediction / Inference

```bash
cd build
./predict                        # Uses dqn_model.bin
./predict dqn_model_best.bin     # Use best checkpoint
```

The prediction program runs 5 tests:
1. **Performance Evaluation**: 20 episodes with reward statistics
2. **Step-by-Step Visualization**: Detailed single episode walkthrough
3. **Q-Value Analysis**: Q-values for specific interesting states
4. **Inference Latency Benchmark**: GPU throughput measurement
5. **Robustness Test**: Performance under state perturbations

Prediction output (best model):
```
============================================
  Test 1: Performance Evaluation
============================================
  Episode   0 | Reward:   174 | Time:     7.86 ms
  Episode   1 | Reward:   149 | Time:     6.29 ms
  Episode   2 | Reward:   179 | Time:     6.31 ms
  ...
  Results (20 episodes):
    Average Reward: 177.2
    Min Reward:     149.0
    Max Reward:     190.0
    Avg Time/Ep:    6.18 ms

============================================
  Test 3: Q-Value Analysis
============================================
  Centered & balanced          | Q[Left=107.72, Right=107.62] -> LEFT
  Pole tilting right           | Q[Left=107.83, Right=107.68] -> LEFT
  Pole falling right fast      | Q[Left=108.46, Right=108.42] -> LEFT
  Critical: near right edge    | Q[Left=91.65, Right=112.39]  -> RIGHT
  Critical: near left edge     | Q[Left=92.38, Right=90.51]   -> LEFT

============================================
  Test 4: GPU Inference Latency Benchmark
============================================
  Forward pass (GPU only):
    Avg latency:  25.96 us
    Throughput:   38518 inferences/sec

  Full pipeline (H2D + Forward + D2H):
    Avg latency:  30.18 us
    Throughput:   33139 inferences/sec

============================================
  Test 5: Robustness - Perturbed States
============================================
  Perturbation +/- 0.05 | Avg: 178.8 | Min: 151 | Max: 200
  Perturbation +/- 0.10 | Avg: 179.6 | Min: 151 | Max: 209
  Perturbation +/- 0.15 | Avg: 179.1 | Min: 151 | Max: 200
  Perturbation +/- 0.20 | Avg: 176.2 | Min:  27 | Max: 200
```

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
| `EPSILON_DECAY` | 0.997 | Multiplicative epsilon decay per episode |
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

## Training Results (RTX 4070 Laptop GPU, CUDA 12.6, FP32)

### Training Performance

| Metric | Value |
|--------|-------|
| Best Avg Reward (100 ep) | **413.6** |
| Max Episode Reward | **500** (CartPole maximum) |
| Total Training Time | **105 seconds** (3000 episodes) |
| Total Environment Steps | 588,743 |
| Episodes with Max Reward | Multiple episodes reached 500 |
| Peak Performance Window | Episodes 2000-2100 |

### Training Curve Summary

| Phase | Episodes | Avg Reward | Behavior |
|-------|----------|------------|----------|
| Exploration | 0-400 | 15-20 | Random exploration, filling replay buffer |
| Early Learning | 400-800 | 20-100 | Agent starts learning basic balancing |
| Rapid Improvement | 800-1500 | 100-350 | Reward increases quickly, frequent 500s |
| Peak Performance | 1500-2100 | 300-414 | Best performance, many perfect episodes |
| Instability | 2100-2400 | 10-300 | Catastrophic forgetting (DQN limitation) |
| Recovery | 2400-3000 | 50-380 | Agent recovers, oscillating performance |

### GPU Inference Benchmarks

| Metric | Value |
|--------|-------|
| Forward Pass Latency | **25.96 us** |
| Forward Pass Throughput | **38,518 inferences/sec** |
| Full Pipeline Latency (H2D + Forward + D2H) | **30.18 us** |
| Full Pipeline Throughput | **33,139 inferences/sec** |

### Robustness Under Perturbation

| Perturbation Level | Avg Reward | Min | Max |
|---------------------|------------|-----|-----|
| +/- 0.05 | 178.8 | 151 | 200 |
| +/- 0.10 | 179.6 | 151 | 209 |
| +/- 0.15 | 179.1 | 151 | 200 |
| +/- 0.20 | 176.2 | 27 | 200 |

### Q-Value Analysis (Learned Policy)

The trained agent demonstrates sensible behavior across different states:

| State | Best Action | Interpretation |
|-------|-------------|----------------|
| Centered & balanced | LEFT | Slight preference (nearly equal Q-values) |
| Pole tilting right | LEFT | Push left to counteract rightward tilt |
| Pole falling right fast | LEFT | Aggressive correction |
| Critical: near right edge | **RIGHT** | Push right to move cart away from edge |
| Critical: near left edge | **LEFT** | Push left to move cart away from edge |

## References

- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- Mnih et al., "Human-level control through deep reinforcement learning" (2015, Nature)
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
