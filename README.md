<div align="center">

# CUDA Advanced Projects

**Production-grade GPU computing: from custom inference engines to real-time simulations**

[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-00599C?style=flat-square&logo=cplusplus)](https://en.cppreference.com/)
[![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?style=flat-square&logo=cmake)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?style=flat-square&logo=linux&logoColor=black)](https://www.kernel.org/)

A collection of **14 CUDA projects** covering deep learning inference, computer vision, scientific computing, reinforcement learning, and generative AI — each built from scratch with custom kernels, optimized memory access patterns, and real benchmark data.

[Getting Started](#getting-started) · [Projects](#projects) · [Benchmarks](#performance-benchmarks) · [Learning Path](#learning-path)

</div>

---

## Overview

This repository demonstrates GPU programming techniques at increasing levels of complexity. Every project compiles and runs independently with CMake and targets NVIDIA GPUs from Turing (SM 7.5) through Ada Lovelace (SM 8.9).

**What makes this collection different:**

- **Custom kernels throughout** — no black-box library calls where a hand-written kernel can do better
- **Real benchmarks** — every number was measured on physical hardware (RTX 4070 Laptop / RTX 4090)
- **Production patterns** — Tensor Cores, KV caches, INT8 quantization, cuDNN/cuBLAS integration, CUDA Graphs
- **Self-contained builds** — each project has its own CMakeLists.txt; no monorepo build system to fight

---

## Projects

### Deep Learning & Inference

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 01 | [Deep Learning Inference Engine](01.deeplearning_inference_engine/) | Multi-model inference framework with INT8/FP16 quantization and layer fusion | cuDNN, cuBLAS, Tensor Cores |
| 03 | [Vision Transformer (ViT-B/16)](03.vit-B16_CUDA_Cpp/) | Pure CUDA C++20 ViT with FP16 Tensor Core GEMM — zero Python dependency | cuBLASLt, Warp Primitives |
| 04 | [EfficientNet-B0 Multi-Backend](04.EfficientNet-B0_CUDA_/) | ONNX Runtime + TensorRT + cuDNN backends; x86_64 & Jetson support | TensorRT, ONNX Runtime, NVJPEG |
| 09 | [EfficientNet TensorRT Optimization](09.EfficientNet-TensorRT-Optimization/) | TensorRT engine builder with FP16/INT8 calibration and fused CUDA preprocessing | TensorRT, INT8 Calibration |
| 08 | [Simple CUDA CNN](08.simple-cuda-cnn/) | Minimal Conv2D + ReLU + Pool with Python ctypes bindings | Python ctypes, NumPy |

### Sequence Models & Generative AI

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 11 | [LSTM CUDA Implementation](11.LSTM-CUDA-Implementation/) | Full LSTM with BPTT, gradient clipping, Xavier init — all custom kernels | BPTT, SGD, Binary I/O |
| 13 | [GPT Transformer Inference](13.GPT-Transformer-Inference/) | GPT decoder with KV Cache, GQA, RoPE, SwiGLU, INT8, speculative decoding | cuBLAS, GQA, Speculative Decoding |
| 14 | [Stable Diffusion CUDA](14.Stable-Diffusion-CUDA/) | Full pipeline: CLIP encoder + UNet + VAE decoder + DDIM scheduler | cuDNN, cuBLAS, cuRAND |

### Computer Vision & Video Processing

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 02 | [Real-Time Image Denoising](02.Real-Time_Image_Denoising_Engine/) | Bilateral, Non-Local Means, Adaptive denoising at 60 FPS / 1080p | OpenCV, PyQt5, Shared Memory |
| 10 | [Real-Time Video Stabilization](10.Real-Time_Video_Stabilization/) | Optical flow, Harris corners, RANSAC, Kalman smoothing on GPU | OpenCV, Gaussian Pyramids |

### Scientific Computing & Simulation

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 05 | [cuFFT Tutorial](05.cuFFt_/) | 1D/2D FFT, batch processing, FFT-based convolution, multi-GPU | cuFFT, CUDA Streams |
| 06 | [GPU N-Body Simulation](06.GPU_NBody_Simulation/) | Gravity sim with 3 kernel tiers + real-time OpenGL 3D visualization | OpenGL 3.3, GLFW, Register Tiling |
| 07 | [Parallel ML Models](07-parallel-ml-models/) | Linear Regression + K-Means trained in parallel with OpenGL visualization | Multi-Stream, Reduction Patterns |

### Reinforcement Learning

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 12 | [DQN CUDA Implementation](12.DQN-CUDA-Implementation/) | Deep Q-Network on CartPole — 13 custom kernels, Adam optimizer on GPU | Experience Replay, Huber Loss |

---

## Performance Benchmarks

All measurements on **RTX 4070 Laptop** (SM 8.9, 7836 MB VRAM) unless noted otherwise.

### Inference Latency

| Model | Precision | Latency | Throughput | Project |
|-------|-----------|---------|------------|---------|
| ResNet-50 | INT8 | 0.4 ms | 2500 img/s | 01 |
| ViT-B/16 | FP16 | 5.2 ms | 192 img/s | 03 |
| EfficientNet-B0 | INT8 | 0.5 ms | 2000 img/s | 09 |
| EfficientNet-B0 | FP16 | 0.8 ms | 1250 img/s | 09 |
| GPT (6L-512D) | FP16 | 0.26 ms/tok | 3842 tok/s | 13 |
| GPT (6L-512D) | INT8 | 0.22 ms/tok | 4514 tok/s | 13 |

### Real-Time Processing

| Task | Resolution | FPS | Project |
|------|------------|-----|---------|
| Image Denoising (Bilateral) | 1920x1080 | 45-60 | 02 |
| Image Denoising (Bilateral) | 3840x2160 | 12-15 | 02 |
| Video Stabilization | 1920x1080 | 45-60 | 10 |
| Video Stabilization | 1280x720 | 90-120 | 10 |

### Scientific Computing

| Workload | Size | Time | Project |
|----------|------|------|---------|
| N-Body (Register-Tiled) | 16,384 bodies | 30.5 ms | 06 |
| N-Body (Register-Tiled) | 1,024 bodies | 0.18 ms | 06 |
| FFT 2D | 2048x2048 | ~1 ms | 05 |

### Reinforcement Learning

| Metric | Value | Project |
|--------|-------|---------|
| CartPole avg reward (100 ep) | 413.6 / 500 | 12 |
| Training time (3000 episodes) | 105 s | 12 |
| GPU inference latency | 25.96 us | 12 |

---

## Technical Highlights

<table>
<tr>
<td width="33%" valign="top">

**Memory Optimization**
- Shared memory tiling
- Register tiling
- Coalesced access patterns
- Memory pooling
- Pinned (page-locked) memory

</td>
<td width="33%" valign="top">

**Compute Optimization**
- FP16/INT8 Tensor Cores
- Warp-level primitives
- Kernel fusion
- CUDA Graphs
- Loop unrolling & ILP

</td>
<td width="33%" valign="top">

**Architecture Patterns**
- KV Cache (GPT)
- Grouped Query Attention
- Speculative decoding
- Classifier-free guidance
- Experience replay buffers

</td>
</tr>
</table>

**Libraries used across projects:** cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, TensorRT, ONNX Runtime, NVJPEG, OpenGL, OpenCV

---

## Getting Started

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | Compute Capability 7.5 (Turing) | CC 8.9 (Ada Lovelace) |
| VRAM | 4 GB | 8 GB+ |
| CUDA Toolkit | 11.0 | 12.6 |
| CMake | 3.18 | 3.28+ |
| C++ Compiler | GCC 9 / Clang 10 / MSVC 2019 | GCC 12+ |
| OS | Ubuntu 20.04 | Ubuntu 22.04/24.04 |

### Build Any Project

```bash
git clone https://github.com/ahmet-f-gumustas/cuda-advanced-projects.git
cd cuda-advanced-projects

# Pick a project
cd 13.GPT-Transformer-Inference
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
./test_transformer

# Run the application
./gpt_inference --prompt "The future of AI"
```

### Verify Your Setup

```bash
nvidia-smi                                          # Driver & GPU info
nvcc --version                                      # CUDA compiler version
nvidia-smi --query-gpu=compute_cap --format=csv     # Compute capability
```

---

## Learning Path

A suggested order from fundamentals to production-grade systems:

```
Level 1: Foundations                Level 2: Optimization              Level 3: Production Systems
─────────────────────              ──────────────────────              ───────────────────────────
08. Simple CUDA CNN                02. Image Denoising                 03. ViT-B/16 CUDA
05. cuFFT Tutorial                 06. N-Body Simulation               04. EfficientNet Multi-Backend
07. Parallel ML Models             10. Video Stabilization             09. EfficientNet TensorRT
                                   11. LSTM Implementation             01. DL Inference Engine
                                   12. DQN Implementation              13. GPT Transformer
                                                                       14. Stable Diffusion
```

| Level | Focus | You will learn |
|-------|-------|----------------|
| **1 — Foundations** | Basic kernels, library usage, thread parallelism | Memory model, kernel launch, streams, reductions |
| **2 — Optimization** | Shared memory, register tiling, real-time pipelines | Coalescing, occupancy, warp-level ops, BPTT |
| **3 — Production** | Tensor Cores, multi-backend, quantization, caching | cuBLAS GEMM, KV cache, INT8 calibration, speculative decode |

---

## Profiling & Debugging

```bash
# System-wide timeline
nsys profile --stats=true ./my_app

# Kernel-level analysis
ncu --set full ./my_app

# Memory errors
compute-sanitizer --tool memcheck ./my_app

# Race conditions
compute-sanitizer --tool racecheck ./my_app
```

---

## Repository Statistics

```
Projects                14
Custom CUDA kernels     120+
Lines of CUDA/C++       35,000+
Benchmark data points   200+
GPU architectures       Turing → Ada Lovelace (SM 7.5–8.9)
```

---

## License

Released under the [MIT License](LICENSE).

---

## Contact

**Ahmet Faruk Gumustas** — [faruk.gmstss@gmail.com](mailto:faruk.gmstss@gmail.com)

Issues and feature requests: [GitHub Issues](https://github.com/ahmet-f-gumustas/cuda-advanced-projects/issues)

---

<div align="center">
<sub>Built with CUDA and Modern C++ — Pushing the boundaries of GPU computing</sub>
</div>
