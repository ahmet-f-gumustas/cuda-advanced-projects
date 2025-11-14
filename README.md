# ⚡ CUDA Advanced Projects

**Professional CUDA programming and GPU-accelerated computing collection** - Comprehensive CUDA applications from deep learning to scientific computing, image processing to physics simulations.

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)](https://en.cppreference.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()

---

## 🎯 About This Project

This repository is a comprehensive collection of **industry-standard CUDA applications**. Each project explores different aspects of GPU programming in depth and demonstrates optimization techniques that can be used in real-world applications.

### 🌟 Key Features

- **Production-Ready Code**: Industry-quality, tested implementations
- **Comprehensive Optimizations**: Shared memory, tensor cores, register tiling, memory coalescing
- **Real-Time Applications**: OpenGL visualization and interactive simulations
- **Multi-Backend Support**: CUDA, cuDNN, cuBLAS, TensorRT, ONNX Runtime
- **Detailed Documentation**: Comprehensive README and code explanations for each project
- **Performance Benchmarks**: Performance metrics measured on real GPUs

---

## 📚 Projects

### 🧠 01. Deep Learning Inference Engine

**High-performance, CUDA-based deep learning inference engine**

Industry-grade inference engine optimized for modern GPUs. Offers competitive performance with TensorRT.

**Features:**
- ✅ Tensor Core support (Ampere, Ada Lovelace)
- ✅ INT8/FP16 quantization (2.5-3x speedup)
- ✅ Layer fusion optimizations
- ✅ Multi-GPU pipeline support
- ✅ ResNet, YOLO, BERT, ViT model support
- ✅ ONNX, TensorFlow, PyTorch model import

**Performance:**
- ResNet-50: 0.4ms latency (INT8, RTX 4090)
- YOLOv5s: 714 FPS (INT8, RTX 4070)
- BERT-Base: 8.3ms latency (INT8, batch=32)

**Technologies:** CUDA, cuDNN, cuBLAS, Tensor Cores, Custom Kernels

📁 [Detailed Documentation](01.deeplearning_inference_engine/README.md)

---

### 🖼️ 02. Real-Time Image Denoising Engine

**GPU-accelerated real-time image denoising engine**

Implements Bilateral Filter, Non-Local Means, and Adaptive denoising algorithms with CUDA kernels.

**Features:**
- ✅ Bilateral Filter (edge-preserving smoothing)
- ✅ Non-Local Means (patch-based denoising)
- ✅ Adaptive Bilateral (dynamic parameter adjustment)
- ✅ 30+ FPS @ 1080p video streams
- ✅ Python and C++ APIs
- ✅ PyQt5 GUI application
- ✅ Camera integration

**Performance:**
- 1080p @ 45-60 FPS (Bilateral, RTX 4070)
- 4K @ 12-15 FPS (Bilateral, RTX 4070)
- Sub-10ms latency on modern GPUs

**Technologies:** CUDA, OpenCV, PyQt5, Custom Kernels, Shared Memory

📁 [Detailed Documentation](02.Real-Time_Image_Denoising_Engine/README.md)

---

### 🔍 03. Vision Transformer (ViT-B/16) CUDA C++

**Pure CUDA C++20 implementation, zero Python dependency**

Production-ready Vision Transformer implementation with FP16 tensor core acceleration.

**Features:**
- ✅ ViT-B/16 full implementation
- ✅ FP16 inference with Tensor Core utilization
- ✅ Custom CUDA kernels (LayerNorm, Softmax, GELU)
- ✅ Optimized GEMM with cuBLASLt
- ✅ Memory-efficient design
- ✅ Multi-batch support
- ✅ 128-bit aligned memory

**Performance:**
- Batch 1: 5.2ms (RTX 4070, FP16)
- Batch 32: 64.8ms, 494 img/s (RTX 4070)
- Batch 256: 308.7ms, 829 img/s (A100)

**Technologies:** CUDA C++20, cuBLASLt, Tensor Cores, Warp Primitives

📁 [Detailed Documentation](03.vit-B16_CUDA_Cpp/README.md)

---

### 🚀 04. EfficientNet-B0 CUDA

**Multi-backend CNN inference: ONNX Runtime, TensorRT, cuDNN**

Cross-platform EfficientNet implementation with x86_64 and aarch64 (Jetson) support.

**Features:**
- ✅ ONNX Runtime CUDA Execution Provider
- ✅ TensorRT (Jetson optimized)
- ✅ cuDNN benchmark backend
- ✅ NVJPEG GPU decoding
- ✅ CUDA preprocessing pipeline
- ✅ FP16/INT8 support
- ✅ CUDA Graphs, IOBinding
- ✅ No OpenCV dependency

**Backend Support:**
- ONNX Runtime (dev/production)
- TensorRT (Jetson deployment)
- cuDNN (benchmarking)

**Technologies:** CUDA, ONNX Runtime, TensorRT, cuDNN, NVJPEG

📁 [Detailed Documentation](04.EfficientNet-B0_CUDA_/README.md)

---

### 📊 05. cuFFT Tutorial

**CUDA Fast Fourier Transform examples and tutorial**

Comprehensive examples demonstrating cuFFT usage for signal processing and scientific computing.

**Features:**
- ✅ 1D, 2D FFT examples
- ✅ Real-to-Complex, Complex-to-Complex
- ✅ Batch FFT processing
- ✅ FFT-based convolution
- ✅ CPU vs GPU benchmark
- ✅ Multi-GPU examples

**Use Cases:**
- Signal processing
- Image filtering
- Spectral analysis
- Scientific simulations

**Technologies:** cuFFT, CUDA Streams, Batch Processing

📁 [Detailed Documentation](05.cuFFt_/README.md)

---

### 🌌 06. GPU N-Body Gravitational Simulation

**High-performance gravity simulation with OpenGL visualization**

Interactive N-body simulation demonstrating three different CUDA optimization levels.

**Features:**
- ✅ 3 different kernel implementations:
  - Naive (baseline)
  - Shared Memory Tiled (3-5x faster)
  - Register-Tiled (6-10x faster)
- ✅ Real-time OpenGL 3D visualization
- ✅ Interactive camera controls
- ✅ Energy conservation tracking
- ✅ Galaxy collision scenarios
- ✅ 16K+ particle support

**Performance (Register-Tiled):**
- 1,024 bodies: 0.18ms (RTX 4070)
- 8,192 bodies: 7.8ms (RTX 4070)
- 16,384 bodies: 30.5ms (RTX 4070)

**Technologies:** CUDA, OpenGL 3.3, GLFW, Shared Memory, Register Optimization

📁 [Detailed Documentation](06.GPU_NBody_Simulation/README.md)

---

### 🤖 07. Parallel ML Models

**CUDA parallel machine learning: Linear Regression + K-Means**

Educational project demonstrating parallel training of two different ML models.

**Features:**
- ✅ Linear Regression (CUDA implementation)
- ✅ K-Means Clustering (CUDA implementation)
- ✅ Parallel training (multi-threading + multi-stream)
- ✅ OpenGL real-time visualization
- ✅ Gradient descent optimization
- ✅ Shared memory reduction patterns

**Learning Content:**
- Thread-level parallelism
- CUDA stream parallelism
- GPU kernel parallelism
- Reduction patterns
- Memory coalescing

**Technologies:** CUDA, OpenGL, C++ Threading, Shared Memory

📁 [Detailed Documentation](07-parallel-ml-models/README.md)

---

### 🔬 08. Simple CUDA CNN

**Simple CNN implementation with Python bindings**

Minimal CNN implementation demonstrating CUDA kernel usage from Python.

**Features:**
- ✅ 2D Convolution kernel
- ✅ ReLU activation
- ✅ Max Pooling
- ✅ Python ctypes bindings
- ✅ NumPy integration
- ✅ Performance timing
- ✅ Shared library (.so) build

**Performance:**
- 10x10 input: ~0.019ms
- Throughput: ~51,000 FPS

**Usage:**
```python
import ctypes
import numpy as np
lib = ctypes.CDLL('./build/libcnn_cuda.so')
ctx = lib.create_cnn_context(10, 10)
lib.run_cnn_forward(ctx)
```

**Technologies:** CUDA, Python ctypes, NumPy, CMake

📁 [Detailed Documentation](08.simple-cuda-cnn/README.md)

---

## 🛠️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU, Compute Capability 7.0+ (Turing, Ampere, Ada Lovelace)
- **VRAM**: 4GB minimum (8GB+ recommended)
- **RAM**: 16GB+ (for compilation and development)
- **Tested GPUs**: RTX 4070, RTX 4090, RTX 3080, RTX 3090, A100

### Software
- **CUDA Toolkit**: 11.0+ (12.0+ recommended)
- **CMake**: 3.18+
- **C++ Compiler**:
  - Linux: GCC 9+, Clang 10+
  - Windows: MSVC 2019+
- **Python**: 3.7+ (for Python projects)
- **Operating System**:
  - Ubuntu 20.04/22.04 LTS
  - Windows 10/11
  - CentOS 8/RHEL 8

### Additional Libraries (project-specific)
- OpenCV 4.x (image processing projects)
- OpenGL 3.3+ (visualization projects)
- GLFW3, GLEW (OpenGL projects)
- cuDNN 8.x (deep learning projects)
- PyQt5 (GUI applications)

---

## 🚀 Quick Start

### 1. Install CUDA Toolkit

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/cuda-advanced-projects.git
cd cuda-advanced-projects
```

### 3. Build a Project

```bash
# Example: N-Body Simulation
cd 06.GPU_NBody_Simulation
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./nbody_visual -n 2048 -init galaxy
```

### 4. Verify GPU Setup

```bash
# Check GPU and CUDA
nvidia-smi
nvcc --version

# Check Compute Capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## 📊 Performance Highlights

### Deep Learning Inference
| Model | Precision | Latency | Hardware |
|-------|-----------|---------|----------|
| ResNet-50 | INT8 | 0.4ms | RTX 4090 |
| YOLOv5s | FP16 | 2.1ms | RTX 4070 |
| BERT-Base | FP16 | 12.5ms | RTX 4070 |
| ViT-B/16 | FP16 | 5.2ms | RTX 4070 |

### Computer Vision
| Task | Resolution | FPS | Hardware |
|------|------------|-----|----------|
| Image Denoising | 1080p | 45-60 | RTX 4070 |
| Image Denoising | 4K | 12-15 | RTX 4070 |

### Scientific Computing
| Application | Size | Time | Hardware |
|------------|------|------|----------|
| N-Body (Tiled) | 16K particles | 30.5ms | RTX 4070 |
| FFT 2D | 2048x2048 | ~1ms | RTX 4070 |

---

## 🎓 Learning Path

We recommend exploring these projects in the following order:

### Level 1: CUDA Fundamentals
1. **08.simple-cuda-cnn** - Basic CUDA kernels and Python binding
2. **05.cuFFt_** - cuFFT library usage
3. **07-parallel-ml-models** - Thread and stream parallelism

### Level 2: Intermediate Optimizations
4. **02.Real-Time_Image_Denoising_Engine** - Shared memory, memory coalescing
5. **06.GPU_NBody_Simulation** - Register tiling, reduction patterns

### Level 3: Advanced and Production
6. **03.vit-B16_CUDA_Cpp** - Tensor cores, warp primitives, cuBLASLt
7. **04.EfficientNet-B0_CUDA_** - Multi-backend, production deployment
8. **01.deeplearning_inference_engine** - Complete inference framework

---

## 🔬 Technical Features and Optimizations

### Memory Optimizations
- ✅ **Shared Memory Tiling** - Reduce global memory access
- ✅ **Register Tiling** - Register usage optimization
- ✅ **Memory Coalescing** - Optimal memory access patterns
- ✅ **Memory Pooling** - Reduce allocation overhead
- ✅ **Pinned Memory** - Async host-device transfers

### Compute Optimizations
- ✅ **Tensor Core Utilization** - FP16/INT8 acceleration
- ✅ **Warp-level Primitives** - Shuffle, ballot, sync
- ✅ **Loop Unrolling** - Instruction-level parallelism
- ✅ **Kernel Fusion** - Memory bandwidth optimization
- ✅ **CUDA Graphs** - Launch overhead reduction

### Parallelism Patterns
- ✅ **Data Parallelism** - Parallel data processing
- ✅ **Task Parallelism** - Multi-stream execution
- ✅ **Pipeline Parallelism** - Overlapped execution
- ✅ **Reduction Patterns** - Efficient aggregation
- ✅ **Scan/Prefix Sum** - Parallel prefix operations

---

## 🧪 Profiling and Debugging

### NVIDIA Nsight Systems
```bash
# System-wide profiling
nsys profile --stats=true ./my_app

# CUDA, cuDNN, cuBLAS tracing
nsys profile -t cuda,cudnn,cublas,nvtx ./my_app
```

### NVIDIA Nsight Compute
```bash
# Kernel-level profiling
ncu --set full ./my_app

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./my_app
```

### CUDA Memory Checker
```bash
# Memory leak detection
cuda-memcheck ./my_app

# Race condition detection
compute-sanitizer --tool racecheck ./my_app
```

---

## 📖 Documentation

Each project contains detailed documentation in its own README:

- 🔧 **Installation instructions**
- 💻 **Usage examples**
- 📊 **Performance benchmarks**
- 🎯 **API references**
- 🐛 **Troubleshooting**
- 📚 **Technical details**

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Ideas
- [ ] New ML model implementations
- [ ] Mobile GPU support (Jetson)
- [ ] Vulkan compute shaders
- [ ] Rust/Go bindings
- [ ] WebGPU backend
- [ ] Distributed computing (multi-node)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

These projects were developed thanks to the following resources and communities:

- **NVIDIA**: CUDA Toolkit, cuDNN, TensorRT, documentation
- **Open Source Community**: PyTorch, TensorFlow, ONNX Runtime
- **Academic Research**: Vision Transformer, EfficientNet papers
- **GPU Programming Guides**: GPU Gems, CUDA Best Practices

Special thanks to:
- NVIDIA Developer program
- CUDA programming community
- All open source contributors

---

## 📞 Contact and Support

- **Email**: faruk.gmstss@gmail.com
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and discussions

---

## 🗺️ Roadmap

### Completed Projects
- [x] Deep Learning Inference Engine
- [x] Image Denoising Engine
- [x] Vision Transformer CUDA
- [x] N-Body Simulation with OpenGL
- [x] EfficientNet Multi-Backend Implementation
- [x] cuFFT Tutorial and Examples
- [x] Parallel ML Models (Linear Regression + K-Means)
- [x] Simple CUDA CNN with Python Bindings

### Planned Features and Projects
- [ ] Benchmark suite for various GPUs
- [ ] Docker containers for easy deployment
- [ ] Transformer decoder CUDA implementation
- [ ] Real-time video processing pipeline
- [ ] Multi-GPU training framework
- [ ] Jetson optimization pack
- [ ] WebGPU compute shaders
- [ ] Sparse tensor operations
- [ ] Neural architecture search
- [ ] Distributed inference
- [ ] Edge deployment tools
- [ ] Automated optimization pipeline

---

## 📊 Project Statistics

```
Total Projects:       8
Total CUDA Kernels:   50+
Lines of CUDA Code:   15,000+
Benchmark Results:    100+
Documentation Pages:  50+
Supported GPUs:       Pascal to Ada Lovelace
```

---

## 🎯 Use Cases

### Research & Academia
- GPU programming education
- Algorithm development
- Performance analysis
- Scientific computing

### Industry & Production
- High-performance inference
- Real-time video processing
- Scientific simulations
- Edge AI deployment

### Learning & Development
- CUDA programming fundamentals
- Optimization techniques
- Parallel algorithm design
- Production deployment

---

## ⭐ Star History

If you like this project, don't forget to give it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/cuda-advanced-projects&type=Date)](https://star-history.com/#yourusername/cuda-advanced-projects&Date)

---

<div align="center">

**🚀 Built with ❤️ using CUDA and Modern C++**

*Pushing the boundaries of GPU computing*

[⬆ Back to Top](#-cuda-advanced-projects)

</div>
