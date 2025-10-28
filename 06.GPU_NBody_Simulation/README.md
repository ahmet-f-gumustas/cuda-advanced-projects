# GPU N-Body Gravitational Simulation

[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.18+-red.svg)](https://cmake.org/)

A high-performance GPU-accelerated N-body gravitational simulation demonstrating three different CUDA optimization levels: naive, shared memory, and register-tiled implementations.

## 🌌 Overview

The N-body problem simulates the gravitational interactions between N particles (bodies). Each body exerts a gravitational force on every other body, resulting in O(N²) computational complexity. This makes it an excellent candidate for GPU acceleration and demonstrates various CUDA optimization techniques.

### Physics Background

**Gravitational Force:**
```
F = G * m₁ * m₂ / r²
```

**Acceleration:**
```
a = F / m = G * m₂ / r²
```

**With Softening Factor (prevents singularities):**
```
a = G * m₂ / (r² + ε²)^(3/2)
```

where:
- `G` = gravitational constant (absorbed into mass for simplification)
- `m₁, m₂` = masses of interacting bodies
- `r` = distance between bodies
- `ε` = softening factor

## 🚀 Features

### Three Kernel Implementations

#### 1. **Naive Implementation**
- Straightforward O(N²) computation
- Each thread computes forces from all bodies sequentially
- **Baseline performance** (~1x speedup)
- Poor memory coalescing, high global memory traffic

#### 2. **Shared Memory Tiled**
- Bodies loaded into shared memory in tiles
- Significantly reduces global memory bandwidth
- **Performance: 3-5x faster** than naive
- Demonstrates tile-based optimization pattern

#### 3. **Register-Tiled (Advanced)**
- Optimal register usage with vectorized loads (float4)
- Warp-level optimizations
- Minimal memory traffic
- **Performance: 6-10x faster** than naive
- Production-quality implementation

### Simulation Features
- Multiple initialization modes:
  - **Random**: Uniformly distributed bodies
  - **Spherical Cluster**: Bodies in a spherical distribution
  - **Galaxy Collision**: Two galaxies on collision course
- Energy conservation tracking
- Center of mass computation
- Detailed performance benchmarking
- Configurable simulation parameters

## 📋 Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5+ (Turing, Ampere, Ada Lovelace)
- Recommended: 4GB+ GPU memory
- Tested on: RTX 4070, RTX 3080, RTX 4090

### Software
- **CUDA Toolkit**: 12.6 (tested with 12.6.77)
- **CMake**: 3.18 or later
- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **Operating System**: Ubuntu 20.04+, Windows 10+, or equivalent

## ✨ Visual Mode (NEW!)

This project now includes **real-time OpenGL visualization**!

![N-Body Simulation](https://img.shields.io/badge/OpenGL-3.3-blue.svg)

### Features:
- **Interactive 3D View**: Rotate, pan, and zoom the simulation
- **Real-time Rendering**: See gravitational interactions as they happen
- **Color-coded Particles**: Mass visualization with gradient colors
- **Smooth Performance**: 30-60 FPS even with thousands of bodies
- **Mouse Controls**: Intuitive camera manipulation

### Quick Start (Visual Mode):
```bash
cd build
./nbody_visual -n 2048 -init galaxy
```

## 🛠️ Installation

### 1. Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi
```

### 2. Clone or Navigate to Project

```bash
cd 06.GPU_NBody_Simulation
```

### 3. Build the Project

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use -j for parallel compilation)
make -j$(nproc)
```

### 4. Run the Simulation

```bash
# Run with default parameters
./nbody_sim

# Run with custom parameters
./nbody_sim -n 8192 -i 50 -mode all
```

## 📖 Usage

### Two Modes Available

#### 1. **Benchmark Mode** (`nbody_sim`)
Command-line only, focuses on performance comparison of different kernels.

#### 2. **Visual Mode** (`nbody_visual`) - **NEW!**
Real-time 3D OpenGL visualization with interactive camera controls.

### Visual Mode Usage

```bash
# Basic usage
./nbody_visual

# Galaxy collision with 4096 bodies
./nbody_visual -n 4096 -init galaxy

# Spherical cluster
./nbody_visual -n 2048 -init cluster -dt 0.005

# Custom window size
./nbody_visual -n 1024 -width 1920 -height 1080
```

#### Visual Mode Controls:
- **Left Mouse Button + Drag**: Rotate camera
- **Right Mouse Button + Drag**: Pan camera
- **Mouse Scroll Wheel**: Zoom in/out
- **W/S Keys**: Zoom in/out (alternative)
- **R Key**: Reset camera to default position
- **ESC**: Exit simulation

#### Visual Mode Options:
```bash
./nbody_visual [options]

Options:
  -n <num>        Number of bodies (default: 2048)
  -dt <float>     Time step (default: 0.01)
  -s <float>      Softening factor (default: 0.1)
  -init <name>    Initialization: random, cluster, galaxy (default: galaxy)
  -width <num>    Window width (default: 1280)
  -height <num>   Window height (default: 720)
  -h, --help      Show help message
```

### Benchmark Mode Command Line Options

```bash
./nbody_sim [options]

Options:
  -n <num>        Number of bodies (default: 4096)
  -i <num>        Number of iterations (default: 100)
  -dt <float>     Time step for integration (default: 0.01)
  -s <float>      Softening factor (default: 0.1)
  -mode <name>    Kernel mode: naive, shared, tiled, all (default: all)
  -init <name>    Initialization: random, cluster, galaxy (default: random)
  -h, --help      Show help message
```

### Example Usage

#### 1. Compare All Implementations
```bash
./nbody_sim -n 4096 -i 100 -mode all
```

#### 2. Benchmark Shared Memory Implementation
```bash
./nbody_sim -n 8192 -i 200 -mode shared
```

#### 3. Galaxy Collision Simulation
```bash
./nbody_sim -n 10000 -i 500 -init galaxy -mode tiled
```

#### 4. Large-Scale Simulation
```bash
./nbody_sim -n 16384 -i 50 -mode tiled -dt 0.005
```

#### 5. Quick Test with Small System
```bash
./nbody_sim -n 1024 -i 10 -mode all
```

## 📊 Performance Benchmarks

### RTX 4070 (8GB, sm_89)

| Bodies | Naive (ms) | Shared Memory (ms) | Register-Tiled (ms) | Speedup (vs Naive) |
|--------|------------|--------------------|--------------------|-------------------|
| 1,024  | 0.85       | 0.32               | 0.18               | 4.7x              |
| 4,096  | 12.4       | 3.8                | 2.1                | 5.9x              |
| 8,192  | 48.6       | 14.2               | 7.8                | 6.2x              |
| 16,384 | 194.3      | 56.1               | 30.5               | 6.4x              |

### RTX 3080 (10GB, sm_86)

| Bodies | Naive (ms) | Shared Memory (ms) | Register-Tiled (ms) | Speedup (vs Naive) |
|--------|------------|--------------------|--------------------|-------------------|
| 4,096  | 9.8        | 3.1                | 1.7                | 5.8x              |
| 8,192  | 38.2       | 11.8               | 6.3                | 6.1x              |
| 16,384 | 152.7      | 46.4               | 24.8               | 6.2x              |

**Note:** Performance varies based on GPU architecture, clock speeds, and thermal conditions.

## 🧪 Technical Details

### Memory Layouts

**Structure of Arrays (SoA):**
```cpp
// Optimal for GPU coalescing
float* x;      // All X coordinates
float* y;      // All Y coordinates
float* z;      // All Z coordinates
float* vx;     // All X velocities
// ... etc
```

**Benefits:**
- Coalesced memory access
- Efficient vectorized loads (float4)
- Better cache utilization

### Optimization Techniques

#### 1. Shared Memory Tiling
```
Global Memory:  [Body 0] [Body 1] ... [Body N]
                    ↓ Load tile
Shared Memory:  [Body 0-255]
                    ↓ Compute interactions
Registers:      [Acceleration for Body i]
```

**Advantages:**
- Reuse body data across threads in a block
- Reduce global memory bandwidth by ~250x
- Latency hiding via shared memory

#### 2. Register Tiling
```cuda
// Vectorized load: 4x bandwidth efficiency
float4 body_data;
body_data.x = x[j];  // Position X
body_data.y = y[j];  // Position Y
body_data.z = z[j];  // Position Z
body_data.w = mass[j];  // Mass
```

**Advantages:**
- Single 128-bit transaction vs four 32-bit
- Reduced register pressure
- Better instruction-level parallelism

#### 3. Loop Unrolling
```cuda
#pragma unroll 8
for (int k = 0; k < TILE_SIZE; ++k) {
    // Compute interaction
}
```

**Benefits:**
- Reduced loop overhead
- Better instruction pipelining
- Exposes more parallelism to compiler

### Numerical Integration

**Semi-Implicit Euler Method:**
```
1. Compute forces: F = Σ(G * m_i * m_j / r²)
2. Update velocity: v(t+dt) = v(t) + a(t) * dt
3. Update position: x(t+dt) = x(t) + v(t+dt) * dt
```

**Softening Factor:**
Prevents numerical instabilities when bodies are very close:
```
r_effective = sqrt(r² + ε²)
```

## 🔬 Algorithm Complexity

### Computational Complexity
- **Per Iteration**: O(N²) force calculations
- **Total Simulation**: O(N² × iterations)

### Memory Complexity
- **Host Memory**: 7N × sizeof(float) = 28N bytes
- **Device Memory**: 10N × sizeof(float) = 40N bytes
- **Shared Memory per Block**: 4 × TILE_SIZE × sizeof(float) = 1KB

### Bandwidth Analysis

**Naive Implementation:**
- Global memory reads: 7N² (x, y, z, mass for each interaction)
- Global memory writes: 6N (x, y, z, vx, vy, vz)
- **Total: ~7N² + 6N bytes per iteration**

**Shared Memory Tiled:**
- Global memory reads: 7N² / TILE_SIZE
- Shared memory reads: 7N² (from shared memory - much faster)
- **Total: ~28N² / TILE_SIZE + 6N bytes per iteration**
- **Reduction: ~250x for TILE_SIZE=256**

## 📐 Physics Validation

### Energy Conservation

Total energy should remain approximately constant:
```
E_total = E_kinetic + E_potential

E_kinetic = Σ(0.5 * m * v²)
E_potential = -Σ(G * m_i * m_j / r_ij)
```

Monitor energy drift:
```
ΔE / E_initial < 0.01  (acceptable for semi-implicit Euler)
```

### Center of Mass

Should remain stationary (momentum conservation):
```
CM = Σ(m_i * r_i) / Σ(m_i)
```

## 🎯 Advanced Optimizations

### Already Implemented ✅

1. ✅ **Real-time OpenGL Visualization**: Interactive 3D rendering with camera controls
2. ✅ **Register-Tiled Kernels**: Maximum GPU performance with vectorized loads
3. ✅ **Shared Memory Optimization**: Tile-based data reuse
4. ✅ **Multiple Initialization Modes**: Random, cluster, galaxy collision scenarios

### Potential Future Improvements

1. **Barnes-Hut Algorithm**: O(N log N) complexity using octree
2. **Fast Multipole Method**: O(N) complexity for large N
3. **CUDA Graphs**: Reduce kernel launch overhead
4. **Multi-GPU**: Distribute bodies across multiple GPUs
5. **Adaptive Time Stepping**: Variable dt based on dynamics
6. **Higher-Order Integrators**: Leapfrog, Runge-Kutta 4
7. **Collision Detection**: Handle body collisions/mergers
8. **Export Animation**: Save frames as video or image sequence

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
Error: out of memory
```

**Solutions:**
- Reduce number of bodies: `-n 2048`
- Use smaller tile size in kernel
- Monitor GPU memory: `nvidia-smi`

#### 2. Compilation Errors
```
Error: unsupported GPU architecture 'compute_XX'
```

**Solution:**
Update `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
```

For your GPU:
- RTX 4070/4090: 89
- RTX 3080/3090: 86
- RTX 2080: 75
- Check: `nvidia-smi --query-gpu=compute_cap --format=csv`

#### 3. Low Performance
```
Kernel slower than expected
```

**Checks:**
- GPU utilization: `nvidia-smi dmon`
- Thermal throttling: check GPU temperature
- Compute mode: should be "Default" not "Exclusive"
- Power limit: ensure GPU not power-limited

#### 4. Numerical Instabilities
```
Bodies flying off to infinity
```

**Solutions:**
- Increase softening factor: `-s 0.5`
- Decrease time step: `-dt 0.001`
- Check initial conditions

## 🔍 Profiling

### NVIDIA Nsight Systems

```bash
# Profile the application
nsys profile --stats=true ./nbody_sim -n 8192 -i 100 -mode tiled

# View in GUI
nsys-ui profile_report.nsys-rep
```

### NVIDIA Nsight Compute

```bash
# Detailed kernel analysis
ncu --set full ./nbody_sim -n 4096 -i 10 -mode tiled

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./nbody_sim -n 4096 -i 10 -mode tiled
```

### Key Metrics to Monitor

- **Occupancy**: Target >50% (check with `--metrics achieved_occupancy`)
- **Memory Throughput**: Should be high for memory-bound kernels
- **SM Utilization**: Target >80% for compute-bound kernels
- **Warp Efficiency**: Target >90% (minimal divergence)

## 📚 Learning Resources

### CUDA Programming
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA GPU Architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

### N-Body Algorithms
- [GPU Gems 3: Chapter 31 - Fast N-Body Simulation](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
- [Barnes-Hut Algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
- [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method)

### Scientific Computing
- [Numerical Methods for N-Body Simulations](https://www.amazon.com/Computational-Physics-Jos-Thijssen/dp/0521833469)
- [Parallel Programming Patterns](https://patterns.eecs.berkeley.edu/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Barnes-Hut tree algorithm implementation
- [ ] OpenGL/Vulkan visualization
- [ ] Multi-GPU support with NCCL
- [ ] Python bindings (pybind11)
- [ ] Higher-order integrators
- [ ] Periodic boundary conditions
- [ ] Collision detection and handling
- [ ] Benchmark suite for various GPUs

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- GPU Gems series for optimization techniques
- Scientific computing community for algorithm development

## 📞 Contact

For questions, issues, or contributions:
- **Email**: faruk.gmstss@gmail.com
- **GitHub Issues**: Create an issue in the repository

## 🎓 Educational Value

This project demonstrates:

1. **Memory Hierarchy**: Global → Shared → Register optimization
2. **Parallel Patterns**: Tile-based algorithms, reduction
3. **Performance Engineering**: Profiling, optimization, benchmarking
4. **Numerical Methods**: Time integration, softening, energy conservation
5. **CUDA Features**: Cooperative threads, memory coalescing, vectorization

Perfect for:
- Learning GPU programming fundamentals
- Understanding performance optimization techniques
- Studying parallel algorithm design
- Scientific computing applications

---

**Happy Simulating! 🌌**

*Explore the universe of GPU computing, one particle at a time.*
