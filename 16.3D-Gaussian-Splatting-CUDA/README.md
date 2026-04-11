<div align="center">

# 3D Gaussian Splatting — CUDA

**Real-time 3D Gaussian Splatting renderer built from scratch in CUDA + OpenGL**

[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus)](https://en.cppreference.com/)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.3-5586A4?style=flat-square&logo=opengl)](https://www.opengl.org/)

</div>

---

## Overview

A tile-based rasterizer for 3D Gaussian Splatting, implementing the core rendering pipeline from [Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Given a set of 3D Gaussians with position, covariance, spherical harmonics color, and opacity, the renderer produces real-time alpha-blended images via a fully GPU-accelerated pipeline.

**No training / backward pass** — this is a pure inference renderer designed for maximum throughput.

---

## Architecture

```
Input: 3D Gaussians (pos, cov3D, SH coeffs, opacity)
         │
         ▼
┌─────────────────────────┐
│  1. Preprocess           │  Frustum culling, 3D→2D projection,
│                          │  EWA splatting (Cov3D→Cov2D),
│                          │  SH evaluation (degree 0-3), tile assignment
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  2. Tile Sort            │  CUB prefix sum + radix sort
│                          │  Key = (tile_id << 32) | depth
│                          │  Per-tile depth-ordered Gaussian lists
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  3. Rasterize            │  One thread block (16×16) per tile
│                          │  Front-to-back alpha compositing
│                          │  Shared memory batching, early exit
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  4. Display              │  OpenGL PBO, fullscreen quad
│                          │  Interactive orbit/pan/zoom camera
└─────────────────────────┘
```

---

## Key Features

- **Tile-based rasterization** — screen divided into 16×16 tiles, each processed by one thread block
- **EWA splatting** — 3D covariance projected to 2D via Jacobian of perspective projection
- **Spherical harmonics** — degree 0–3 view-dependent color evaluation (up to 16 coefficients)
- **CUB radix sort** — GPU-accelerated sorting for per-tile depth ordering
- **Front-to-back alpha compositing** — with early termination when transmittance drops below threshold
- **Interactive OpenGL viewer** — orbit, pan, zoom with real-time FPS display
- **PLY file support** — load standard 3DGS `.ply` output from training pipelines
- **No GLEW, No GLM** — manual OpenGL function loading, custom matrix math

---

## Build

```bash
cd 16.3D-Gaussian-Splatting-CUDA
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Dependencies

| Library | Purpose | Notes |
|---------|---------|-------|
| CUDA Toolkit 12.x | GPU compute + CUB | CUB is header-only, ships with CUDA |
| OpenGL 3.3+ | Display | System driver |
| GLFW3 | Window + input | `sudo apt install libglfw3-dev` |

### Executables

| Binary | Description |
|--------|-------------|
| `gs_viewer` | Interactive real-time viewer |
| `gs_benchmark` | Performance benchmarks |
| `gs_test` | Unit tests with CPU references |

---

## Usage

```bash
# Run tests
./gs_test

# Interactive viewer (random scene)
./gs_viewer

# Benchmark
./gs_benchmark
```

### Viewer Controls

| Input | Action |
|-------|--------|
| Left mouse drag | Orbit camera |
| Right mouse drag | Pan camera |
| Scroll wheel | Zoom in/out |
| `R` | Reset camera |
| `P` | Print stats |
| `ESC` | Exit |

---

## Project Structure

```
16.3D-Gaussian-Splatting-CUDA/
├── CMakeLists.txt
├── README.md
├── PLAN.md
├── include/
│   ├── cuda_utils.h            # Error checks, timer, memory helpers, float3 math
│   ├── gaussian.h              # GaussianData (SoA), GaussianModel class
│   ├── camera.h                # Mat4, Camera (view/proj), orbit/pan/zoom
│   ├── splatting_kernels.cuh   # Kernel launch declarations
│   ├── renderer.h              # Render pipeline orchestrator
│   ├── ply_reader.h            # Binary PLY parser
│   └── viewer.h                # OpenGL viewer + GLFW
├── src/
│   ├── gaussian.cu             # Gaussian data management + random generation
│   ├── splatting_kernels.cu    # All CUDA kernels (preprocess, sort, rasterize)
│   ├── renderer.cu             # Pipeline orchestration
│   ├── ply_reader.cpp          # PLY file I/O
│   ├── viewer.cu               # OpenGL display loop
│   ├── main.cu                 # Viewer entry point
│   └── benchmark.cu            # Performance measurement
├── tests/
│   └── test_splatting.cu       # Unit tests
├── data/                       # Sample .ply files
└── output/                     # Rendered frames
```

---

## Technical Details

### EWA Splatting

3D covariance is projected to 2D screen space using the Jacobian of perspective projection:

```
Cov3D = R · S · Sᵀ · Rᵀ          (R from quaternion, S = diag(exp(scale)))
J = [[fx/z,  0,    -fx·x/z²],     (2×3 Jacobian)
     [0,     fy/z, -fy·y/z²]]
T = J · W                          (W = view rotation 3×3)
Cov2D = T · Cov3D · Tᵀ + 0.3·I   (low-pass filter)
```

### Spherical Harmonics

View-dependent color from hardcoded polynomial basis functions:

| Degree | Coefficients | Effect |
|--------|-------------|--------|
| 0 | 1 | Constant (DC) color |
| 1 | 4 | Diffuse-like shading |
| 2 | 9 | Specular-like highlights |
| 3 | 16 | Full 3DGS quality |

### Alpha Compositing

```
T = 1.0, C = 0.0
for each gaussian (front-to-back):
    α = opacity · exp(-0.5 · d_mahalanobis)
    C += T · α · color
    T *= (1 - α)
    if T < 1e-4: break    // early termination
final = C + T · bg_color
```

---

## Performance Targets

RTX 4070 Laptop (SM 8.9, 7836 MB VRAM):

| Scene Size | Resolution | Target FPS |
|-----------|------------|------------|
| 10K Gaussians | 800×600 | >200 |
| 100K Gaussians | 1280×720 | >60 |
| 500K Gaussians | 1920×1080 | >30 |
| 1M Gaussians | 1920×1080 | >15 |

---

## References

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- Zwicker et al., "EWA Splatting", IEEE TVCG 2002
- Green, "Spherical Harmonic Lighting: The Gritty Details", GDC 2003

---

## License

Released under the [MIT License](../LICENSE).
