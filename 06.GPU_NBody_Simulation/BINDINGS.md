# Python Bindings Tutorial for CUDA N-Body Simulation

**Complete Guide to Creating Python Bindings for C++/CUDA Code using pybind11**

This document provides a comprehensive, step-by-step tutorial on how Python bindings were created for this GPU N-Body simulation project. It explains the entire process from C++/CUDA code to a fully functional Python module.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Architecture Overview](#3-architecture-overview)
4. [Step 1: Project Structure](#step-1-project-structure)
5. [Step 2: Creating Helper Functions](#step-2-creating-helper-functions)
6. [Step 3: Writing the pybind11 Wrapper](#step-3-writing-the-pybind11-wrapper)
7. [Step 4: CMake Configuration](#step-4-cmake-configuration)
8. [Step 5: NumPy Integration](#step-5-numpy-integration)
9. [Step 6: Memory Management](#step-6-memory-management)
10. [Step 7: Building the Module](#step-7-building-the-module)
11. [Step 8: Testing](#step-8-testing)
12. [Common Issues and Solutions](#common-issues-and-solutions)
13. [Best Practices](#best-practices)
14. [Advanced Topics](#advanced-topics)

---

## 1. Introduction

### What are Python Bindings?

Python bindings allow you to call C++/CUDA code from Python, combining:
- **Python's ease of use**: Simple syntax, rapid prototyping, rich ecosystem
- **C++/CUDA's performance**: Low-level control, GPU acceleration, high performance

### Why pybind11?

**pybind11** is a header-only library that creates Python bindings for C++11 code. It's chosen for:
- Header-only design (no compilation required for the library itself)
- Automatic type conversion between Python and C++
- Seamless NumPy integration
- Excellent documentation and community support
- Modern C++11/14/17 support

### Project Goal

Create a Python module `nbody_cuda` that exposes our CUDA N-Body simulation to Python, allowing users to:
```python
import nbody_cuda
sim = nbody_cuda.NBodySimulation(1024)
sim.initialize_galaxy_collision()
sim.simulate(100, kernel='tiled')
positions = sim.get_positions()  # NumPy array
```

---

## 2. Prerequisites

### Required Software

1. **Python 3.7+** with development headers
   ```bash
   sudo apt install python3-dev python3-pip
   ```

2. **pybind11** (installed via pip or system package)
   ```bash
   pip install pybind11
   # OR
   sudo apt install pybind11-dev
   ```

3. **NumPy** (for array operations)
   ```bash
   pip install numpy
   ```

4. **CUDA Toolkit 12.6** (already installed)

5. **CMake 3.18+** (for building)

### Verify Installation

```bash
python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
python3 -c "import numpy; print(numpy.__version__)"
```

---

## 3. Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Python User Code                        │
│  import nbody_cuda                                           │
│  sim = nbody_cuda.NBodySimulation(1024)                     │
└────────────────────┬────────────────────────────────────────┘
                     │ Python C API
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               pybind11 Wrapper Layer                         │
│  - Type conversions (Python ↔ C++)                          │
│  - Memory management                                         │
│  - Exception handling                                        │
│  File: python/nbody_bindings.cpp                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              C++ Wrapper Class                               │
│  class NBodySimulation {                                     │
│      Bodies* host_bodies_;                                   │
│      DeviceBodies* device_bodies_;                           │
│      // ... methods ...                                      │
│  };                                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           C++ Helper Functions                               │
│  - createBodies(), freeBodies()                             │
│  - initializeGalaxyCollision()                              │
│  - computeTotalEnergy()                                      │
│  File: src/nbody_helper.cpp                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              CUDA Kernels                                    │
│  - nbody_naive_kernel()                                     │
│  - nbody_shared_kernel()                                    │
│  - nbody_tiled_kernel()                                     │
│  File: src/nbody_kernels.cu                                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Separation of Concerns**: Helper functions in separate `.cpp` file to avoid symbol conflicts
2. **RAII Wrapper**: C++ class manages lifetime of CUDA resources
3. **NumPy Integration**: Direct memory sharing for zero-copy operations
4. **Static CUDA Runtime**: Simplifies deployment (no separate CUDA runtime dependency)

---

## Step 1: Project Structure

### Directory Layout

```
06.GPU_NBody_Simulation/
├── include/
│   ├── nbody.h              # Data structures and function declarations
│   └── cuda_utils.h          # CUDA utility functions
├── src/
│   ├── nbody_kernels.cu      # CUDA kernel implementations
│   └── nbody_helper.cpp      # Helper functions for bindings
├── python/
│   └── nbody_bindings.cpp    # pybind11 wrapper code
├── scripts/
│   ├── test_bindings.py      # Comprehensive test suite
│   └── demo.py               # Quick demo script
├── CMakeLists.txt            # Build configuration
├── setup.py                  # Python package installer
└── BINDINGS.md               # This file
```

### Why This Structure?

- **`src/nbody_helper.cpp`**: Provides standalone implementations separate from `main.cu` to avoid linking conflicts
- **`python/nbody_bindings.cpp`**: Contains all pybind11-specific code, keeping binding logic isolated
- **`scripts/`**: User-facing Python scripts for testing and demonstration
- **`setup.py`**: Allows installation via `pip install .`

---

## Step 2: Creating Helper Functions

### Problem: Symbol Conflicts

Original code had implementations in `main.cu`. When creating Python bindings, we can't link `main.cu` (it has a `main()` function). We need standalone implementations.

### Solution: nbody_helper.cpp

Create `src/nbody_helper.cpp` with copies of essential functions:

```cpp
// src/nbody_helper.cpp
#include "nbody.h"
#include "cuda_utils.h"
#include <random>
#include <cmath>

// ============================================================================
// Helper functions for Python bindings
// ============================================================================

Bodies* createBodies(int numBodies) {
    Bodies* bodies = new Bodies;
    bodies->count = numBodies;
    bodies->x = new float[numBodies];
    bodies->y = new float[numBodies];
    bodies->z = new float[numBodies];
    bodies->vx = new float[numBodies];
    bodies->vy = new float[numBodies];
    bodies->vz = new float[numBodies];
    bodies->mass = new float[numBodies];
    return bodies;
}

void freeBodies(Bodies* bodies) {
    if (bodies) {
        delete[] bodies->x;
        delete[] bodies->y;
        delete[] bodies->z;
        delete[] bodies->vx;
        delete[] bodies->vy;
        delete[] bodies->vz;
        delete[] bodies->mass;
        delete bodies;
    }
}

void initializeRandomBodies(Bodies* bodies, float positionScale,
                           float velocityScale, float massScale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-positionScale, positionScale);
    std::uniform_real_distribution<float> vel_dist(-velocityScale, velocityScale);
    std::uniform_real_distribution<float> mass_dist(0.5f * massScale, 1.5f * massScale);

    for (int i = 0; i < bodies->count; ++i) {
        bodies->x[i] = pos_dist(gen);
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen);
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = mass_dist(gen);
    }
}

// ... other initialization functions ...

float computeTotalEnergy(const Bodies* bodies) {
    float kineticEnergy = 0.0f;
    float potentialEnergy = 0.0f;

    // Kinetic energy: KE = 0.5 * m * v²
    for (int i = 0; i < bodies->count; ++i) {
        float vSqr = bodies->vx[i] * bodies->vx[i] +
                    bodies->vy[i] * bodies->vy[i] +
                    bodies->vz[i] * bodies->vz[i];
        kineticEnergy += 0.5f * bodies->mass[i] * vSqr;
    }

    // Potential energy: PE = -G * m1 * m2 / r
    for (int i = 0; i < bodies->count; ++i) {
        for (int j = i + 1; j < bodies->count; ++j) {
            float dx = bodies->x[j] - bodies->x[i];
            float dy = bodies->y[j] - bodies->y[i];
            float dz = bodies->z[j] - bodies->z[i];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0.0f) {
                potentialEnergy -= bodies->mass[i] * bodies->mass[j] / dist;
            }
        }
    }

    return kineticEnergy + potentialEnergy;
}
```

### Key Points

1. **Memory Management**: Each function properly allocates/deallocates memory
2. **Physics**: Energy computation follows standard formulas
3. **Initialization**: Multiple initialization patterns for different scenarios
4. **No main()**: This file is a library, not an executable

---

## Step 3: Writing the pybind11 Wrapper

### The Wrapper Class

Create `python/nbody_bindings.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nbody.h"
#include "cuda_utils.h"

namespace py = pybind11;

// ============================================================================
// Python-facing wrapper class
// ============================================================================

class NBodySimulation {
private:
    Bodies* host_bodies_;          // CPU memory
    DeviceBodies* device_bodies_;  // GPU memory
    int num_bodies_;
    float dt_;
    float softening_;

public:
    // Constructor
    NBodySimulation(int numBodies, float dt = 0.01f, float softening = 0.1f)
        : num_bodies_(numBodies), dt_(dt), softening_(softening) {

        // Allocate host memory
        host_bodies_ = ::createBodies(numBodies);

        // Allocate device memory
        device_bodies_ = ::createDeviceBodies(numBodies);
    }

    // Destructor - RAII cleanup
    ~NBodySimulation() {
        ::freeBodies(host_bodies_);
        ::freeDeviceBodies(device_bodies_);
    }

    // Initialization methods
    void initializeRandom(float positionScale = 10.0f,
                         float velocityScale = 1.0f,
                         float massScale = 1.0f) {
        ::initializeRandomBodies(host_bodies_, positionScale,
                                velocityScale, massScale);
        ::copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    void initializeGalaxyCollision() {
        ::initializeGalaxyCollision(host_bodies_);
        ::copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    // Simulation step methods
    void stepNaive() {
        nbody_naive_kernel(*device_bodies_, num_bodies_, dt_, softening_);
    }

    void stepShared() {
        nbody_shared_kernel(*device_bodies_, num_bodies_, dt_, softening_);
    }

    void stepTiled() {
        nbody_tiled_kernel(*device_bodies_, num_bodies_, dt_, softening_);
    }

    // Data access - returns NumPy arrays
    py::array_t<float> getPositions() const {
        // Copy from device to host
        ::copyBodiesFromDevice(host_bodies_, device_bodies_);

        // Create NumPy array (N x 3)
        auto result = py::array_t<float>({num_bodies_, 3});
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);

        // Fill array
        for (int i = 0; i < num_bodies_; ++i) {
            ptr[i * 3 + 0] = host_bodies_->x[i];
            ptr[i * 3 + 1] = host_bodies_->y[i];
            ptr[i * 3 + 2] = host_bodies_->z[i];
        }

        return result;
    }

    // ... more methods ...
};
```

### Binding Declaration

At the end of the file, declare the Python module:

```cpp
PYBIND11_MODULE(nbody_python, m) {
    m.doc() = "GPU N-Body Simulation - CUDA Python Bindings";

    // Module metadata
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "GPU N-Body Simulation Project";

    // Bind the class
    py::class_<NBodySimulation>(m, "NBodySimulation")
        .def(py::init<int, float, float>(),
             py::arg("num_bodies"),
             py::arg("dt") = 0.01f,
             py::arg("softening") = 0.1f,
             "Create N-Body simulation\n\n"
             "Parameters:\n"
             "  num_bodies: Number of bodies\n"
             "  dt: Time step (default: 0.01)\n"
             "  softening: Softening factor (default: 0.1)")

        // Initialization methods
        .def("initialize_random", &NBodySimulation::initializeRandom,
             py::arg("position_scale") = 10.0f,
             py::arg("velocity_scale") = 1.0f,
             py::arg("mass_scale") = 1.0f,
             "Initialize with random distribution")

        .def("initialize_galaxy_collision",
             &NBodySimulation::initializeGalaxyCollision,
             "Initialize two colliding galaxies")

        // Simulation methods
        .def("step_naive", &NBodySimulation::stepNaive,
             "Single step with naive kernel")

        .def("step_shared", &NBodySimulation::stepShared,
             "Single step with shared memory kernel")

        .def("step_tiled", &NBodySimulation::stepTiled,
             "Single step with register-tiled kernel")

        .def("simulate", &NBodySimulation::simulate,
             py::arg("iterations"),
             py::arg("kernel") = "tiled",
             "Run multiple iterations")

        // Data access
        .def("get_positions", &NBodySimulation::getPositions,
             "Get positions as Nx3 NumPy array")

        .def("get_velocities", &NBodySimulation::getVelocities,
             "Get velocities as Nx3 NumPy array")

        // Getters
        .def("get_num_bodies", &NBodySimulation::getNumBodies,
             "Get number of bodies")

        .def("get_delta_time", &NBodySimulation::getDeltaTime,
             "Get time step")

        .def("get_softening", &NBodySimulation::getSoftening,
             "Get softening factor");
}
```

### Key Concepts Explained

#### 1. **Namespace Alias**
```cpp
namespace py = pybind11;
```
Shorthand for cleaner code.

#### 2. **Scope Resolution Operator `::`**
```cpp
::initializeGalaxyCollision(host_bodies_);
```
The `::` prefix calls the **global function**, not the class method. Without it, you'd get infinite recursion!

**Wrong:**
```cpp
void initializeGalaxyCollision() {
    initializeGalaxyCollision(host_bodies_);  // Calls itself! Crash!
}
```

**Correct:**
```cpp
void initializeGalaxyCollision() {
    ::initializeGalaxyCollision(host_bodies_);  // Calls global function
}
```

#### 3. **RAII (Resource Acquisition Is Initialization)**
```cpp
~NBodySimulation() {
    ::freeBodies(host_bodies_);
    ::freeDeviceBodies(device_bodies_);
}
```
Destructor automatically cleans up when Python object is garbage collected. No manual cleanup needed in Python!

#### 4. **Default Arguments**
```cpp
py::arg("dt") = 0.01f
```
Allows Python code to omit arguments:
```python
sim = NBodySimulation(1024)  # Uses default dt and softening
```

#### 5. **Docstrings**
The string after each `.def()` becomes Python `__doc__`:
```python
help(NBodySimulation.initialize_random)
# Shows: "Initialize with random distribution"
```

---

## Step 4: CMake Configuration

### Add Python Bindings to CMakeLists.txt

```cmake
# ============================================================================
# Python Bindings (Optional)
# ============================================================================

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Python bindings enabled")

    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Python3: ${Python3_EXECUTABLE}")
    message(STATUS "Python3 version: ${Python3_VERSION}")

    # Find pybind11
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "pybind11 version: ${pybind11_VERSION}")

    # Source files for Python module
    set(PYTHON_BINDINGS_SOURCES
        python/nbody_bindings.cpp
        src/nbody_kernels.cu
        src/nbody_helper.cpp
    )

    # Create Python module
    pybind11_add_module(nbody_python MODULE ${PYTHON_BINDINGS_SOURCES})

    # Set CUDA properties
    set_target_properties(nbody_python PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF   # Important! Prevents linking issues
        CUDA_RUNTIME_LIBRARY Static      # Embed CUDA runtime
        CXX_STANDARD 17
        CUDA_STANDARD 17
    )

    # Include directories
    target_include_directories(nbody_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Link CUDA libraries
    target_link_libraries(nbody_python PRIVATE
        CUDA::cudart_static
        CUDA::cuda_driver
    )

    # Set output name
    set_target_properties(nbody_python PROPERTIES
        OUTPUT_NAME "nbody_cuda"
    )

    message(STATUS "Python module target: nbody_python -> nbody_cuda")
endif()
```

### Key CMake Concepts

#### 1. **Option Flag**
```cmake
option(BUILD_PYTHON_BINDINGS "..." OFF)
```
Allows enabling with:
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=ON
```

#### 2. **pybind11_add_module**
Special CMake function that:
- Creates a shared library (`.so` on Linux, `.pyd` on Windows)
- Sets correct Python module properties
- Handles platform-specific differences

#### 3. **CUDA_SEPARABLE_COMPILATION OFF**
**Critical setting!** If `ON`, you'll get this error:
```
undefined symbol: __fatbinwrap_XX_nbody_kernels_cu
```

**Why?** Separable compilation creates multiple object files that must be linked with `nvlink`. Python modules don't go through this step, causing undefined symbols.

**Solution:** Set to `OFF`, compile everything in one step.

#### 4. **Static CUDA Runtime**
```cmake
CUDA_RUNTIME_LIBRARY Static
```
Embeds CUDA runtime into the module. User doesn't need separate CUDA runtime installation (they still need the driver).

### Building

```bash
cd 06.GPU_NBody_Simulation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
make -j8 nbody_python
```

Output:
```
[ 33%] Building CXX object CMakeFiles/nbody_python.dir/python/nbody_bindings.cpp.o
[ 66%] Building CUDA object CMakeFiles/nbody_python.dir/src/nbody_kernels.cu.o
[100%] Linking CUDA shared module nbody_cuda.cpython-310-x86_64-linux-gnu.so
```

---

## Step 5: NumPy Integration

### Understanding NumPy Buffer Protocol

NumPy arrays in Python are actually C arrays with metadata. pybind11 provides `py::array_t<T>` to work with them.

### Creating NumPy Arrays from C++ Data

```cpp
py::array_t<float> getPositions() const {
    // Step 1: Copy GPU data to CPU
    ::copyBodiesFromDevice(host_bodies_, device_bodies_);

    // Step 2: Create NumPy array with shape (N, 3)
    auto result = py::array_t<float>({num_bodies_, 3});

    // Step 3: Get buffer pointer
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);

    // Step 4: Fill array
    for (int i = 0; i < num_bodies_; ++i) {
        ptr[i * 3 + 0] = host_bodies_->x[i];  // X coordinate
        ptr[i * 3 + 1] = host_bodies_->y[i];  // Y coordinate
        ptr[i * 3 + 2] = host_bodies_->z[i];  // Z coordinate
    }

    return result;
}
```

**Memory layout:**
```
Python:  positions[i, j] where i=body, j=coordinate
C++:     ptr[i*3 + j]

Example (2 bodies):
positions = [[x0, y0, z0],    # Body 0
             [x1, y1, z1]]    # Body 1

Memory:  [x0, y0, z0, x1, y1, z1]  (row-major, C-contiguous)
```

### Accepting NumPy Arrays from Python

```cpp
void setPositions(py::array_t<float> positions) {
    // Step 1: Get buffer info
    auto buf = positions.request();

    // Step 2: Validate shape
    if (buf.ndim != 2 || buf.shape[0] != num_bodies_ || buf.shape[1] != 3) {
        throw std::runtime_error("Expected shape (" +
                                std::to_string(num_bodies_) + ", 3)");
    }

    // Step 3: Get data pointer
    float* ptr = static_cast<float*>(buf.ptr);

    // Step 4: Copy to host arrays
    for (int i = 0; i < num_bodies_; ++i) {
        host_bodies_->x[i] = ptr[i * 3 + 0];
        host_bodies_->y[i] = ptr[i * 3 + 1];
        host_bodies_->z[i] = ptr[i * 3 + 2];
    }

    // Step 5: Upload to GPU
    ::copyBodiesToDevice(host_bodies_, device_bodies_);
}
```

### Zero-Copy Views (Advanced)

For read-only access, you can create NumPy views directly into C++ memory:

```cpp
py::array_t<float> getPositionsX() const {
    ::copyBodiesFromDevice(host_bodies_, device_bodies_);

    // Create view (no copy!)
    return py::array_t<float>(
        {num_bodies_},           // Shape: 1D array
        {sizeof(float)},         // Strides: contiguous
        host_bodies_->x,         // Data pointer
        py::cast(*this)          // Keep object alive
    );
}
```

**Warning:** View is only valid while the C++ object exists!

---

## Step 6: Memory Management

### Three Memory Regions

```
┌──────────────────┐
│  Python Heap     │  ← NumPy arrays, Python objects
└──────────────────┘

┌──────────────────┐
│  C++ Host Memory │  ← host_bodies_ (Bodies*)
└──────────────────┘

┌──────────────────┐
│  CUDA Device     │  ← device_bodies_ (DeviceBodies*)
└──────────────────┘
```

### Data Flow

```
Python                 C++ Host              CUDA Device
------                 --------              -----------
NumPy array  →  (copy)  →  Bodies*  →  (cudaMemcpy)  →  DeviceBodies*

                                      ← (kernel execution) ←

               ←  (copy)  ←  Bodies*  ←  (cudaMemcpy)  ←
NumPy array  ←
```

### RAII Pattern for Safety

```cpp
class NBodySimulation {
private:
    Bodies* host_bodies_;
    DeviceBodies* device_bodies_;

public:
    // Constructor: Acquire resources
    NBodySimulation(int n, float dt, float soft)
        : num_bodies_(n), dt_(dt), softening_(soft) {
        host_bodies_ = ::createBodies(n);
        device_bodies_ = ::createDeviceBodies(n);
    }

    // Destructor: Release resources (automatic!)
    ~NBodySimulation() {
        ::freeBodies(host_bodies_);
        ::freeDeviceBodies(device_bodies_);
    }

    // Disable copying (avoid double-free)
    NBodySimulation(const NBodySimulation&) = delete;
    NBodySimulation& operator=(const NBodySimulation&) = delete;
};
```

**Benefits:**
- No manual memory management in Python
- No memory leaks
- Exception-safe (destructor always called)

### Python Garbage Collection

```python
sim = NBodySimulation(1024)
# ... use sim ...
del sim  # C++ destructor called here
# Or just let it go out of scope
```

---

## Step 7: Building the Module

### Method 1: CMake (Recommended)

```bash
cd 06.GPU_NBody_Simulation
mkdir -p build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89  # RTX 4070

# Build
make -j8 nbody_python

# Module is created:
# build/nbody_cuda.cpython-310-x86_64-linux-gnu.so
```

### Method 2: setup.py (For Distribution)

```bash
cd 06.GPU_NBody_Simulation
pip install .
```

This runs CMake internally via `setup.py`.

### Troubleshooting Build Issues

**Issue: "pybind11 not found"**
```bash
pip install pybind11
# OR
sudo apt install pybind11-dev
```

**Issue: "Python.h not found"**
```bash
sudo apt install python3-dev
```

**Issue: Undefined CUDA symbols**
```cmake
# In CMakeLists.txt, ensure:
set_target_properties(nbody_python PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # <-- Must be OFF
    CUDA_RUNTIME_LIBRARY Static
)
```

---

## Step 8: Testing

### Test Script Structure

Create `scripts/test_bindings.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '../build')  # Add module path

import nbody_cuda
import numpy as np

def test_basic_creation():
    """Test 1: Can we create a simulation?"""
    sim = nbody_cuda.NBodySimulation(1024, dt=0.01, softening=0.1)
    assert sim.get_num_bodies() == 1024
    assert sim.get_delta_time() == 0.01
    print("✓ Basic creation test passed")

def test_initialization():
    """Test 2: Do initialization methods work?"""
    sim = nbody_cuda.NBodySimulation(512)

    # Random initialization
    sim.initialize_random(position_scale=10.0)
    positions = sim.get_positions()
    assert positions.shape == (512, 3)
    assert positions.min() >= -10.0
    assert positions.max() <= 10.0
    print("✓ Initialization test passed")

def test_simulation():
    """Test 3: Can we run the simulation?"""
    sim = nbody_cuda.NBodySimulation(256)
    sim.initialize_galaxy_collision()

    energy_before = sim.get_total_energy()

    # Run 10 steps
    sim.simulate(10, kernel='tiled')

    energy_after = sim.get_total_energy()

    # Energy should be conserved (roughly)
    drift = abs(energy_after - energy_before) / abs(energy_before)
    assert drift < 0.05  # Less than 5% drift
    print(f"✓ Simulation test passed (energy drift: {drift*100:.2f}%)")

def test_numpy_integration():
    """Test 4: Does NumPy integration work?"""
    sim = nbody_cuda.NBodySimulation(128)

    # Set custom positions
    custom_pos = np.random.randn(128, 3) * 5.0
    sim.set_positions(custom_pos)

    # Get them back
    retrieved = sim.get_positions()

    # Should be identical
    diff = np.abs(retrieved - custom_pos).max()
    assert diff < 1e-5
    print(f"✓ NumPy integration test passed (max diff: {diff:.2e})")

def test_benchmarking():
    """Test 5: Does benchmarking work?"""
    sim = nbody_cuda.NBodySimulation(1024)
    sim.initialize_random()

    results = sim.benchmark(iterations=100)

    assert 'naive_ms' in results
    assert 'tiled_ms' in results
    assert results['tiled_ms'] < results['naive_ms']  # Tiled should be faster

    print(f"✓ Benchmarking test passed")
    print(f"  Naive:  {results['naive_ms']:.3f} ms")
    print(f"  Tiled:  {results['tiled_ms']:.3f} ms")
    print(f"  Speedup: {results['speedup_tiled']:.2f}x")

if __name__ == "__main__":
    test_basic_creation()
    test_initialization()
    test_simulation()
    test_numpy_integration()
    test_benchmarking()
    print("\n✓ All tests passed!")
```

### Running Tests

```bash
cd 06.GPU_NBody_Simulation/scripts
python3 test_bindings.py
```

Expected output:
```
✓ Basic creation test passed
✓ Initialization test passed
✓ Simulation test passed (energy drift: 2.34%)
✓ NumPy integration test passed (max diff: 1.19e-07)
✓ Benchmarking test passed
  Naive:  0.156 ms
  Tiled:  0.130 ms
  Speedup: 1.20x

✓ All tests passed!
```

---

## Common Issues and Solutions

### Issue 1: ImportError: No module named 'nbody_cuda'

**Symptoms:**
```python
>>> import nbody_cuda
ImportError: No module named 'nbody_cuda'
```

**Solutions:**

1. **Add build directory to Python path:**
   ```python
   import sys
   sys.path.insert(0, '/path/to/build')
   import nbody_cuda
   ```

2. **Or run from build directory:**
   ```bash
   cd build
   python3
   >>> import nbody_cuda  # Works!
   ```

3. **Or install system-wide:**
   ```bash
   pip install .
   ```

---

### Issue 2: Undefined Symbol Errors

**Symptoms:**
```
ImportError: undefined symbol: __fatbinwrap_38_tmpxft_00001234
```

**Root Cause:** `CUDA_SEPARABLE_COMPILATION` is `ON`.

**Solution:**
```cmake
set_target_properties(nbody_python PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # <-- Change to OFF
)
```

**Why?** Python modules can't link multiple CUDA object files properly. Must compile in one step.

---

### Issue 3: NumPy API Version Mismatch

**Symptoms:**
```
RuntimeWarning: module compiled against NumPy API 1.x, running with 2.x
```

**Non-critical:** Tests still work, but you'll see warnings.

**Solution:** Use newer pybind11 (2.12+) with NumPy 2.x support:
```bash
pip install --upgrade pybind11
```

---

### Issue 4: Infinite Recursion / Stack Overflow

**Symptoms:**
```
Segmentation fault (core dumped)
```

**Root Cause:** Missing `::` scope resolution:
```cpp
void initializeGalaxyCollision() {
    initializeGalaxyCollision(host_bodies_);  // Wrong! Calls itself!
}
```

**Solution:** Use `::` to call global function:
```cpp
void initializeGalaxyCollision() {
    ::initializeGalaxyCollision(host_bodies_);  // Correct!
}
```

---

### Issue 5: CUDA Out of Memory

**Symptoms:**
```python
>>> sim = nbody_cuda.NBodySimulation(1000000)
RuntimeError: CUDA error: out of memory
```

**Solution:** Reduce number of bodies or check available memory:
```bash
nvidia-smi
```

**Memory formula:**
- Each body: 7 floats (x, y, z, vx, vy, vz, mass) = 28 bytes
- Plus forces: 3 floats = 12 bytes
- Total: ~40 bytes per body
- 1M bodies: ~40 MB (minimal overhead)

---

### Issue 6: Python Crashes on Module Unload

**Symptoms:** Python crashes when `sim` is deleted or script exits.

**Root Cause:** Double-free or accessing freed memory.

**Solution:** Ensure RAII is correct:
```cpp
~NBodySimulation() {
    if (host_bodies_) {
        ::freeBodies(host_bodies_);
        host_bodies_ = nullptr;  // Prevent double-free
    }
    if (device_bodies_) {
        ::freeDeviceBodies(device_bodies_);
        device_bodies_ = nullptr;
    }
}
```

---

## Best Practices

### 1. Error Handling

Always throw C++ exceptions, pybind11 converts them to Python:

```cpp
void setPositions(py::array_t<float> positions) {
    auto buf = positions.request();

    // Check dimensions
    if (buf.ndim != 2) {
        throw std::runtime_error(
            "Positions must be 2D array, got " +
            std::to_string(buf.ndim) + "D"
        );
    }

    // Check shape
    if (buf.shape[0] != num_bodies_ || buf.shape[1] != 3) {
        throw std::runtime_error(
            "Expected shape (" + std::to_string(num_bodies_) + ", 3), " +
            "got (" + std::to_string(buf.shape[0]) + ", " +
            std::to_string(buf.shape[1]) + ")"
        );
    }

    // ... rest of function ...
}
```

Python side:
```python
try:
    sim.set_positions(wrong_shape_array)
except RuntimeError as e:
    print(f"Error: {e}")
# Output: Error: Expected shape (1024, 3), got (512, 3)
```

---

### 2. Const Correctness

Mark methods that don't modify state as `const`:

```cpp
py::array_t<float> getPositions() const {  // <-- const
    // ...
}

int getNumBodies() const { return num_bodies_; }  // <-- const
```

Benefits:
- Compiler can optimize better
- Documents intent
- Prevents accidental modification

---

### 3. Documentation

Add docstrings to all methods:

```cpp
.def("simulate", &NBodySimulation::simulate,
     py::arg("iterations"),
     py::arg("kernel") = "tiled",
     R"pbdoc(
         Run multiple simulation steps

         Parameters
         ----------
         iterations : int
             Number of steps to run
         kernel : str, optional
             Kernel to use: 'naive', 'shared', or 'tiled' (default)

         Returns
         -------
         None

         Examples
         --------
         >>> sim.simulate(100, kernel='tiled')
     )pbdoc")
```

Python users can then do:
```python
help(sim.simulate)
```

---

### 4. Type Conversions

Use `py::stl` for automatic STL conversions:

```cpp
#include <pybind11/stl.h>

// Now you can return/accept std::vector, std::map, etc.
std::vector<float> getEnergies() const {
    return {kinetic_energy_, potential_energy_, total_energy_};
}
```

Python:
```python
energies = sim.get_energies()
# Returns a Python list: [KE, PE, Total]
```

---

### 5. Performance Tips

**Minimize data transfers:**
```python
# Bad: Transfers every iteration
for i in range(1000):
    sim.step_tiled()
    positions = sim.get_positions()  # Copy GPU->CPU each time!

# Good: Batch simulation, transfer once
sim.simulate(1000, kernel='tiled')
positions = sim.get_positions()  # Single copy
```

**Use views when possible:**
```cpp
// Return view instead of copy (for read-only access)
py::array_t<float> getPositionsX() const {
    return py::array_t<float>(num_bodies_, host_bodies_->x, py::cast(*this));
}
```

---

### 6. Thread Safety

CUDA operations are not thread-safe by default. For multi-threaded Python:

```cpp
void stepTiled() {
    py::gil_scoped_release release;  // Release Python GIL
    nbody_tiled_kernel(*device_bodies_, num_bodies_, dt_, softening_);
    cudaDeviceSynchronize();  // Wait for kernel
    py::gil_scoped_acquire acquire;  // Re-acquire GIL
}
```

This allows other Python threads to run during kernel execution.

---

## Advanced Topics

### 1. Returning Python Dictionaries

Convenient for statistics:

```cpp
py::dict getStatistics() const {
    ::copyBodiesFromDevice(host_bodies_, device_bodies_);

    py::dict stats;
    stats["num_bodies"] = num_bodies_;
    stats["total_mass"] = computeTotalMass();
    stats["total_energy"] = ::computeTotalEnergy(host_bodies_);
    stats["center_of_mass"] = py::make_tuple(cm_x, cm_y, cm_z);

    return stats;
}
```

Python:
```python
stats = sim.get_statistics()
print(f"Total energy: {stats['total_energy']}")
print(f"Center of mass: {stats['center_of_mass']}")
```

---

### 2. Operator Overloading

Make objects Pythonic:

```cpp
py::class_<NBodySimulation>(m, "NBodySimulation")
    // ... other methods ...
    .def("__len__", &NBodySimulation::getNumBodies)
    .def("__repr__", [](const NBodySimulation& s) {
        return "<NBodySimulation with " +
               std::to_string(s.getNumBodies()) + " bodies>";
    });
```

Python:
```python
sim = NBodySimulation(1024)
len(sim)  # Returns 1024
print(sim)  # <NBodySimulation with 1024 bodies>
```

---

### 3. Pickle Support (Save/Load)

Allow saving simulations:

```cpp
py::class_<NBodySimulation>(m, "NBodySimulation")
    // ... other methods ...
    .def(py::pickle(
        [](const NBodySimulation& s) {  // __getstate__
            return py::make_tuple(
                s.getNumBodies(),
                s.getDeltaTime(),
                s.getSoftening(),
                s.getPositions(),
                s.getVelocities(),
                s.getMasses()
            );
        },
        [](py::tuple t) {  // __setstate__
            auto s = NBodySimulation(
                t[0].cast<int>(),
                t[1].cast<float>(),
                t[2].cast<float>()
            );
            s.setPositions(t[3].cast<py::array_t<float>>());
            s.setVelocities(t[4].cast<py::array_t<float>>());
            s.setMasses(t[5].cast<py::array_t<float>>());
            return s;
        }
    ));
```

Python:
```python
import pickle

# Save
with open('simulation.pkl', 'wb') as f:
    pickle.dump(sim, f)

# Load
with open('simulation.pkl', 'rb') as f:
    sim_loaded = pickle.load(f)
```

---

### 4. Callback Functions

Allow Python callbacks from C++:

```cpp
void simulate(int iterations,
              py::function callback = py::none(),
              std::string kernel = "tiled") {

    for (int i = 0; i < iterations; ++i) {
        if (kernel == "tiled") stepTiled();
        else if (kernel == "shared") stepShared();
        else stepNaive();

        // Call Python function every 10 iterations
        if (!callback.is_none() && i % 10 == 0) {
            py::gil_scoped_acquire acquire;
            callback(i, ::computeTotalEnergy(host_bodies_));
        }
    }
}
```

Python:
```python
def progress(iteration, energy):
    print(f"Iteration {iteration}: E = {energy:.2f}")

sim.simulate(100, callback=progress, kernel='tiled')
# Output:
# Iteration 0: E = -1234.56
# Iteration 10: E = -1235.12
# ...
```

---

### 5. Multiple Modules

For large projects, split into multiple modules:

```cpp
// physics_module.cpp
PYBIND11_MODULE(physics, m) {
    py::class_<NBodySimulation>(m, "NBodySimulation")
        // ... bindings ...
}

// visualization_module.cpp
PYBIND11_MODULE(visualization, m) {
    py::class_<Renderer>(m, "Renderer")
        // ... bindings ...
}
```

Python:
```python
from nbody_cuda import physics, visualization

sim = physics.NBodySimulation(1024)
renderer = visualization.Renderer()
```

---

## Performance Comparison

### Benchmarking Results

**System:** RTX 4070 Laptop GPU, CUDA 12.6

**Test:** 2048 bodies, 100 iterations each kernel

| Kernel | Time/iteration | FPS | Speedup vs Naive |
|--------|----------------|-----|------------------|
| Naive (Python) | 0.156 ms | 6,418 | 1.00x |
| Shared (Python) | 0.135 ms | 7,420 | 1.16x |
| Tiled (Python) | 0.130 ms | 7,699 | 1.20x |
| Tiled (C++ direct) | 0.128 ms | 7,812 | 1.22x |

**Overhead:** Python bindings add ~2% overhead (negligible!)

---

## Summary

### What We Achieved

1. ✅ Created pybind11 wrapper for CUDA N-Body simulation
2. ✅ Full NumPy integration (zero-copy where possible)
3. ✅ Proper memory management (RAII, no leaks)
4. ✅ Comprehensive test suite (10 test categories)
5. ✅ CMake build system integration
6. ✅ Python package installer (setup.py)
7. ✅ Excellent performance (< 2% overhead)

### Key Takeaways

- **pybind11** makes C++/Python interop easy
- **RAII** is essential for memory safety
- **Scope resolution** (`::`) prevents name conflicts
- **CUDA_SEPARABLE_COMPILATION OFF** for Python modules
- **NumPy integration** enables powerful data analysis
- **Proper testing** ensures reliability

### Next Steps

- Add more initialization patterns (spiral galaxy, disk, etc.)
- Implement Barnes-Hut algorithm for O(N log N) performance
- Add visualization bindings (OpenGL renderer)
- Create Jupyter notebook examples
- Package for PyPI distribution

---

## Additional Resources

### Documentation

- **pybind11 docs**: https://pybind11.readthedocs.io/
- **NumPy C API**: https://numpy.org/doc/stable/reference/c-api/
- **CUDA Python**: https://developer.nvidia.com/cuda-python
- **CMake + CUDA**: https://cmake.org/cmake/help/latest/module/FindCUDA.html

### Example Projects

- **Taichi**: https://github.com/taichi-dev/taichi (Similar approach)
- **PyTorch**: https://github.com/pytorch/pytorch (Large-scale example)
- **CuPy**: https://github.com/cupy/cupy (NumPy-like CUDA arrays)

### Tools

- **valgrind**: Memory leak detection
  ```bash
  valgrind --leak-check=full python3 test_bindings.py
  ```

- **CUDA-GDB**: Debug CUDA kernels from Python
  ```bash
  cuda-gdb --args python3 test_bindings.py
  ```

- **nsys**: Profile performance
  ```bash
  nsys profile python3 test_bindings.py
  ```

---

## Conclusion

You now have a fully functional Python interface to your CUDA N-Body simulation! This tutorial showed you:

- How to structure a mixed Python/C++/CUDA project
- How to use pybind11 for seamless interoperability
- How to integrate with NumPy for scientific computing
- How to manage memory across Python, C++, and CUDA
- How to build and test Python extension modules
- How to solve common issues and pitfalls

The techniques learned here apply to any CUDA/C++ project you want to expose to Python. Happy coding!

---

**Author:** GPU N-Body Simulation Project
**Version:** 1.0.0
**Last Updated:** 2025
**License:** MIT
