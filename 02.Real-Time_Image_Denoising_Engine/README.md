# Real-Time Image Denoising Engine

[![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-blue.svg)](https://opencv.org/)
[![CMake](https://img.shields.io/badge/CMake-3.18+-red.svg)](https://cmake.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-yellow.svg)](https://python.org/)

A high-performance, GPU-accelerated image denoising engine designed for real-time applications. This project implements advanced denoising algorithms using CUDA for maximum performance, with both C++ and Python interfaces.

## 🚀 Features

### Core Algorithms
- **Bilateral Filter**: Edge-preserving smoothing with spatial and color distance weighting
- **Non-Local Means (NLM)**: Advanced patch-based denoising with similarity search
- **Gaussian Filter**: Classical smoothing with separable implementation
- **Adaptive Bilateral**: Dynamic parameter adjustment based on local image characteristics

### Performance Optimizations
- **CUDA Kernels**: Custom GPU kernels for maximum performance
- **Shared Memory**: Optimized memory access patterns
- **Memory Pooling**: Efficient GPU memory management
- **Async Processing**: Non-blocking operations with CUDA streams
- **Separable Filters**: Reduced computational complexity

### Real-Time Capabilities
- **30+ FPS** on 1080p video streams
- **Low Latency**: Sub-10ms processing on modern GPUs
- **Camera Integration**: Direct camera feed processing
- **Queue Management**: Frame dropping and buffering strategies

### Interfaces
- **C++ API**: High-performance native interface
- **Python API**: Easy-to-use Python wrapper
- **GUI Application**: Interactive parameter tuning
- **Command Line**: Batch processing and automation

## 📋 Requirements

### Hardware
- NVIDIA GPU with Compute Capability 6.0+ (Pascal, Turing, Ampere, Ada Lovelace)
- 4GB+ GPU memory (8GB recommended for 4K processing)
- Modern CPU with OpenMP support

### Software
- **CUDA Toolkit**: 11.0+ (12.4+ recommended)
- **OpenCV**: 4.0+ with CUDA support
- **CMake**: 3.18+
- **Python**: 3.7+ (for Python interface)
- **C++ Compiler**: GCC 9+, MSVC 2019+, or Clang 10+

### Tested Platforms
- Ubuntu 20.04/22.04 LTS
- Windows 10/11
- CentOS 8/RHEL 8

## 🛠️ Installation

### 1. Install Dependencies

#### Ubuntu/Debian
```bash
# Update package manager
sudo apt update

# Install build essentials
sudo apt install build-essential cmake git

# Install OpenCV with CUDA support
sudo apt install libopencv-dev libopencv-contrib-dev

# Install Python dependencies
sudo apt install python3-dev python3-pip
pip3 install -r requirements.txt
```

#### Windows
```powershell
# Install Visual Studio 2019+ with C++ support
# Install CUDA Toolkit from NVIDIA website
# Install OpenCV (pre-built binaries or compile from source)

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Build the Project

```bash
# Clone the repository
git clone https://github.com/your-username/real-time-image-denoising.git
cd real-time-image-denoising

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES="70;75;80;86"

# Build (use -j$(nproc) for parallel compilation)
make -j$(nproc)

# Optional: Install system-wide
sudo make install
```

### 3. Verify Installation

```bash
# Test the C++ application
./image_denoiser --help

# Test Python interface
cd ../python
python3 denoising_pipeline.py --help

# Run benchmark
./image_denoiser -b
```

## 🎯 Quick Start

### C++ Interface

```cpp
#include "cuda_denoiser.h"
#include <opencv2/opencv.hpp>

using namespace rtid;

int main() {
    // Load image
    cv::Mat input = cv::imread("noisy_image.jpg");
    cv::Mat output;
    
    // Create denoiser
    CudaDenoiser denoiser(input.cols, input.rows);
    
    // Set parameters
    DenoiseParams params;
    params.sigma_color = 50.0f;
    params.sigma_space = 50.0f;
    
    // Denoise image
    bool success = denoiser.denoise(input, output, 
                                   DenoiseAlgorithm::BILATERAL, params);
    
    if (success) {
        cv::imwrite("denoised_image.jpg", output);
        denoiser.printPerformanceStats();
    }
    
    return 0;
}
```

### Python Interface

```python
from python.denoising_pipeline import DenoisingPipeline, DenoiseAlgorithm, DenoiseParams
import cv2

# Create pipeline
pipeline = DenoisingPipeline()

# Load and process image
image = cv2.imread("noisy_image.jpg")
result = pipeline.denoise_image(image)

# Save result
cv2.imwrite("denoised_image.jpg", (result * 255).astype(np.uint8))

# Print statistics
stats = pipeline.get_statistics()
print(f"Processing time: {stats.avg_latency_ms:.2f}ms")
```

### Real-Time Camera Processing

```bash
# C++ version
./image_denoiser -c -a bilateral --sigma-color 50 --sigma-space 50

# Python version  
python3 denoising_pipeline.py --camera --algorithm bilateral

# GUI version
python3 denoising_pipeline.py --gui
```

## 📊 Usage Examples

### Single Image Processing

```bash
# Process single image with bilateral filter
./image_denoiser -i input.jpg -o output.jpg -a bilateral

# Process with Non-Local Means
./image_denoiser -i input.jpg -a nlm --h-param 15 --template-size 7

# Process with custom parameters
./image_denoiser -i input.jpg -a adaptive --sigma-color 30 --sigma-space 40
```

### Batch Processing

```python
from python.denoising_pipeline import DenoisingPipeline
import os
import cv2

pipeline = DenoisingPipeline()

# Process all images in directory
input_dir = "noisy_images/"
output_dir = "denoised_images/"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.bmp')):
        image = cv2.imread(os.path.join(input_dir, filename))
        result = pipeline.denoise_image(image)
        
        output_path = os.path.join(output_dir, f"denoised_{filename}")
        cv2.imwrite(output_path, (result * 255).astype(np.uint8))
```

### Real-Time Video Processing

```python
# Process video file
pipeline = DenoisingPipeline()
cap = cv2.VideoCapture("input_video.mp4")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    denoised = pipeline.denoise_image(frame)
    denoised_8bit = (denoised * 255).astype(np.uint8)
    
    out.write(denoised_8bit)
    cv2.imshow('Denoised', denoised_8bit)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## ⚙️ Algorithm Parameters

### Bilateral Filter
- `sigma_color` (1-100): Color similarity threshold. Higher values smooth more aggressively.
- `sigma_space` (1-100): Spatial distance threshold. Higher values consider more distant pixels.
- `kernel_size` (3-15): Filter kernel size. Must be odd.

### Non-Local Means
- `h` (1-50): Filtering strength. Higher h removes more noise but removes details too.
- `template_window_size` (3-11): Size of template patch. Should be odd.
- `search_window_size` (7-35): Size of search area. Should be odd and larger than template.

### Gaussian Filter
- `gaussian_sigma` (0.1-10): Standard deviation. Higher values blur more.
- `kernel_size` (3-15): Filter kernel size. Must be odd.

### Adaptive Bilateral
- Uses bilateral parameters but adapts them based on local image characteristics
- Automatically adjusts smoothing strength in different regions

## 📈 Performance Benchmarks

### RTX 4070 (Test System)
| Algorithm | 1080p FPS | 4K FPS | Memory Usage |
|-----------|-----------|--------|--------------|
| Bilateral | 45-60 | 12-15 | ~150MB |
| Gaussian | 80-100 | 20-25 | ~100MB |
| NLM (Fast) | 25-35 | 6-8 | ~200MB |
| Adaptive | 40-55 | 10-12 | ~180MB |

### RTX 3080
| Algorithm | 1080p FPS | 4K FPS | Memory Usage |
|-----------|-----------|--------|--------------|
| Bilateral | 60-80 | 15-20 | ~150MB |
| Gaussian | 120-150 | 30-35 | ~100MB |
| NLM (Fast) | 35-45 | 8-12 | ~200MB |
| Adaptive | 55-70 | 12-16 | ~180MB |

*Note: Performance varies based on image content, parameters, and system configuration.*

## 🔧 Advanced Configuration

### Memory Pool Settings

```cpp
// Reserve 2GB for memory pool
auto& memory_pool = GpuMemoryPool::getInstance();
memory_pool.reserve(2ULL * 1024 * 1024 * 1024);
```

### Custom CUDA Streams

```cpp
CudaStream stream1, stream2;
denoiser.denoiseAsync(input1, output1, algorithm, params, stream1.get());
denoiser.denoiseAsync(input2, output2, algorithm, params, stream2.get());

// Synchronize both streams
stream1.synchronize();
stream2.synchronize();
```

### Multi-GPU Support

```cpp
// Set device for processing
cudaSetDevice(1);
CudaDenoiser denoiser_gpu1;

cudaSetDevice(2);  
CudaDenoiser denoiser_gpu2;
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce image resolution or batch size
# Check available GPU memory: nvidia-smi
# Reduce memory pool size in initialization
```

**2. Low Performance**
```bash
# Ensure GPU is being used: nvidia-smi
# Check CUDA architecture compatibility
# Verify OpenCV was built with CUDA support
```

**3. Build Errors**
```bash
# Update CUDA toolkit and drivers
# Check CMake CUDA architecture settings
# Verify all dependencies are installed
```

### Performance Tuning

**1. Optimize for Your GPU**
```cmake
# Set specific architecture in CMakeLists.txt
set(CMAKE_CUDA_ARCHITECTURES "86")  # For RTX 30xx/40xx
```

**2. Adjust Thread Block Sizes**
```cpp
// In CUDA kernels, experiment with different block sizes
dim3 block_size(32, 32);  // May be better than 16x16 for some GPUs
```

**3. Memory Access Patterns**
```cpp
// Use shared memory for frequently accessed data
// Coalesce global memory access
// Minimize memory transfers between host and device
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
cd build && ctest

# Run Python tests
cd python && python -m pytest

# Format code
black python/
clang-format -i src/**/*.cpp src/**/*.cu include/**/*.h
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- OpenCV community for computer vision algorithms
- Contributors and testers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/real-time-image-denoising/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/real-time-image-denoising/discussions)
- **Email**: support@yourproject.com

## 🔗 Related Projects

- [OpenCV CUDA Modules](https://docs.opencv.org/master/d1/d1a/group__cuda.html)
- [NVIDIA NPP](https://docs.nvidia.com/cuda/npp/index.html)
- [DnCNN PyTorch](https://github.com/SaoYan/DnCNN-PyTorch)

---
