#!/bin/bash

# Build script for CUDA Real-Time Video Stabilization
# Target: NVIDIA RTX 4070 (sm_89, CUDA 12.6)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"

echo "========================================"
echo "CUDA Video Stabilization Build Script"
echo "========================================"
echo "Build Type: ${BUILD_TYPE}"
echo "Build Directory: ${BUILD_DIR}"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please install CUDA Toolkit or add it to PATH"
    exit 1
fi

# Print CUDA version
echo "CUDA Version:"
nvcc --version | grep "release"
echo ""

# Check for OpenCV
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV4 not found!"
    echo "Please install OpenCV4 development packages"
    exit 1
fi

echo "OpenCV Version: $(pkg-config --modversion opencv4)"
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Executable: ${BUILD_DIR}/video_stabilizer"
echo ""
echo "Usage examples:"
echo "  # Process video file:"
echo "  ./video_stabilizer -i input.mp4 -o output.mp4"
echo ""
echo "  # Real-time from webcam:"
echo "  ./video_stabilizer -c 0 -p"
echo ""
echo "  # With custom smoothing:"
echo "  ./video_stabilizer -i input.mp4 -s 50 -r 0.85"
echo ""
echo "Run './video_stabilizer -h' for all options"
