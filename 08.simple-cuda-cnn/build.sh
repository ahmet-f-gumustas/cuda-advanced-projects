#!/bin/bash
# Build script for Simple CUDA CNN

set -e  # Exit on error

echo "========================================="
echo "  Simple CUDA CNN - Build Script"
echo "========================================="

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA (nvcc) not found!"
    echo "Please install CUDA Toolkit first."
    exit 1
fi

echo "CUDA Version:"
nvcc --version | grep "release"

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Run CMake
echo ""
echo "Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building project..."
make -j$(nproc)

# Check output
echo ""
echo "Build complete!"
ls -lh libcnn_cuda.so

echo ""
echo "========================================="
echo "  Build successful!"
echo "  Library: build/libcnn_cuda.so"
echo ""
echo "  Run demo:"
echo "    python3 python/demo.py"
echo "========================================="
