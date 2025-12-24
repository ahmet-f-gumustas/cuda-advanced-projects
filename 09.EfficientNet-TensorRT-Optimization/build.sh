#!/bin/bash

# EfficientNet TensorRT Build Script
# Builds the project with TensorRT support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EfficientNet TensorRT Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
BUILD_TYPE="Release"
CUDA_ARCH="89"  # RTX 4070
TENSORRT_ROOT=""
CLEAN_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --tensorrt)
            TENSORRT_ROOT="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug           Build in debug mode"
            echo "  --cuda-arch <arch>  CUDA architecture (default: 89 for RTX 4070)"
            echo "  --tensorrt <path> TensorRT installation path"
            echo "  --clean           Clean build directory first"
            echo "  --help            Show this help"
            echo ""
            echo "Common CUDA architectures:"
            echo "  75 - Turing (RTX 2000 series)"
            echo "  80 - Ampere (RTX 3000 series, A100)"
            echo "  86 - Ampere (RTX 3000 desktop)"
            echo "  87 - Jetson Orin"
            echo "  89 - Ada Lovelace (RTX 4000 series)"
            echo "  90 - Hopper (H100)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
echo -e "CUDA version: ${GREEN}$CUDA_VERSION${NC}"

# Check for TensorRT
if [ -z "$TENSORRT_ROOT" ]; then
    # Try to find TensorRT automatically
    if [ -d "/usr/include/x86_64-linux-gnu" ] && [ -f "/usr/include/x86_64-linux-gnu/NvInfer.h" ]; then
        echo -e "TensorRT: ${GREEN}Found (system installation)${NC}"
    elif [ -d "/usr/local/tensorrt" ]; then
        TENSORRT_ROOT="/usr/local/tensorrt"
        echo -e "TensorRT: ${GREEN}$TENSORRT_ROOT${NC}"
    else
        echo -e "${YELLOW}Warning: TensorRT not found automatically.${NC}"
        echo -e "${YELLOW}Use --tensorrt <path> to specify TensorRT location.${NC}"
    fi
else
    echo -e "TensorRT: ${GREEN}$TENSORRT_ROOT${NC}"
fi

echo -e "Build type: ${GREEN}$BUILD_TYPE${NC}"
echo -e "CUDA arch: ${GREEN}$CUDA_ARCH${NC}"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "\n${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo -e "\n${GREEN}Configuring with CMake...${NC}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
)

if [ -n "$TENSORRT_ROOT" ]; then
    CMAKE_ARGS+=(-DTENSORRT_ROOT=$TENSORRT_ROOT)
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo -e "\n${GREEN}Building...${NC}"
cmake --build . -j$(nproc)

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Executables:"
echo "  ./build/bin/efficientnet_trt  - Main inference"
echo "  ./build/bin/benchmark_trt     - Benchmark tool"
echo ""
echo "Quick start:"
echo "  1. Export model:  python3 python/export_model.py --model efficientnet_b0"
echo "  2. Run inference: ./build/bin/efficientnet_trt --model models/efficientnet_b0.onnx --image <image.jpg>"
echo "  3. Benchmark:     ./build/bin/benchmark_trt --model models/efficientnet_b0.onnx --compare"
