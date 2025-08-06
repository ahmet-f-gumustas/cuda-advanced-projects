#!/bin/bash
# Download and extract ONNX Runtime C++ binaries

set -e

# Default version
ORT_VERSION=${1:-"1.18.1"}
CUDA_VERSION="12"
CUDNN_VERSION="9"

echo "Downloading ONNX Runtime ${ORT_VERSION} for CUDA ${CUDA_VERSION}"

# Determine platform
PLATFORM="linux-x64"
if [[ $(uname -m) == "aarch64" ]]; then
    PLATFORM="linux-aarch64"
fi

# Download URL
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${PLATFORM}-gpu-cuda${CUDA_VERSION}-${ORT_VERSION}.tgz"

# Create directory
mkdir -p third_party/onnxruntime

# Download and extract
echo "Downloading from: ${URL}"
wget -q --show-progress "${URL}" -O onnxruntime.tgz

echo "Extracting..."
tar -xzf onnxruntime.tgz

# Move files to correct location
mv onnxruntime-*/include third_party/onnxruntime/
mv onnxruntime-*/lib third_party/onnxruntime/

# Cleanup
rm -rf onnxruntime-*
rm onnxruntime.tgz

echo "✓ ONNX Runtime ${ORT_VERSION} installed to third_party/onnxruntime/"
echo ""
echo "To use a different installation, set ONNXRUNTIME_ROOT environment variable:"
echo "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime"