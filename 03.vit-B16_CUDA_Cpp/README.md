# ViT-B/16 CUDA C++ Implementation

High-performance Vision Transformer (ViT-B/16) inference implementation using CUDA 12.x, cuBLASLt, and custom kernels.

## Features
- FP16 inference with Tensor Cores on RTX 4070/A100
- Custom CUDA kernels for LayerNorm, Softmax, GELU
- Fused bias+residual operations
- cuBLASLt for optimized GEMM operations
- Zero Python dependencies at runtime

## Build Instructions

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

# Generate dummy weights for testing
./tools/gen_dummy_weights

# Run inference
./vit_infer --image demo.jpg --batch 8 --precision fp16 --device 0 --weights weights.bin
```

## Requirements
- CUDA 12.x
- cuBLAS/cuBLASLt
- OpenCV (for image I/O only)
- C++20 compatible compiler
- NVIDIA GPU with Tensor Core support (RTX 4070, A100, etc.)

## Model Specifications
- Model: ViT-B/16
- Image size: 224×224
- Patch size: 16×16
- Hidden dimension: 768
- Attention heads: 12
- Encoder layers: 12
- Number of patches: 196 + 1 (class token)

## Weight Format
Binary format with header:
```
[4 bytes] magic number (0x56495442)
[4 bytes] version (1)
[4 bytes] precision (0=FP32, 1=FP16)
[4 bytes] hidden_dim (768)
[4 bytes] num_heads (12)
[4 bytes] num_layers (12)
[weights data...]
```

## Benchmarking
```bash
# Benchmark different batch sizes
./vit_infer --image test.jpg --batch 1 --precision fp16 --benchmark
./vit_infer --image test.jpg --batch 8 --precision fp16 --benchmark
./vit_infer --image test.jpg --batch 32 --precision fp16 --benchmark
```

## Performance on RTX 4070
- Batch 1: ~X ms/image
- Batch 8: ~Y ms/image
- Batch 32: ~Z ms/image