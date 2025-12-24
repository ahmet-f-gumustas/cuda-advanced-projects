# EfficientNet TensorRT Optimization

High-performance EfficientNet inference using NVIDIA TensorRT with FP16/INT8 quantization and custom CUDA preprocessing kernels.

## Features

- **TensorRT Engine Builder**: Automatic ONNX to TensorRT conversion with caching
- **Multiple Precision Modes**: FP32, FP16, and INT8 quantization
- **INT8 Calibration**: Entropy-based calibrator for accurate INT8 quantization
- **Custom CUDA Kernels**: Fused resize + normalize + HWC→NCHW preprocessing
- **Batch Processing**: Efficient batch inference support
- **Performance Metrics**: Detailed latency statistics (avg, min, max, P50, P95, P99)

## Requirements

- CUDA 11.0+
- TensorRT 8.0+
- CMake 3.18+
- Python 3.8+ (for model export)

### Python Dependencies

```bash
pip install torch torchvision timm onnx onnxsim
```

## Quick Start

### 1. Build the Project

```bash
# Build with default settings (RTX 4070, Release)
./build.sh

# Or specify CUDA architecture
./build.sh --cuda-arch 86  # RTX 3090

# Clean build
./build.sh --clean
```

### 2. Export Model

```bash
# Export EfficientNet-B0 to ONNX
python3 python/export_model.py --model efficientnet_b0

# Export with dynamic batch
python3 python/export_model.py --model efficientnet_b0 --dynamic
```

### 3. Run Inference

```bash
# Basic inference with FP16
./build/bin/efficientnet_trt \
    --model models/efficientnet_b0.onnx \
    --image data/cat.jpg

# With INT8 quantization
./build/bin/efficientnet_trt \
    --model models/efficientnet_b0.onnx \
    --image data/cat.jpg \
    --precision int8 \
    --calib data/calibration_images/
```

### 4. Benchmark

```bash
# Compare all precision modes
./build/bin/benchmark_trt \
    --model models/efficientnet_b0.onnx \
    --compare \
    --calib data/calibration_images/

# PyTorch baseline comparison
python3 python/benchmark_pytorch.py --model efficientnet_b0 --compare
```

## Usage

### Main Inference Tool

```bash
./build/bin/efficientnet_trt [options]

Required:
  --model, -m <path>       Path to ONNX model file
  --image, -i <path>       Path to input image

Optional:
  --precision, -p <mode>   Precision: fp32, fp16, int8 (default: fp16)
  --engine, -e <path>      Path to save/load TensorRT engine
  --labels, -l <path>      Path to class labels file
  --calib, -c <path>       Calibration data path (required for int8)
  --warmup, -w <n>         Warmup iterations (default: 10)
  --repeat, -r <n>         Benchmark iterations (default: 100)
  --topk, -k <n>           Show top K predictions (default: 5)
```

### Benchmark Tool

```bash
./build/bin/benchmark_trt [options]

Required:
  --model, -m <path>       Path to ONNX model file

Optional:
  --batch, -b <n>          Batch size (default: 1)
  --warmup, -w <n>         Warmup iterations (default: 50)
  --repeat, -r <n>         Benchmark iterations (default: 500)
  --compare, -c            Compare FP32, FP16, INT8
  --calib, -C <path>       Calibration data for INT8
```

## Performance

Expected performance on RTX 4070 (batch size 1):

| Precision | Latency (ms) | Throughput (img/s) | Speedup |
|-----------|-------------|-------------------|---------|
| FP32      | ~2.5        | ~400              | 1.0x    |
| FP16      | ~0.8        | ~1250             | 3.1x    |
| INT8      | ~0.5        | ~2000             | 5.0x    |

*Actual results may vary based on hardware and model.*

## INT8 Calibration

For best INT8 accuracy, provide representative calibration images:

```bash
# Prepare calibration data (100-500 images recommended)
mkdir -p data/calibration_images
cp /path/to/imagenet/val/* data/calibration_images/

# Or create a file list
find /path/to/images -name "*.jpg" > data/calib_list.txt
```

The calibrator supports both directory paths and text file lists.

## Project Structure

```
09.EfficientNet-TensorRT-Optimization/
├── CMakeLists.txt           # Build configuration
├── build.sh                 # Build script
├── README.md
├── include/
│   ├── trt_engine.h         # TensorRT engine wrapper
│   ├── cuda_preprocess.h    # CUDA preprocessing
│   ├── calibrator.h         # INT8 calibrator
│   └── stb_image.h          # Image loading
├── src/
│   ├── trt_engine.cpp       # Engine implementation
│   ├── calibrator.cpp       # Calibrator implementation
│   ├── cuda_preprocess.cu   # CUDA kernels
│   ├── main.cpp             # Inference application
│   └── benchmark.cpp        # Benchmark application
├── python/
│   ├── export_model.py      # ONNX export
│   └── benchmark_pytorch.py # PyTorch baseline
├── models/                  # ONNX models (generated)
├── data/                    # Test images, calibration data
└── build/                   # Build output
```

## CUDA Preprocessing Pipeline

The project includes optimized CUDA kernels for image preprocessing:

1. **Bilinear Resize**: GPU-accelerated image resizing
2. **Normalization**: ImageNet mean/std normalization
3. **Format Conversion**: HWC → NCHW tensor format
4. **Fused Kernel**: Combined resize + normalize + convert in single pass

```cpp
// Example: Fused preprocessing on GPU
efficientnet::preprocessImageFused(
    d_input_image,      // GPU: source image (HWC, uint8)
    d_output_tensor,    // GPU: output tensor (NCHW, float)
    src_width, src_height,
    224, 224,           // Target size
    mean, std,          // Normalization params
    stream
);
```

## Engine Caching

TensorRT engines are cached automatically:
- First run: Builds engine from ONNX (may take 30-60 seconds)
- Subsequent runs: Loads cached `.engine` file (instant)

Engine files are precision-specific: `model.fp16.engine`, `model.int8.engine`

## Troubleshooting

### TensorRT not found
```bash
# Set TensorRT path explicitly
./build.sh --tensorrt /path/to/TensorRT
```

### CUDA architecture mismatch
```bash
# Check your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for specific architecture
./build.sh --cuda-arch 86  # RTX 3090
./build.sh --cuda-arch 89  # RTX 4070
```

### INT8 accuracy issues
- Use more calibration images (500+ recommended)
- Ensure calibration images represent your target data
- Try different calibration methods (entropy vs minmax)

## License

This project is part of cuda-advanced-projects.
