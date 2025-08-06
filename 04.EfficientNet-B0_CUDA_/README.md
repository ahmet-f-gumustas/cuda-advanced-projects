# EfficientNet C++ Inference Engine

High-performance C++ implementation of EfficientNet inference with CUDA optimization, supporting ONNX Runtime, TensorRT, and cuDNN backends.

## Features

- **Multiple Backends**: ONNX Runtime (CUDA EP), TensorRT (Jetson), cuDNN (benchmark)
- **Optimized Pipeline**: NVJPEG decoding, CUDA preprocessing, FP16/INT8 support
- **Production Ready**: Pinned memory, CUDA streams, IOBinding, CUDA Graphs
- **Cross-Platform**: x86_64 (dev) and aarch64 (Jetson) support
- **No OpenCV**: Pure CUDA/NVJPEG pipeline with CPU fallback

## Quick Start (x86_64 with ONNX Runtime)

```bash
# 1. Download ONNX Runtime C++ binaries
bash tools/download_ort.sh 1.18.1

# 2. Export EfficientNet model to ONNX
python3 tools/export_to_onnx.py --model efficientnet_b0

# 3. Build the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBACKEND=ORT -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# 4. Run inference
./build/apps/efficientnet_ort --model models/efficientnet_b0.onnx --image data/cat.jpg --fp16
```

## Jetson Deployment (aarch64 with TensorRT)

```bash
# 1. Install JetPack (includes TensorRT/cuDNN)
sudo apt update && sudo apt install -y nvidia-jetpack

# 2. Export model (on dev machine or Jetson)
python3 tools/export_to_onnx.py --model efficientnet_b0

# 3. Build with TensorRT backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBACKEND=TRT -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j

# 4. Run with FP16 (default)
./build/apps/efficientnet_trt --model models/efficientnet_b0.onnx --image data/cat.jpg

# 5. Run with INT8 calibration
./build/apps/efficientnet_trt --model models/efficientnet_b0.onnx --image data/cat.jpg --int8 --calib tools/calib_list.txt
```

## Build Options

### CMake Options
- `-DBACKEND=ORT|TRT|CUDNN`: Select inference backend (default: ORT)
- `-DCMAKE_CUDA_ARCHITECTURES=XX`: CUDA compute capability (89 for RTX 4070, 87 for Jetson Orin)
- `-DENABLE_NVJPEG=ON|OFF`: Enable NVJPEG for GPU JPEG decoding (default: ON)
- `-DCMAKE_BUILD_TYPE=Release|Debug`: Build configuration

### Environment Variables
- `ONNXRUNTIME_ROOT`: Override ONNX Runtime installation path
- `CUDA_VISIBLE_DEVICES`: Select GPU device

## Application Options

### classify_ort
```bash
./efficientnet_ort --help
  --model         Path to ONNX model file (required)
  --image         Path to input image (required)
  --fp16          Use FP16 precision
  --warmup N      Number of warmup iterations (default: 5)
  --repeat N      Number of timing iterations (default: 50)
  --cuda-graphs   Enable CUDA Graphs
  --class-names   Path to class names file
```

### classify_trt
```bash
./efficientnet_trt --help
  --model         Path to ONNX model file (required)
  --image         Path to input image (required)
  --fp16          Use FP16 precision (default: true)
  --int8          Use INT8 precision
  --calib         Path to calibration file list for INT8
  --warmup N      Number of warmup iterations (default: 5)
  --repeat N      Number of timing iterations (default: 50)
  --class-names   Path to class names file
```

## Performance Notes

### First Run
The first inference run may be slower due to:
- cuDNN algorithm search (can be configured via `set_cudnn_conv_algo_search()`)
- TensorRT engine building (cached as `.engine` file)
- CUDA context initialization

### CUDA Graphs
Enable CUDA Graphs for lowest latency with fixed input sizes:
```bash
./efficientnet_ort --model model.onnx --image image.jpg --cuda-graphs
```

**Requirements for CUDA Graphs**:
- Fixed input/output dimensions
- Consistent GPU memory addresses
- No dynamic shapes

### Memory Optimization
The implementation uses:
- Pinned host memory for async transfers
- Persistent device allocations
- Stream-ordered memory operations
- IOBinding for reduced overhead

## Profiling

### NVIDIA Nsight Systems
```bash
nsys profile -t cuda,cudnn,cublas,nvtx -o profile.nsys-rep ./build/apps/efficientnet_ort --model model.onnx --image image.jpg
```

### NVIDIA Nsight Compute
```bash
ncu --set full -o profile ./build/apps/efficientnet_ort --model model.onnx --image image.jpg
```

## Common Issues

### ONNX Runtime library not found
```
Error: libonnxruntime.so: cannot open shared object file
```
**Solution**: Add library path:
```bash
export LD_LIBRARY_PATH=$PWD/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
```

### CUDA version mismatch
```
Error: CUDA runtime version X.Y doesn't match ONNX Runtime CUDA version
```
**Solution**: Download matching ONNX Runtime version or rebuild from source

### TensorRT not found on x86_64
This is expected - TensorRT backend is only enabled on aarch64 (Jetson) by default.

### High first-run latency
Normal behavior due to:
1. cuDNN algorithm search - mitigate with:
   ```cpp
   backend->set_cudnn_conv_algo_search("HEURISTIC");
   ```
2. JIT compilation - use warmup iterations
3. TensorRT engine building - engines are cached after first run

## Architecture Notes

### Pipeline Flow
1. **Image Decode**: NVJPEG (GPU) or STB (CPU fallback)
2. **Preprocessing**: CUDA kernels for resize, normalize, HWC→NCHW
3. **Inference**: Backend-specific (ORT/TRT/cuDNN)
4. **Postprocessing**: Softmax + Top-K on GPU
5. **Results**: Transfer to host and map class names

### Key Design Decisions
- **No OpenCV**: Reduces dependencies, pure GPU pipeline
- **Header-only utilities**: Better inlining for CUDA helpers
- **Backend abstraction**: Easy to add new inference engines
- **RAII everywhere**: Automatic resource management
- **Async by default**: All operations use CUDA streams

## Development

### Adding a new backend
1. Inherit from `InferenceBackend` interface
2. Implement required virtual methods
3. Add CMake target and build logic
4. Create app in `apps/` directory

### Custom preprocessing
Modify `src/preprocess/resize_norm.cu` to add custom augmentations or normalization.

### Different models
The pipeline supports any model with:
- Single image input (3 channels, any size)
- Classification output (logits)
- ONNX format (opset 14+)

## License

This project have a licence!