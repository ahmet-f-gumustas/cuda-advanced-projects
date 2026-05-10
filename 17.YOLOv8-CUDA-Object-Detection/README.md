# YOLOv8-CUDA Object Detection

A from-scratch CUDA / cuDNN / LibTorch implementation of a YOLOv8-style
real-time object detector. Everything from JPEG decode to final NMS boxes
runs on the GPU — including a webcam pipeline that loads any ultralytics
`.pt` model and pushes 30 FPS at 1280×720.

Two executables ship in the same project:

| Binary           | What it does                                                                  |
|------------------|-------------------------------------------------------------------------------|
| `yolo_detect`    | **Custom CUDA model** (our own backbone+neck+head, random weights) on a static image — demonstrates the architecture and kernels end-to-end. |
| `yolo_camera`    | **Trained ultralytics `.pt`** (via LibTorch TorchScript) on a live webcam or video file — uses our custom CUDA kernels for all pre/postprocess and OpenCV for I/O and bbox rendering. |

Plus a benchmark (`yolo_benchmark`) and a test suite (`test_yolo`).

---

## Contents

1. [Architecture](#architecture)
2. [Custom CUDA kernels](#custom-cuda-kernels)
3. [Build](#build)
4. [File layout](#file-layout)
5. [Usage — `yolo_detect` (custom CUDA model)](#usage--yolo_detect-custom-cuda-model)
6. [Usage — `yolo_camera` (webcam + `.pt`)](#usage--yolo_camera-webcam--pt)
7. [Benchmarks](#benchmarks)
8. [Troubleshooting & gotchas](#troubleshooting--gotchas)
9. [Design notes](#design-notes)

---

## Architecture

```
Image (HxW, uint8 HWC)
   │  letterbox (CUDA: aspect-preserving resize + 114-pad)
   │  HWC→CHW float32 normalize
   ▼
Backbone: simplified CSPDarknet
   Stem(3→16, s2) → Down1(16→32, s2) → Bot1 → Down2(32→64, s2) → Bot2 ─┐ P3 (64ch,  /8)
                                                                       ├─►
   Down3(64→128, s2) → Bot3 ───────────────────────────────────────────┘ P4 (128ch, /16)
                                                                       ▼
   Down4(128→256, s2) → Bot4 → SPPF (3×MaxPool-5 + 4-way concat) ──► P5 (256ch, /32)

Neck: PAN-FPN
   Top-down : P5 → up → cat(P4) → conv → P4n
              P4n → up → cat(P3) → conv → P3_out
   Bottom-up: P3_out → down → cat(P4n) → conv → P4_out
              P4_out → down → cat(P5)  → conv → P5_out

Detect head (×3 levels, decoupled):
   cls branch: 2×ConvBNSiLU(3×3) + Conv1×1(→num_classes)
   reg branch: 2×ConvBNSiLU(3×3) + Conv1×1(→4×reg_max)      # DFL

Postprocess (CUDA):
   DFL decode (softmax + expected value) → anchor-free decode (xyxy)
   → score filter → class-aware NMS → unletterbox to original coords
```

Total anchors at 640×640: **8400** = 80² + 40² + 20² (strides 8/16/32).

### Two inference paths

```
       ┌───────────────────────────────────────────────────────────┐
       │                       yolo_detect                         │
       │   Image → CUDA preprocess → Custom CUDA model → CUDA NMS  │
       │   (cuDNN Conv2D, our SPPF/PAN-FPN/Detect head)            │
       └───────────────────────────────────────────────────────────┘
                                       │
                                       │  same kernels, same NMS
                                       ▼
       ┌───────────────────────────────────────────────────────────┐
       │                       yolo_camera                         │
       │   OpenCV webcam → CUDA preprocess → LibTorch (.pt fwd)    │
       │   → CUDA xywh-decode + NMS → OpenCV draw → imshow / mp4   │
       └───────────────────────────────────────────────────────────┘
```

Both paths share the same custom CUDA kernels for preprocessing and
postprocessing; only the model forward differs.

---

## Custom CUDA kernels

All in [src/yolo_kernels.cu](src/yolo_kernels.cu); declared in
[include/yolo_kernels.cuh](include/yolo_kernels.cuh).

| Kernel                              | Purpose                                       |
|-------------------------------------|-----------------------------------------------|
| `letterbox_kernel`                  | Aspect-preserving resize + 114-gray padding    |
| `hwc_uint8_to_chw_float_kernel`     | Layout transform + uint8 → float [0, 1]        |
| `silu_kernel`                       | x · σ(x) activation                            |
| `bn_silu_kernel`                    | Fused BatchNorm + SiLU                         |
| `concat_channel_kernel`             | NCHW channel-axis concat                       |
| `slice_channel_kernel`              | NCHW channel-axis slice                        |
| `upsample_nearest_2x_kernel`        | 2× nearest-neighbor upsample                   |
| `maxpool2d_same_kernel`             | 5×5 maxpool with same-padding (SPPF)           |
| `dfl_decode_kernel`                 | Per-anchor 16-bin softmax → expected value     |
| `decode_predictions_kernel`         | ltrb + stride + sigmoid(class) → xyxy          |
| `yolov8_decode_xywh_kernel`         | Ultralytics-style xywh + sigmoid'd cls decode  |
| `score_filter_kernel`               | Atomic compaction below threshold              |
| `nms_single_block_kernel`           | Class-aware NMS (sort + IoU suppression)       |
| `unletterbox_kernel`                | Inverse letterbox boxes back to original coords|
| `nchw_to_anchor_major_kernel`       | Flatten per-level outputs into anchor-major    |

`Conv2D` ([src/conv2d.cu](src/conv2d.cu)) wraps cuDNN with
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` and Tensor Op math. Bias
is added via `cudnnAddTensor`; SiLU runs as a custom kernel afterward
(cuDNN has no native SiLU).

---

## Build

### Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA Toolkit | 12.x  | tested on 12.6 |
| cuDNN | 9 | `apt install libcudnn9-dev` |
| CMake | ≥ 3.18 | |
| GCC | ≥ 11 (C++17) | |
| OpenCV | 4.x with `videoio` | only needed for `yolo_camera` |
| LibTorch | from any `python3-torch` install (≥ 2.x) | only needed for `yolo_camera`; CMake auto-discovers it |
| Python `ultralytics` | 8.x | only needed once to export `.pt` → `.torchscript` |

```bash
mkdir -p build && cd build
cmake ..
make -j
```

`BUILD_CAMERA` is `ON` by default; if LibTorch or OpenCV isn't found the
camera target is automatically disabled with a warning and the rest builds
fine. To force-disable:

```bash
cmake -DBUILD_CAMERA=OFF ..
```

CMake auto-discovers LibTorch from the Python `torch` package via
`python3 -c "import torch; ..."`. To override, pass `-DTorch_DIR=...`.

Resulting binaries (in `build/`):

```
build/yolo_detect       # custom model demo
build/yolo_camera       # .pt + webcam (built if LibTorch+OpenCV found)
build/yolo_benchmark    # synthetic-input throughput test
build/test_yolo         # unit + integration tests
```

---

## File layout

```
17.YOLOv8-CUDA-Object-Detection/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── cuda_utils.h          # CUDA_CHECK, CUDNN_CHECK, CudaTimer
│   ├── yolo_kernels.cuh      # All custom-kernel declarations
│   ├── conv2d.h              # cuDNN Conv2D wrapper (+ SiLU)
│   ├── backbone.h            # CSPDarknet stem → 4 stages → SPPF
│   ├── neck.h                # PAN-FPN
│   ├── head.h                # Decoupled cls+reg head with DFL
│   ├── yolov8.h              # Top-level model (backbone+neck+head + ws)
│   ├── postprocess.h         # DFL/NMS host wrapper, run() & run_decoded()
│   ├── pipeline.h            # Image-in → detections-out (custom model)
│   ├── ts_pipeline.h         # Image-in → detections-out (LibTorch .pt)
│   └── image_io.h            # PPM reader/writer, bbox drawer (host)
├── src/
│   ├── yolo_kernels.cu       # All kernel implementations
│   ├── conv2d.cu             # cuDNN Conv2D
│   ├── backbone.cu
│   ├── neck.cu
│   ├── head.cu
│   ├── yolov8.cu
│   ├── postprocess.cu
│   ├── pipeline.cu
│   ├── ts_pipeline.cpp       # LibTorch glue
│   ├── image_io.cpp
│   ├── main.cu               # yolo_detect demo
│   ├── benchmark.cu          # yolo_benchmark
│   └── yolo_camera.cpp       # yolo_camera: OpenCV + LibTorch + V4L2 tweaks
├── tests/
│   └── test_yolo.cu          # 14 unit + integration tests
├── models/
│   ├── yolov8n.pt            # downloaded from ultralytics (6.2 MB)
│   └── yolov8n.torchscript   # CUDA-traced export (12.4 MB)
└── build/                    # CMake out-of-source build
```

---

## Usage — `yolo_detect` (custom CUDA model)

Runs our own backbone+neck+head with random Kaiming-init weights. Purpose:
exercise every CUDA kernel end-to-end on a real image and produce an
annotated output.

```bash
# Synthetic 800×600 image (no input file needed)
./build/yolo_detect

# With a PPM image (P6 binary, 8-bit RGB)
./build/yolo_detect --input photo.ppm --output annotated.ppm \
                   --score 0.25 --iou 0.45 --seed 42

# Convert a JPEG/PNG to PPM with ImageMagick if needed:
convert photo.jpg photo.ppm
```

Flags:

| Flag       | Default          | Meaning                              |
|------------|------------------|--------------------------------------|
| `--input`  | (none)           | PPM input; falls back to synthetic    |
| `--output` | `detections.ppm` | Annotated PPM output                  |
| `--score`  | `0.25`           | Per-class score threshold             |
| `--iou`    | `0.45`           | NMS IoU threshold                     |
| `--seed`   | `42`             | Random-weight seed (reproducibility)  |

### Tests

```bash
./build/test_yolo
```

14 tests covering SiLU, concat-channel, upsample-2x, maxpool-same,
letterbox parameters, DFL decode at known peaks, Conv2D shape correctness,
and an end-to-end pipeline smoke test.

### Synthetic benchmark

```bash
./build/yolo_benchmark --iters 100 --warmup 10 --width 1280 --height 720
```

---

## Usage — `yolo_camera` (webcam + `.pt`)

Runs an ultralytics-trained `.pt` model on a live webcam (or a video file)
via LibTorch. Custom CUDA kernels handle the pre/postprocess; the model
forward runs through `torch::jit::Module`.

### Step 1 — Export your `.pt` to TorchScript

```bash
cd models
python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')          # auto-downloads if missing; or use your own .pt path
m.export(format='torchscript', imgsz=640, device=0)
"
# produces yolov8n.torchscript
```

> ⚠️ **`device=0` is mandatory.** Without it, ultralytics traces on CPU and
> bakes anchor constants as CPU tensors. LibTorch will then raise
> `Expected all tensors to be on the same device, but found at least two
> devices, cuda:0 and cpu!` at the first forward.

Works with any ultralytics YOLOv8 variant — `yolov8n/s/m/l/x.pt`,
fine-tuned models, custom-class models. The pipeline auto-detects the
class count from the model's output shape (4 + num_classes).

### Step 2 — Run

```bash
# Live webcam, 1280×720, COCO classes
./build/yolo_camera --model models/yolov8n.torchscript --cam 0

# Lower resolution = higher pipeline headroom
./build/yolo_camera --model models/yolov8n.torchscript --cam 0 --width 640 --height 480

# Run on a video file
./build/yolo_camera --model models/yolov8n.torchscript --video clip.mp4 --save out.mp4

# Headless (no GUI, useful over SSH) — 60-frame smoke run
./build/yolo_camera --model models/yolov8n.torchscript --headless --save out.mp4

# Stricter thresholds
./build/yolo_camera --model models/yolov8n.torchscript --score 0.4 --iou 0.5
```

| Flag          | Default                       | Meaning                                  |
|---------------|-------------------------------|------------------------------------------|
| `--model`     | `models/yolov8n.torchscript`  | Path to `.torchscript` file              |
| `--cam`       | `0`                           | `/dev/video<N>` index                    |
| `--video`     | (none)                        | Use a video file instead of a camera     |
| `--width`     | `1280`                        | Requested capture width                  |
| `--height`    | `720`                         | Requested capture height                 |
| `--score`     | `0.25`                        | Score threshold                          |
| `--iou`       | `0.45`                        | NMS IoU threshold                        |
| `--headless`  | (off)                         | No window; runs 60 frames then quits     |
| `--save PATH` | (off)                         | Write annotated MP4 to `PATH`            |

### Window controls

- `q` or `ESC` → quit
- HUD top-left: `FPS x | read y | infer z | draw w | write v ms | N dets`
  - `read` — `cap.read()` time (camera → host buffer, including BGR conversion)
  - `infer` — full GPU pipeline (upload + pre + forward + post)
  - `draw` — OpenCV bbox rendering
  - `write` — MP4 encode (0 if `--save` is off)

### Pipeline breakdown (RTX 4070 Laptop, yolov8n, 1280×720)

| Stage                    | Time   |
|--------------------------|--------|
| Upload (BGR→GPU)         | ~0.4 ms |
| Preprocess (CUDA)        | ~0.1 ms |
| Forward (LibTorch)       | ~5 ms   |
| Postprocess (CUDA)       | ~0.5 ms |
| **GPU end-to-end**       | **~6 ms (>150 FPS)** |
| `cap.read()` (V4L2 MJPG) | ~28 ms (= 30 FPS limit) |
| **Wall-clock FPS**       | **~30 (camera-bound)** |

The detection pipeline is ~5× faster than the webcam can deliver frames.
Observed FPS is capped by the USB camera's native rate (30 FPS at 720p).

---

## Benchmarks

### `yolo_detect` custom model — 1280×720 input, 640×640 net

| Stage         | Mean    | p50     | p99     |
|---------------|---------|---------|---------|
| Preprocess    |  0.03   |  0.03   |  0.16   |
| Forward       |  2.50   |  2.58   |  2.92   |
| Postprocess   |  1.29   |  1.29   |  1.63   |
| **Total**     | **4.19**| **4.23**| **5.15**|

Throughput: **~239 FPS** (random weights, 80 classes, 100 iters).

### `yolo_camera` LibTorch model — yolov8n, 1280×720 webcam

| Stage         | Mean    |
|---------------|---------|
| Upload        |  0.4    |
| Preprocess    |  0.1    |
| Forward       |  5.0    |
| Postprocess   |  0.5    |
| **GPU total** | **~6**  |

GPU pipeline: **>150 FPS** | Wall-clock: **~30 FPS** (camera-bound).

---

## Troubleshooting & gotchas

### Webcam appears to run at ~7 FPS even though hardware advertises 30

Two unrelated V4L2 traps. `yolo_camera` already works around both, but
they're worth knowing for any future webcam code.

#### Trap 1: `exposure_dynamic_framerate=1` (UVC autoexposure)

Linux UVC cameras default this flag to `1`, which gives the autoexposure
loop permission to **drop the device FPS** (e.g. 30 → 6) when light is
low — so it can give the sensor more time per frame. `cap.read()` then
blocks ~120-180 ms per frame regardless of how fast inference is.

`yolo_camera` disables this at startup via direct ioctl
(`v4l2_pin_framerate()` in [src/yolo_camera.cpp](src/yolo_camera.cpp)).
Manual equivalent:

```bash
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
```

Check what the camera is currently doing:
```bash
v4l2-ctl -d /dev/video0 --get-ctrl=exposure_dynamic_framerate
v4l2-ctl -d /dev/video0 -L | grep -A1 exposure
```

#### Trap 2: V4L2 silently keeps YUYV pixel format

Most laptop webcams support both YUYV and MJPG at 720p, but YUYV is capped
at **10 FPS** while MJPG runs at the full 30. OpenCV opens YUYV by default.

You must call `cap.set(CAP_PROP_FOURCC, ...)` **before** `CAP_PROP_FRAME_WIDTH/HEIGHT`,
otherwise V4L2 ignores the format change. `yolo_camera` does this in the
right order. Check the open format:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --get-fmt-video       # current Pixel Format
```

### `.pt` import → "found at least two devices, cuda:0 and cpu"

The TorchScript was traced on CPU. Re-export with `device=0`:

```bash
python3 -c "from ultralytics import YOLO; YOLO('your.pt').export(format='torchscript', imgsz=640, device=0)"
```

### `cv::imshow` errors with "cannot open display"

You're running over SSH or without an X/Wayland session. Either set
`DISPLAY` (and use `ssh -X`), or run headless:

```bash
./build/yolo_camera --model models/yolov8n.torchscript --headless --save out.mp4
vlc out.mp4
```

### `Failed to read .../yolov8n.torchscript`

Run from the project root (not from `build/`), or pass an absolute path:

```bash
cd /home/red/git-projects/cuda-advanced-projects/17.YOLOv8-CUDA-Object-Detection
./build/yolo_camera --model "$PWD/models/yolov8n.torchscript"
```

### LibTorch not found at configure time

CMake tries to auto-discover it from your Python `torch` install. If you
have multiple Python environments, force the one with torch:

```bash
cmake -DPython_EXECUTABLE=$(which python3) ..
# or pass the path directly:
cmake -DTorch_DIR=/home/red/.local/lib/python3.10/site-packages/torch/share/cmake/Torch ..
```

### cuDNN not found

```bash
sudo apt install libcudnn9-dev   # Debian / Ubuntu
# or point CMake at a tarball install:
cmake -DCUDNN_LIB=/path/to/libcudnn.so -DCUDNN_INCLUDE=/path/to/cudnn.h/dir ..
```

---

## Design notes

### Why a simplified architecture in `yolo_detect`?

The custom model uses **single-bottleneck stages** instead of ultralytics'
variable-depth C2f modules. All real ingredients are present (CSPDarknet
structure, SPPF, PAN-FPN, decoupled head, DFL), just with fixed depth = 1.
This keeps the code readable and lets the architecture fit in one
half-day's worth of files.

For inference with trained weights, use the `yolo_camera` path — LibTorch
loads the real architecture from your `.pt`, while our CUDA kernels still
handle every pre/postprocess step.

### Why both a custom model and a LibTorch path?

The custom model proves we can implement every kernel needed for YOLOv8
inference from scratch in CUDA. The LibTorch path makes those kernels
actually useful with real trained weights without having to write a
weight-file parser or match ultralytics' exact C2f topology layer by
layer.

### Why is NMS single-block-serial?

After score-threshold filtering, K is typically ≤ a few hundred. A
single-block sequential NMS (`nms_single_block_kernel`) is the simplest
correct implementation at that scale and avoids the complexity of a
bitonic-sort + IoU-matrix approach. For mass-detection workloads (e.g.
crowd counting with K in the thousands), swap in a parallel NMS — the
rest of the pipeline doesn't change.

### Why custom CUDA kernels for pre/postprocess when LibTorch could do it?

Two reasons:
1. Pre/postprocess is where most "real" inference frameworks lose
   throughput — keeping it in our own kernels means no host round-trips
   and no torch tensor allocation overhead per frame.
2. The custom kernels are the educational point of this project.
   `yolo_camera` lets you swap in any trained `.pt` while still going
   through our letterbox/decode/NMS pipeline.

### Weight initialization in the custom model

`Conv2D` uses Kaiming-normal init (`N(0, sqrt(2/fan_in))`) with biases
zeroed. The custom-model output is therefore random — the project is
about kernel and pipeline correctness, not about producing accurate boxes
without training. To verify correctness end-to-end with meaningful
detections, use `yolo_camera`.

---

## Repository context

Project 17 in a series of CUDA learning projects living at
`/home/red/git-projects/cuda-advanced-projects/`. Sibling projects span
denoising, ViT, EfficientNet, GPT inference, Stable Diffusion, Whisper,
3D Gaussian Splatting, and more. This project follows the same
conventions: `CUDA_CHECK` / `CudaTimer`, no namespaces, `.cuh` for GPU
headers, `find_package(CUDAToolkit)` instead of legacy CUDA CMake.
