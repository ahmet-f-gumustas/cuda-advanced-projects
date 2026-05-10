# YOLOv8-CUDA Object Detection

A from-scratch CUDA/cuDNN implementation of a YOLOv8-style real-time object
detector. Anchor-free decoupled head with Distribution Focal Loss regression
and class-aware NMS — everything from preprocessing to NMS runs on the GPU.

## Architecture

```
Image (HxW, uint8 HWC)
   │  letterbox (CUDA: aspect-preserving resize + 114-pad)
   │  HWC→CHW float32 normalize
   ▼
Backbone: simplified CSPDarknet
   Stem(3→16, s2) → Down1(16→32, s2) → Bot1 → Down2(32→64, s2) → Bot2 ─┐ P3 (64ch, /8)
                                                                       ├─►
   Down3(64→128, s2) → Bot3 ───────────────────────────────────────────┘ P4 (128ch, /16)
                                                                       ▼
   Down4(128→256, s2) → Bot4 → SPPF (3×MaxPool-5 + 4-way concat) ──► P5 (256ch, /32)

Neck: PAN-FPN
   Top-down: P5→up→cat(P4)→conv → P4n
             P4n→up→cat(P3)→conv → P3_out
   Bottom-up: P3_out→down→cat(P4n)→conv → P4_out
              P4_out→down→cat(P5)→conv → P5_out

Detect head (×3 levels, decoupled):
   cls branch: 2×ConvBNSiLU(3×3) + Conv1×1(→num_classes)
   reg branch: 2×ConvBNSiLU(3×3) + Conv1×1(→4×reg_max)   # DFL

Postprocess (CUDA):
   DFL decode (softmax+expected value) → anchor-free decode (xyxy)
   → score filter → class-aware NMS → unletterbox to original coords
```

Total anchors at 640×640: **8400** (80² + 40² + 20²).

## Custom CUDA Kernels

`src/yolo_kernels.cu` implements the full preprocess + postprocess path:

| Kernel                              | Purpose                                    |
|-------------------------------------|--------------------------------------------|
| `letterbox_kernel`                  | aspect-preserving resize + gray padding    |
| `hwc_uint8_to_chw_float_kernel`     | layout transform + uint8→float[0,1]        |
| `silu_kernel`                       | x · σ(x) activation                        |
| `bn_silu_kernel`                    | fused BatchNorm + SiLU                     |
| `concat_channel_kernel`             | NCHW channel-axis concat                   |
| `upsample_nearest_2x_kernel`        | 2× nearest-neighbor upsample               |
| `maxpool2d_same_kernel`             | 5×5 maxpool with same-padding for SPPF     |
| `dfl_decode_kernel`                 | per-anchor 16-bin softmax→expected value   |
| `decode_predictions_kernel`         | ltrb + stride + sigmoid(class)→xyxy        |
| `score_filter_kernel`               | atomic compaction below threshold          |
| `nms_single_block_kernel`           | class-aware NMS (sort + IoU suppression)   |
| `unletterbox_kernel`                | inverse-letterbox boxes back to original   |
| `nchw_to_anchor_major_kernel`       | flatten per-level outputs into anchor-major|

`Conv2D` uses cuDNN with `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`
and Tensor Op math; bias is added separately and SiLU runs as a custom kernel
afterwards (cuDNN has no native SiLU).

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j
```

Requires CUDA 12.x and cuDNN 9 (Debian/Ubuntu: `apt install libcudnn9-dev`).

## Run

```bash
# Synthetic demo (no input image needed)
./yolo_detect

# With a PPM image
./yolo_detect --input photo.ppm --output annotated.ppm --score 0.25 --iou 0.45

# Benchmark
./yolo_benchmark --iters 100 --warmup 10 --width 1280 --height 720

# Unit + integration tests
./test_yolo
```

## Live Webcam + Trained `.pt` (yolo_camera)

A second executable, `yolo_camera`, runs a TorchScript-exported ultralytics
YOLOv8 `.pt` model on a live webcam feed. It uses **LibTorch** for the model
forward and the **custom CUDA kernels** for all pre/postprocessing (letterbox,
HWC→CHW, xywh-decode, score filter, NMS, unletterbox).

### Export a `.pt` to TorchScript (one-time, run from `models/`)

```bash
cd models
python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')          # or your own .pt
m.export(format='torchscript', imgsz=640, device=0)
"
# produces yolov8n.torchscript
```

Important: pass `device=0` so the trace embeds constants on CUDA — otherwise
LibTorch raises a CPU/GPU mismatch at forward time.

### Run on webcam

```bash
./yolo_camera --model models/yolov8n.torchscript --cam 0
              --width 1280 --height 720 --score 0.25 --iou 0.45

# Run a video file instead of a camera
./yolo_camera --model models/yolov8n.torchscript --video clip.mp4 --save out.mp4

# Headless smoke run (no window — useful over SSH)
./yolo_camera --model models/yolov8n.torchscript --headless --save out.mp4
```

Press `q` or `ESC` to quit. `--save` writes an annotated mp4 alongside live display.

### Pipeline timings (RTX 4070 Laptop, yolov8n, 1280×720 webcam input)

| Stage                | ms   |
|----------------------|------|
| Upload (BGR→GPU)     | ~0.4 |
| Preprocess (CUDA)    | ~0.1 |
| Forward (LibTorch)   | ~5   |
| Postprocess (CUDA)   | ~0.5 |
| **End-to-end**       | **~6** |

Detection pipeline alone is well over 150 FPS; observed FPS is capped by the
USB webcam's native frame rate (30 FPS) and OpenCV frame-grab latency.

### Webcam FPS gotcha — V4L2 `exposure_dynamic_framerate`

On Linux UVC webcams, if you see ~7 FPS even though `v4l2-ctl --list-formats-ext`
advertises 30 FPS at your resolution, the culprit is almost always
**`exposure_dynamic_framerate=1`**. This flag lets the autoexposure loop *drop
the device FPS* (e.g. 30 → 6) to give the sensor more light per frame in
dim conditions. `cap.read()` blocks until the next frame arrives, so the
pipeline sits idle ~120 ms each frame regardless of how fast inference is.

`yolo_camera` calls `v4l2_pin_framerate()` at startup to set this control to
0 via direct ioctl, which forces the camera back to its rated FPS. Manual
equivalent:

```bash
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
```

Also worth knowing: V4L2 silently keeps **YUYV** unless you set
`CAP_PROP_FOURCC` **before** `CAP_PROP_FRAME_WIDTH/HEIGHT`. YUYV caps at 10
FPS at 1280×720 on most laptop webcams; **MJPG** runs the full 30 FPS. The
code already does this in the right order.

## Benchmarks (RTX 4070 Laptop, sm_89, 640×640)

| Stage         | Mean   | p50    | p99    |
|---------------|--------|--------|--------|
| Preprocess    |  0.03  |  0.03  |  0.16  |
| Forward       |  2.50  |  2.58  |  2.92  |
| Postprocess   |  1.29  |  1.29  |  1.63  |
| **Total**     | **4.19**| **4.23** | **5.15** |

Throughput: **~239 FPS** on 1280×720 input (random weights, 80 classes).

## Notes

- Weights are randomly initialized (Kaiming); this project focuses on the
  CUDA inference pipeline and architecture, not training. The full pipeline
  is byte-compatible with a YOLOv8 export — swapping in trained weights is a
  straightforward extension.
- The model is a simplified-but-faithful YOLOv8 (single-bottleneck stages
  instead of C2f's variable depth) for clarity. All real ingredients are
  present: CSPDarknet structure, SPPF, PAN-FPN, decoupled head, DFL.
- NMS runs single-block-serial for simplicity (correct for post-filter
  K ≲ a few hundred). For production extreme detection counts, a
  parallel-friendly NMS (bitonic sort + IoU matrix) is the natural upgrade.
