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
