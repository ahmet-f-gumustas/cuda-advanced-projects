#!/usr/bin/env python3
"""
PyTorch Baseline Benchmark for EfficientNet

Compare PyTorch inference speed with TensorRT to measure optimization gains.
"""

import argparse
import time
import statistics
from typing import List, Tuple

import torch
import torch.cuda


def get_model(model_name: str, device: str = 'cuda'):
    """Load EfficientNet model."""
    try:
        import timm
        model = timm.create_model(model_name, pretrained=True)
    except ImportError:
        import torchvision.models as models
        model_map = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7,
        }
        model = model_map[model_name](weights='IMAGENET1K_V1')

    model = model.to(device)
    model.eval()
    return model


def benchmark_pytorch(
    model: torch.nn.Module,
    batch_size: int,
    input_size: int,
    warmup: int,
    iterations: int,
    device: str,
    use_fp16: bool = False,
    use_compile: bool = False
) -> Tuple[List[float], float]:
    """Run PyTorch benchmark."""

    # Optional: torch.compile (PyTorch 2.0+)
    if use_compile:
        try:
            model = torch.compile(model)
            print("Using torch.compile()")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Create input tensor
    dtype = torch.float16 if use_fp16 else torch.float32
    x = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=dtype)

    if use_fp16:
        model = model.half()

    # Warmup
    print(f"Warming up ({warmup} iterations)...", end=' ', flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    print("done")

    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...", end=' ', flush=True)
    latencies = []

    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms

    print("done")

    throughput = 1000.0 * batch_size / statistics.mean(latencies)
    return latencies, throughput


def print_stats(name: str, latencies: List[float], throughput: float):
    """Print benchmark statistics."""
    sorted_lat = sorted(latencies)

    print(f"\n{name}:")
    print(f"  Average:    {statistics.mean(latencies):.3f} ms")
    print(f"  Min:        {min(latencies):.3f} ms")
    print(f"  Max:        {max(latencies):.3f} ms")
    print(f"  Std Dev:    {statistics.stdev(latencies):.3f} ms")
    print(f"  P50:        {sorted_lat[len(sorted_lat)//2]:.3f} ms")
    print(f"  P95:        {sorted_lat[int(len(sorted_lat)*0.95)]:.3f} ms")
    print(f"  P99:        {sorted_lat[int(len(sorted_lat)*0.99)]:.3f} ms")
    print(f"  Throughput: {throughput:.1f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='PyTorch EfficientNet Benchmark')
    parser.add_argument('--model', '-m', type=str, default='efficientnet_b0')
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--input-size', '-s', type=int, default=224)
    parser.add_argument('--warmup', '-w', type=int, default=50)
    parser.add_argument('--iterations', '-i', type=int, default=500)
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare FP32, FP16, and compiled modes')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*50}")
    print("PyTorch EfficientNet Benchmark")
    print(f"{'='*50}")
    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input size: {args.input_size}")
    print(f"{'='*50}")

    if device == 'cpu':
        print("Warning: Running on CPU, results may not be representative")

    results = []

    if args.compare:
        # FP32
        print("\n--- FP32 ---")
        model = get_model(args.model, device)
        lat, tp = benchmark_pytorch(model, args.batch_size, args.input_size,
                                    args.warmup, args.iterations, device,
                                    use_fp16=False, use_compile=False)
        results.append(("PyTorch FP32", lat, tp))
        del model
        torch.cuda.empty_cache()

        # FP16
        print("\n--- FP16 ---")
        model = get_model(args.model, device)
        lat, tp = benchmark_pytorch(model, args.batch_size, args.input_size,
                                    args.warmup, args.iterations, device,
                                    use_fp16=True, use_compile=False)
        results.append(("PyTorch FP16", lat, tp))
        del model
        torch.cuda.empty_cache()

        # Compiled FP16
        print("\n--- Compiled FP16 ---")
        model = get_model(args.model, device)
        lat, tp = benchmark_pytorch(model, args.batch_size, args.input_size,
                                    args.warmup, args.iterations, device,
                                    use_fp16=True, use_compile=True)
        results.append(("PyTorch Compiled FP16", lat, tp))
        del model
        torch.cuda.empty_cache()

    else:
        model = get_model(args.model, device)
        lat, tp = benchmark_pytorch(model, args.batch_size, args.input_size,
                                    args.warmup, args.iterations, device,
                                    use_fp16=True, use_compile=False)
        results.append(("PyTorch FP16", lat, tp))

    # Print all results
    print(f"\n{'='*50}")
    print("Results Summary")
    print(f"{'='*50}")

    for name, lat, tp in results:
        print_stats(name, lat, tp)

    # Comparison table
    if len(results) > 1:
        print(f"\n{'='*50}")
        print("Comparison")
        print(f"{'='*50}")
        base_lat = statistics.mean(results[0][1])
        for name, lat, tp in results:
            avg_lat = statistics.mean(lat)
            speedup = base_lat / avg_lat
            print(f"  {name:25s}: {avg_lat:8.3f} ms ({speedup:.2f}x vs FP32)")

    print(f"\n{'='*50}")
    print("Note: Compare these results with TensorRT benchmark")
    print("Run: ./build/bin/benchmark_trt --model models/efficientnet_b0.onnx --compare")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
