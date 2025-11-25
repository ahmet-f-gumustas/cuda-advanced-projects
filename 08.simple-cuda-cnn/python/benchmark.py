#!/usr/bin/env python3
"""
Detaylı performans benchmark'ı
- Farklı input boyutları
- İstatistiksel analiz (min, max, std, percentiles)
- Memory kullanımı
- Throughput hesaplamaları
"""

import ctypes
import numpy as np
import os
import sys
import time

lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libcnn_cuda.so')

if not os.path.exists(lib_path):
    print(f"Hata: {lib_path} bulunamadı!")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)

# Function signatures
lib.create_cnn_context.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_cnn_context.restype = ctypes.c_void_p
lib.destroy_cnn_context.argtypes = [ctypes.c_void_p]
lib.set_input_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.set_filter_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.run_cnn_forward.argtypes = [ctypes.c_void_p]
lib.get_last_inference_time.argtypes = [ctypes.c_void_p]
lib.get_last_inference_time.restype = ctypes.c_float

print("=" * 90)
print("CUDA CNN Detailed Performance Benchmark")
print("=" * 90)

# Farklı input boyutları
sizes = [10, 32, 64, 128, 256, 512, 1024]
iterations = 100
warmup_iterations = 10

filter_data = np.random.randn(3, 3).astype(np.float32)
filter_ptr = filter_data.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

print(f"\nBenchmark parametreleri:")
print(f"  - Warmup iterations: {warmup_iterations}")
print(f"  - Test iterations: {iterations}")
print(f"  - Filter: 3x3 (edge detection)")
print()

results = []

for size in sizes:
    print(f"Benchmarking {size}x{size}...", end=" ", flush=True)

    ctx = lib.create_cnn_context(size, size)

    # Input verisi
    input_data = np.random.randn(size, size).astype(np.float32)
    input_ptr = input_data.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.set_input_data(ctx, input_ptr, size * size)
    lib.set_filter_data(ctx, filter_ptr, 9)

    # Warmup
    for _ in range(warmup_iterations):
        lib.run_cnn_forward(ctx)

    # Benchmark
    times = []
    total_start = time.time()
    for _ in range(iterations):
        lib.run_cnn_forward(ctx)
        times.append(lib.get_last_inference_time(ctx))
    total_end = time.time()

    # İstatistikler
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p50 = np.percentile(times, 50)  # Median
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    output_size = ((size - 2) // 2)
    fps = 1000.0 / avg_time

    # Memory hesaplama (MB)
    input_mem = (size * size * 4) / (1024 * 1024)  # float32 = 4 bytes
    output_mem = (output_size * output_size * 4) / (1024 * 1024)
    total_mem = input_mem + output_mem + (9 * 4) / (1024 * 1024)  # + filter

    # Total throughput
    total_time = total_end - total_start
    total_fps = iterations / total_time

    results.append({
        'size': size,
        'output_size': output_size,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'p50': p50,
        'p95': p95,
        'p99': p99,
        'fps': fps,
        'total_fps': total_fps,
        'memory_mb': total_mem
    })

    print(f"Done! (avg: {avg_time:.4f}ms)")

    lib.destroy_cnn_context(ctx)

# Sonuçları göster
print("\n" + "=" * 90)
print("DETAILED RESULTS")
print("=" * 90)

# Özet tablo
print("\n{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
    "Input Size", "Output Size", "Avg (ms)", "Std (ms)", "Min (ms)", "Max (ms)"
))
print("-" * 90)

for r in results:
    print("{:<12} {:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        f"{r['size']}x{r['size']}",
        f"{r['output_size']}x{r['output_size']}",
        r['avg_time'],
        r['std_time'],
        r['min_time'],
        r['max_time']
    ))

# Percentiles
print("\n{:<12} {:<12} {:<12} {:<12} {:<12}".format(
    "Input Size", "P50 (ms)", "P95 (ms)", "P99 (ms)", "FPS"
))
print("-" * 90)

for r in results:
    print("{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.2f}".format(
        f"{r['size']}x{r['size']}",
        r['p50'],
        r['p95'],
        r['p99'],
        r['fps']
    ))

# Memory ve Throughput
print("\n{:<12} {:<15} {:<15} {:<15}".format(
    "Input Size", "Memory (MB)", "GPU FPS", "Total FPS"
))
print("-" * 90)

for r in results:
    print("{:<12} {:<15.3f} {:<15.2f} {:<15.2f}".format(
        f"{r['size']}x{r['size']}",
        r['memory_mb'],
        r['fps'],
        r['total_fps']
    ))

print("\n" + "=" * 90)
print("✓ Benchmark tamamlandı!")
print("=" * 90)

# Performans özeti
print("\nPerformance Summary:")
best_fps = max(r['fps'] for r in results)
best_size = next(r for r in results if r['fps'] == best_fps)
print(f"  - Best GPU performance: {best_fps:.2f} FPS at {best_size['size']}x{best_size['size']}")
print(f"  - Total tests: {len(sizes)} different input sizes")
print(f"  - Total iterations: {len(sizes) * iterations}")
print()
