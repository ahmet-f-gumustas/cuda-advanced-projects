#!/usr/bin/env python3
"""
Farklı input boyutları ile performans benchmark'ı
"""

import ctypes
import numpy as np
import os
import sys

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

print("=" * 70)
print("CUDA CNN Performance Benchmark - Farklı Input Boyutları")
print("=" * 70)

# Farklı input boyutları
sizes = [10, 32, 64, 128, 256, 512]
iterations = 100

filter_data = np.random.randn(3, 3).astype(np.float32)
filter_ptr = filter_data.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

print("\n{:<12} {:<15} {:<15} {:<15} {:<12}".format(
    "Input Size", "Output Size", "Avg Time (ms)", "Throughput", "FPS"
))
print("-" * 70)

for size in sizes:
    ctx = lib.create_cnn_context(size, size)

    # Input verisi
    input_data = np.random.randn(size, size).astype(np.float32)
    input_ptr = input_data.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.set_input_data(ctx, input_ptr, size * size)
    lib.set_filter_data(ctx, filter_ptr, 9)

    # Warmup
    for _ in range(10):
        lib.run_cnn_forward(ctx)

    # Benchmark
    times = []
    for _ in range(iterations):
        lib.run_cnn_forward(ctx)
        times.append(lib.get_last_inference_time(ctx))

    avg_time = np.mean(times)
    output_size = ((size - 2) // 2)
    fps = 1000.0 / avg_time

    print("{:<12} {:<15} {:<15.4f} {:<15.2f} {:<12.2f}".format(
        f"{size}x{size}",
        f"{output_size}x{output_size}",
        avg_time,
        1.0 / (avg_time / 1000.0),
        fps
    ))

    lib.destroy_cnn_context(ctx)

print("=" * 70)
print("✓ Benchmark tamamlandı!")
print("=" * 70)
