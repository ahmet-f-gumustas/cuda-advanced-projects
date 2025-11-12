#!/usr/bin/env python3
"""
Basit test scripti - 4x4 input ile küçük test
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

lib.create_cnn_context.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_cnn_context.restype = ctypes.c_void_p
lib.destroy_cnn_context.argtypes = [ctypes.c_void_p]
lib.set_input_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.set_filter_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.get_output_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.run_cnn_forward.argtypes = [ctypes.c_void_p]
lib.get_last_inference_time.argtypes = [ctypes.c_void_p]
lib.get_last_inference_time.restype = ctypes.c_float

# Küçük test
input_size = 6
ctx = lib.create_cnn_context(input_size, input_size)

# Basit input
input_data = np.arange(36).reshape(6, 6).astype(np.float32)
print("Input:\n", input_data)

# Identity filter
filter_data = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)
print("\nFilter:\n", filter_data)

input_ptr = input_data.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
filter_ptr = filter_data.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

lib.set_input_data(ctx, input_ptr, 36)
lib.set_filter_data(ctx, filter_ptr, 9)

# Timing ile çalıştır
print("\nInference çalıştırılıyor...")
start = time.time()
lib.run_cnn_forward(ctx)
end = time.time()

gpu_time = lib.get_last_inference_time(ctx)
print(f"GPU Time: {gpu_time:.4f} ms")
print(f"Total Time: {(end-start)*1000:.4f} ms")

output_data = np.zeros(4, dtype=np.float32)  # (6-2)/2 x (6-2)/2 = 2x2
output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
lib.get_output_data(ctx, output_ptr, 4)

print("\nOutput (2x2):\n", output_data.reshape(2, 2))

lib.destroy_cnn_context(ctx)
print("\n✓ Test başarılı!")
