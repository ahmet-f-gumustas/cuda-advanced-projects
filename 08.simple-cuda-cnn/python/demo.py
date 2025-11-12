#!/usr/bin/env python3
"""
Basit CUDA CNN Demo
Bu script .so kütüphanesini kullanarak basit bir CNN forward pass yapar.
"""

import ctypes
import numpy as np
import os
import sys
import time

# .so dosyasının yolunu bul
lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libcnn_cuda.so')

if not os.path.exists(lib_path):
    print(f"Hata: {lib_path} bulunamadı!")
    print("Önce 'cmake . && make' ile projeyi build edin.")
    sys.exit(1)

# Kütüphaneyi yükle
lib = ctypes.CDLL(lib_path)

# C fonksiyonlarının imzalarını tanımla
lib.create_cnn_context.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_cnn_context.restype = ctypes.c_void_p

lib.destroy_cnn_context.argtypes = [ctypes.c_void_p]
lib.destroy_cnn_context.restype = None

lib.set_input_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.set_input_data.restype = None

lib.set_filter_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.set_filter_data.restype = None

lib.get_output_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.get_output_data.restype = None

lib.run_cnn_forward.argtypes = [ctypes.c_void_p]
lib.run_cnn_forward.restype = None

lib.get_last_inference_time.argtypes = [ctypes.c_void_p]
lib.get_last_inference_time.restype = ctypes.c_float


def main():
    print("=" * 60)
    print("CUDA CNN Demo - Basit Convolution + ReLU + MaxPooling")
    print("=" * 60)

    # Input boyutları
    input_width = 10
    input_height = 10

    # CNN context oluştur
    print(f"\n1. CNN Context oluşturuluyor ({input_width}x{input_height} input)...")
    ctx = lib.create_cnn_context(input_width, input_height)

    # Input verisi oluştur (rastgele)
    print("\n2. Input verisi hazırlanıyor...")
    input_data = np.random.randn(input_height, input_width).astype(np.float32)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Input örnek değerler:\n{input_data[:3, :3]}")

    # 3x3 Edge detection filter (Sobel-like)
    print("\n3. 3x3 Filter hazırlanıyor (edge detection)...")
    filter_data = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    print(f"   Filter:\n{filter_data}")

    # Verileri GPU'ya kopyala
    print("\n4. Veriler GPU'ya aktarılıyor...")
    input_flat = input_data.flatten()
    filter_flat = filter_data.flatten()

    input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    filter_ptr = filter_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.set_input_data(ctx, input_ptr, len(input_flat))
    lib.set_filter_data(ctx, filter_ptr, len(filter_flat))

    # CNN forward pass çalıştır
    print("\n5. CNN Forward Pass çalıştırılıyor...")
    print("   - Convolution (3x3)")
    print("   - ReLU Activation")
    print("   - Max Pooling (2x2)")

    # Python tarafında da toplam süreyi ölç (warmup için önce bir kere çalıştır)
    lib.run_cnn_forward(ctx)  # Warmup

    # Asıl ölçüm
    start_time = time.time()
    lib.run_cnn_forward(ctx)
    end_time = time.time()

    # GPU timing bilgisini al
    gpu_time_ms = lib.get_last_inference_time(ctx)
    total_time_ms = (end_time - start_time) * 1000

    print(f"\n   ⚡ Inference Performansı:")
    print(f"      GPU Kernel Time: {gpu_time_ms:.4f} ms")
    print(f"      Total Time:      {total_time_ms:.4f} ms")
    print(f"      Throughput:      {1000.0/gpu_time_ms:.2f} FPS")

    # Sonuçları al
    print("\n6. Sonuçlar GPU'dan alınıyor...")
    # Conv output: (10-2) x (10-2) = 8x8
    # After pooling: 8/2 x 8/2 = 4x4
    output_width = (input_width - 2) // 2
    output_height = (input_height - 2) // 2
    output_size = output_width * output_height

    output_data = np.zeros(output_size, dtype=np.float32)
    output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.get_output_data(ctx, output_ptr, output_size)

    # Sonuçları göster
    output_2d = output_data.reshape(output_height, output_width)
    print(f"\n7. Sonuçlar:")
    print(f"   Output shape: {output_2d.shape}")
    print(f"   Output değerleri:\n{output_2d}")

    # Benchmark - birden fazla çalıştır
    print("\n8. Benchmark (10 iterasyon)...")
    times = []
    for i in range(10):
        lib.run_cnn_forward(ctx)
        times.append(lib.get_last_inference_time(ctx))

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"   Ortalama: {avg_time:.4f} ms (±{std_time:.4f})")
    print(f"   Min:      {min_time:.4f} ms")
    print(f"   Max:      {max_time:.4f} ms")
    print(f"   FPS:      {1000.0/avg_time:.2f}")

    # Temizlik
    print("\n9. Kaynaklar temizleniyor...")
    lib.destroy_cnn_context(ctx)

    print("\n" + "=" * 60)
    print("Demo başarıyla tamamlandı! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
