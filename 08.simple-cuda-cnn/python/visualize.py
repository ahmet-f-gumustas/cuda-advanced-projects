#!/usr/bin/env python3
"""
CNN Visualization Script
Convolution, activation ve pooling sonuçlarını görselleştirir
"""

import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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
lib.set_input_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.set_filter_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.run_cnn_forward.argtypes = [ctypes.c_void_p]
lib.get_output_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.get_last_inference_time.argtypes = [ctypes.c_void_p]
lib.get_last_inference_time.restype = ctypes.c_float


def visualize_cnn():
    """CNN işlemlerini görselleştir"""

    # Input boyutları
    input_width = 16
    input_height = 16

    # Context oluştur
    ctx = lib.create_cnn_context(input_width, input_height)

    # Input: Basit bir pattern (çapraz çizgiler)
    input_data = np.zeros((input_height, input_width), dtype=np.float32)
    input_data[7:9, :] = 1.0  # Yatay çizgi
    input_data[:, 7:9] = 1.0  # Dikey çizgi

    # Edge detection filtreleri
    filters = {
        'Vertical Edge': np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]], dtype=np.float32),
        'Horizontal Edge': np.array([[-1, -1, -1],
                                      [ 0,  0,  0],
                                      [ 1,  1,  1]], dtype=np.float32),
        'Diagonal Edge': np.array([[ 0,  1,  1],
                                   [-1,  0,  1],
                                   [-1, -1,  0]], dtype=np.float32),
    }

    # Her filtre için sonuçları göster
    fig, axes = plt.subplots(len(filters), 4, figsize=(16, 4 * len(filters)))
    fig.suptitle('CUDA CNN Visualization', fontsize=16)

    for idx, (filter_name, filter_data) in enumerate(filters.items()):
        # Input verisini GPU'ya kopyala
        input_flat = input_data.flatten()
        filter_flat = filter_data.flatten()

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        filter_ptr = filter_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.set_input_data(ctx, input_ptr, len(input_flat))
        lib.set_filter_data(ctx, filter_ptr, len(filter_flat))

        # Forward pass
        lib.run_cnn_forward(ctx)
        inference_time = lib.get_last_inference_time(ctx)

        # Output al
        output_width = (input_width - 2) // 2
        output_height = (input_height - 2) // 2
        output_size = output_width * output_height

        output_data = np.zeros(output_size, dtype=np.float32)
        output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.get_output_data(ctx, output_ptr, output_size)
        output_2d = output_data.reshape(output_height, output_width)

        # Visualize
        row = idx if len(filters) > 1 else [axes]

        # Input
        axes[row][0].imshow(input_data, cmap='gray')
        axes[row][0].set_title(f'Input ({input_width}x{input_height})')
        axes[row][0].axis('off')

        # Filter
        axes[row][1].imshow(filter_data, cmap='RdBu', vmin=-1, vmax=1)
        axes[row][1].set_title(f'Filter: {filter_name}')
        axes[row][1].axis('off')

        # Convolution result (before pooling)
        conv_size = input_width - 2
        axes[row][2].set_title(f'After Convolution ({conv_size}x{conv_size})')
        axes[row][2].text(0.5, 0.5, 'See Output →',
                         ha='center', va='center', transform=axes[row][2].transAxes)
        axes[row][2].axis('off')

        # Final output
        im = axes[row][3].imshow(output_2d, cmap='viridis')
        axes[row][3].set_title(f'Output ({output_width}x{output_height})\nTime: {inference_time:.4f}ms')
        axes[row][3].axis('off')
        plt.colorbar(im, ax=axes[row][3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Kaydet
    output_path = os.path.join(os.path.dirname(__file__), '..', 'cnn_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization kaydedildi: {output_path}")

    plt.show()

    # Temizlik
    lib.destroy_cnn_context(ctx)


if __name__ == "__main__":
    print("=" * 60)
    print("CUDA CNN Visualization")
    print("=" * 60)
    visualize_cnn()
    print("=" * 60)
    print("Tamamlandı!")
    print("=" * 60)
