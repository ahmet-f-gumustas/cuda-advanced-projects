# 🚀 Simple CUDA CNN

CUDA ve C++ ile yazılmış basit bir CNN (Convolutional Neural Network) uygulaması. Proje, CUDA kernellerini `.so` shared library olarak derleyip Python'dan kullanmayı gösterir.

## 📋 Özellikler

- **2D Convolution** - 3x3 filtre ile convolution işlemi
- **ReLU Activation** - Non-linear activation function
- **Max Pooling** - 2x2 pooling operasyonu
- **Python Bindings** - ctypes ile Python entegrasyonu

## 🏗️ Mimari

```
Input (HxW)
    ↓
Convolution 3x3 → (H-2)x(W-2)
    ↓
ReLU Activation
    ↓
Max Pooling 2x2 → (H-2)/2 x (W-2)/2
    ↓
Output
```

## 📂 Proje Yapısı

```
08.simple-cuda-cnn/
├── CMakeLists.txt          # Build konfigürasyonu
├── include/
│   └── cnn_cuda.h         # C/C++ header dosyası
├── src/
│   ├── cnn_kernels.cu     # CUDA kernel implementasyonları
│   └── cnn_wrapper.cpp    # C++ wrapper API
├── python/
│   ├── demo.py            # Ana demo scripti
│   └── test_simple.py     # Basit test scripti
└── build/                 # Build output (libcnn_cuda.so)
```

## 🔧 Gereksinimler

- CUDA Toolkit (>= 11.0)
- CMake (>= 3.18)
- GCC/G++ compiler
- Python 3.x
- NumPy

## ⚙️ Kurulum ve Build

### 1. Projeyi Build Et

```bash
cd 08.simple-cuda-cnn
mkdir -p build
cd build
cmake ..
make
```

Bu komut `build/libcnn_cuda.so` dosyasını oluşturur.

### 2. Build'i Kontrol Et

```bash
ls -lh build/libcnn_cuda.so
```

## 🎮 Kullanım

### Demo Script'ini Çalıştır

```bash
cd 08.simple-cuda-cnn
python3 python/demo.py
```

**Çıktı örneği:**
```
============================================================
CUDA CNN Demo - Basit Convolution + ReLU + MaxPooling
============================================================

1. CNN Context oluşturuluyor (10x10 input)...
2. Input verisi hazırlanıyor...
3. 3x3 Filter hazırlanıyor (edge detection)...
4. Veriler GPU'ya aktarılıyor...
5. CNN Forward Pass çalıştırılıyor...
   - Convolution (3x3)
   - ReLU Activation
   - Max Pooling (2x2)

   ⚡ Inference Performansı:
      GPU Kernel Time: 0.0195 ms
      Total Time:      0.0312 ms
      Throughput:      51398.03 FPS

6. Sonuçlar GPU'dan alınıyor...
7. Sonuçlar:
   Output shape: (4, 4)
   Output değerleri:
   [[...]]
8. Benchmark (10 iterasyon)...
   Ortalama: 0.0194 ms (±0.0016)
   Min:      0.0184 ms
   Max:      0.0239 ms
   FPS:      51516.65
9. Kaynaklar temizleniyor...
============================================================
Demo başarıyla tamamlandı! ✓
============================================================
```

### Basit Test

```bash
python3 python/test_simple.py
```

## 📊 Teknik Detaylar

### CUDA Kernels

1. **conv2d_kernel**: 2D convolution operasyonu
   - 3x3 filter boyutu
   - Thread per pixel paralelizasyon
   - 16x16 thread block boyutu

2. **relu_kernel**: ReLU activation
   - Element-wise operasyon
   - `f(x) = max(0, x)`
   - 256 thread per block

3. **max_pool_kernel**: Max pooling
   - 2x2 pooling window
   - Stride = 2
   - 16x16 thread block boyutu

### Memory Management

- GPU memory allocation: `cudaMalloc`
- Host-Device transfer: `cudaMemcpy`
- Otomatik cleanup: `destroy_cnn_context`

### Performance Timing

- **CUDA Events**: GPU kernel execution time ölçümü
- **Precision**: Milisaniye (ms) cinsinden
- **Warmup**: İlk çalıştırma sonrası stabil timing
- **Benchmark**: Çoklu iterasyonla ortalama/std hesaplama

### Python Entegrasyonu

- **ctypes** ile C library binding
- NumPy array'leri GPU'ya transfer
- Zero-copy pointer passing

## 🔬 Örnek Kullanım

```python
import ctypes
import numpy as np

# Kütüphaneyi yükle
lib = ctypes.CDLL('./build/libcnn_cuda.so')

# Context oluştur
ctx = lib.create_cnn_context(10, 10)

# Input ve filter hazırla
input_data = np.random.randn(10, 10).astype(np.float32)
filter_data = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

# GPU'ya aktar
lib.set_input_data(ctx, input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 100)
lib.set_filter_data(ctx, filter_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 9)

# Forward pass
lib.run_cnn_forward(ctx)

# Timing bilgisini al
inference_time_ms = lib.get_last_inference_time(ctx)
print(f"Inference time: {inference_time_ms:.4f} ms")

# Sonuçları al
output = np.zeros(16, dtype=np.float32)  # 4x4
lib.get_output_data(ctx, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 16)

# Temizle
lib.destroy_cnn_context(ctx)
```

## 📈 Performans

### Benchmark Sonuçları (10x10 input)

```
Input Size:  10x10
Output Size: 4x4 (after conv + pool)

GPU Kernel Time:  ~0.019 ms
Throughput:       ~51,000 FPS
Min/Max Variance: ±0.002 ms
```

### Optimizasyonlar

- **GPU Parallelizasyon**: Tüm operasyonlar CUDA kernels ile paralel
- **Memory Coalescing**: Optimized memory access patterns
- **Synchronization**: Kernel sonrası otomatik sync
- **Timing**: CUDA Events ile hassas ölçüm

## 🧪 Test

```bash
# Basit test
python3 python/test_simple.py

# Detaylı demo
python3 python/demo.py
```

## 🐛 Troubleshooting

**Hata: libcnn_cuda.so bulunamadı**
```bash
# Build klasörünü kontrol et
ls build/libcnn_cuda.so

# Tekrar build et
cd build && cmake .. && make
```

**CUDA Runtime Error**
```bash
# CUDA kurulumunu kontrol et
nvcc --version

# GPU varlığını test et
nvidia-smi
```

## 📝 Notlar

- Bu basit bir demo projesidir
- Gerçek CNN eğitimi için PyTorch/TensorFlow kullanın
- Backpropagation implementasyonu yok
- Tek channel (grayscale) destekler

## 🚀 Genişletme İmkanları

1. **Multi-channel support** - RGB images için
2. **Batch processing** - Çoklu image processing
3. **Backward pass** - Gradient hesaplama
4. **Optimizasyon** - Shared memory kullanımı
5. **Daha fazla layer** - Fully connected, dropout, etc.

## 📚 Kaynaklar

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CNN Fundamentals](https://cs231n.github.io/)

---

**Geliştirici**: CUDA Advanced Projects
**Lisans**: MIT
