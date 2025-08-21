# cuFFT Tutorial - CUDA Fast Fourier Transform

## Giriş
cuFFT (CUDA Fast Fourier Transform), NVIDIA GPU'lar üzerinde yüksek performanslı FFT hesaplamaları yapmak için tasarlanmış bir kütüphanedir. FFT, sinyal işleme, görüntü işleme ve bilimsel hesaplamalarda yaygın olarak kullanılan temel algoritmalardan biridir.

## cuFFT Nedir?
- **FFT (Fast Fourier Transform)**: Sinyalleri zaman domeninden frekans domenine dönüştüren bir algoritma
- **cuFFT**: NVIDIA'nın GPU-hızlandırmalı FFT kütüphanesi
- CPU'daki FFTW kütüphanesine benzer API sağlar
- 1D, 2D ve 3D FFT'leri destekler

## Kurulum Kontrolü
CUDA 12.4 kurulumunuzda cuFFT zaten yüklü durumda. Kontrol etmek için:

```bash
# cuFFT header dosyasını kontrol et
ls /usr/local/cuda/include/cufft*

# cuFFT kütüphanelerini kontrol et
ls /usr/local/cuda/lib64/libcufft*

# pkg-config ile kontrol (varsa)
pkg-config --libs cufft
```

## Derleme Komutları
cuFFT programlarını derlerken aşağıdaki flag'leri kullanın:

```bash
# Temel derleme
nvcc -o program program.cu -lcufft

# Optimizasyonlu derleme
nvcc -O3 -arch=sm_89 -o program program.cu -lcufft -lcufftw

# Debug modu
nvcc -g -G -o program program.cu -lcufft
```

## Proje İçeriği

### 1. Temel Örnekler
- `01_simple_1d_fft.cu` - Basit 1D FFT örneği
- `02_simple_2d_fft.cu` - 2D FFT (görüntü işleme için)
- `03_batch_fft.cu` - Birden fazla FFT'yi aynı anda işleme
- `04_real_to_complex.cu` - Gerçek sayılardan kompleks sayılara FFT

### 2. İleri Seviye Örnekler
- `05_convolution_fft.cu` - FFT kullanarak konvolüsyon
- `06_fft_benchmark.cu` - CPU vs GPU performans karşılaştırması
- `07_multi_gpu_fft.cu` - Çoklu GPU kullanımı

### 3. Yardımcı Dosyalar
- `utils.h` - Yardımcı fonksiyonlar
- `CMakeLists.txt` - CMake build dosyası
- `Makefile` - Alternatif build sistemi

## FFT Türleri

### 1D FFT Türleri
- **C2C (Complex-to-Complex)**: Kompleks girdi → Kompleks çıktı
- **R2C (Real-to-Complex)**: Gerçek girdi → Kompleks çıktı
- **C2R (Complex-to-Real)**: Kompleks girdi → Gerçek çıktı

### 2D/3D FFT
- Görüntü işleme ve bilimsel simülasyonlar için
- Row-major veya column-major düzenleme

## Performans İpuçları

1. **Plan Oluşturma**: Plan'ları bir kez oluşturup tekrar kullanın
2. **Batch İşleme**: Birden fazla FFT'yi aynı anda işleyin
3. **Memory Alignment**: Verilerinizi düzgün hizalayın
4. **Stream Kullanımı**: Asenkron işlemler için CUDA stream'leri kullanın
5. **Uygun Boyut Seçimi**: 2^n boyutları genellikle daha hızlıdır

## Sık Karşılaşılan Hatalar ve Çözümleri

### 1. "cufft.h not found"
```bash
# Include path'i ekleyin
nvcc -I/usr/local/cuda/include ...
```

### 2. "undefined reference to cufftPlan1d"
```bash
# -lcufft flag'ini ekleyin
nvcc ... -lcufft
```

### 3. Out of Memory
- Daha küçük batch boyutları kullanın
- cufftSetWorkArea() ile work area boyutunu ayarlayın

## Referanslar
- [NVIDIA cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [FFT Algorithm Basics](https://en.wikipedia.org/wiki/Fast_Fourier_transform)

## Build ve Çalıştırma

```bash
# Klasöre git
cd ~/git-projects/Cuda-Programming/23.cuFFT_tutor

# Tüm örnekleri derle
make all

# Veya tek tek derle
nvcc -o 01_simple_1d_fft 01_simple_1d_fft.cu -lcufft
./01_simple_1d_fft

# CMake ile build
mkdir build && cd build
cmake ..
make
```