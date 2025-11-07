# CUDA Parallel ML Models - Paralel Makine Öğrenmesi Modelleri

## 🎯 Proje Amacı

Bu proje, **CUDA kullanarak iki farklı makine öğrenmesi modelinin paralel olarak eğitilmesini** gösterir. Proje, GPU programlama ve paralel işleme kavramlarını öğrenmek isteyenler için tasarlanmıştır.

### Kullanılan Modeller:
1. **Linear Regression (Doğrusal Regresyon)** - En uygun doğru çizgisini bulma
2. **K-Means Clustering** - Verileri kümelere ayırma

## 🚀 Öne Çıkan Özellikler

- ✅ **Tam CUDA implementasyonu** - Her model GPU üzerinde çalışır
- ✅ **Paralel eğitim** - İki model aynı anda, farklı CUDA stream'lerinde eğitilir
- ✅ **OpenGL görselleştirme** - Gerçek zamanlı sonuç görüntüleme
- ✅ **Modern C++17** ve CUDA
- ✅ **CMake build sistemi**

## 📊 Modeller Hakkında

### 1. Linear Regression (Doğrusal Regresyon)

**Ne yapar?**
- Verilen noktalara en uygun doğru çizgisini bulur
- Formül: `y = mx + b` (m: eğim, b: kesişim noktası)

**CUDA Paralelleştirmesi:**
- Her örnek için tahmin hesaplama paralel
- Gradient hesaplama paralel
- Ağırlık güncelleme paralel

**Kernel'ler:**
- `predictKernel`: y_pred = w*x + b hesaplar
- `computeGradientsKernel`: Gradientleri hesaplar
- `updateWeightsKernel`: Ağırlıkları günceller
- `computeLossKernel`: MSE loss'u hesaplar (shared memory reduction ile)

### 2. K-Means Clustering

**Ne yapar?**
- Verileri benzerliklerine göre K tane kümeye ayırır
- Her kümenin bir merkez noktası (centroid) vardır

**CUDA Paralelleştirmesi:**
- Her veri noktası için en yakın merkezi bulma paralel
- Yeni merkezleri hesaplama paralel
- Inertia (küme içi mesafe toplamı) hesaplama paralel

**Kernel'ler:**
- `assignClustersKernel`: Her noktayı en yakın kümeye atar
- `updateCentroidsKernel`: Yeni küme merkezlerini hesaplar
- `computeInertiaKernel`: Toplam küme içi mesafeyi hesaplar

## 🔧 Kurulum

### Gereksinimler

- CUDA Toolkit (11.0+)
- CMake (3.18+)
- C++17 uyumlu derleyici (GCC 9+, Clang 10+)
- OpenGL
- GLEW
- GLFW3
- NVIDIA GPU (Compute Capability 7.5+)

### Ubuntu/Debian Kurulumu

```bash
# CUDA Toolkit yüklü olduğunu varsayıyoruz
sudo apt update
sudo apt install cmake build-essential
sudo apt install libglew-dev libglfw3-dev libgl1-mesa-dev
```

### Derleme

```bash
cd 07-parallel-ml-models
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 🎮 Çalıştırma

```bash
./parallel_ml_models
```

### Beklenen Çıktı

Program çalıştığında:
1. Model parametreleri başlatılır
2. Eğitim verisi oluşturulur
3. İki model paralel olarak eğitilir (farklı thread'lerde)
4. OpenGL penceresi açılır ve sonuçlar görselleştirilir

**Görselleştirme:**
- **Sol panel**: Linear Regression - Mavi noktalar (veri), kırmızı çizgi (model)
- **Sağ panel**: K-Means - Renkli noktalar (veriler), beyaz merkezli noktalar (centroid'ler)

**Çıkmak için:** ESC tuşuna basın

## 📁 Proje Yapısı

```
07-parallel-ml-models/
├── include/
│   ├── linear_regression.h    # Linear Regression model tanımı
│   ├── kmeans.h               # K-Means model tanımı
│   ├── model_manager.h        # Paralel eğitim koordinatörü
│   └── visualizer.h           # OpenGL görselleştirme
├── src/
│   ├── linear_regression.cu   # Linear Regression CUDA implementasyonu
│   ├── kmeans.cu              # K-Means CUDA implementasyonu
│   ├── model_manager.cpp      # Model koordinasyon kodu
│   ├── visualizer.cpp         # Görselleştirme kodu
│   └── main.cpp               # Ana program
├── CMakeLists.txt
└── README.md
```

## 🧠 Paralel İşleme Nasıl Çalışıyor?

### 1. Thread-Level Parallelism (CPU)

```cpp
// Model Manager içinde
std::thread linearThread(&ModelManager::trainLinearModel, this);
std::thread kmeansThread(&ModelManager::trainKMeansModel, this);
```

İki ayrı CPU thread'i, iki farklı modeli eğitir.

### 2. CUDA Stream-Level Parallelism (GPU)

Her model kendi CUDA stream'ini kullanır:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
// Kernel çağrıları bu stream üzerinde
kernel<<<grid, block, 0, stream>>>(...);
```

Bu sayede GPU, iki modelin kernel'lerini **aynı anda** çalıştırabilir!

### 3. GPU Parallelism (CUDA Kernels)

Her kernel içinde binlerce thread paralel çalışır:

```cuda
__global__ void predictKernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        // Her thread bir örneği işler
        predictions[idx] = compute(...);
    }
}
```

## 📈 Performans Optimizasyonları

### 1. Shared Memory Kullanımı

Loss ve inertia hesaplamalarında **reduction pattern** ile shared memory kullanılır:

```cuda
extern __shared__ float sharedData[];
// Her thread kendi sonucunu shared memory'e yazar
sharedData[tid] = localResult;
__syncthreads();
// Reduction ile toplam hesaplanır
```

### 2. Coalesced Memory Access

Veri düzeni, GPU bellek erişimlerini optimize eder:
- Ardışık thread'ler ardışık bellek adreslerine erişir
- `X[idx * numFeatures + f]` düzeni kullanılır

### 3. Asynchronous Operations

CPU-GPU veri transferi asenkron yapılır:

```cpp
cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream);
kernel<<<...>>>(...);
cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, stream);
```

## 🎓 Öğrenme Kaynakları

### Anlaşılması Gereken Kavramlar:

1. **CUDA Threads ve Blocks**
   - Thread: GPU'da paralel çalışan en küçük birim
   - Block: Thread grupları
   - Grid: Block grupları

2. **Memory Hierarchy**
   - Global Memory: Yavaş ama büyük
   - Shared Memory: Hızlı ama sınırlı
   - Registers: En hızlı ama çok sınırlı

3. **Synchronization**
   - `__syncthreads()`: Block içi senkronizasyon
   - `cudaStreamSynchronize()`: Stream senkronizasyonu

4. **Gradient Descent**
   - Loss fonksiyonunu minimize etmek için iteratif algoritma
   - Her iterasyonda: Forward pass → Compute gradient → Update weights

5. **K-Means Algorithm**
   - 1. Adım: Her noktayı en yakın merkeze ata
   - 2. Adım: Yeni merkezleri hesapla
   - Yakınsama olana kadar tekrarla

## 🔍 Kod İnceleme Önerileri

1. **Önce basit kernel'lere bakın:**
   - `predictKernel` (linear_regression.cu)
   - `assignClustersKernel` (kmeans.cu)

2. **Reduction pattern'i anlayın:**
   - `computeLossKernel` fonksiyonunu inceleyin
   - Shared memory kullanımını gözlemleyin

3. **Paralel koordinasyonu inceleyin:**
   - `model_manager.cpp` dosyasındaki thread yönetimi
   - CUDA stream kullanımı

## 🐛 Sorun Giderme

### CUDA Out of Memory
- Batch size'ı küçültün
- Daha az örnek kullanın

### Derleme Hataları
- CUDA Toolkit kurulu mu kontrol edin: `nvcc --version`
- CMake versiyonu: `cmake --version` (3.18+ olmalı)

### Görselleştirme Açılmıyor
- OpenGL sürücüleri kurulu mu?
- `glxinfo | grep "OpenGL version"` ile kontrol edin

### Düşük Performans
- GPU kullanımını kontrol edin: `nvidia-smi`
- Compute capability'nizi kontrol edin ve CMakeLists.txt'de ayarlayın

## 📝 Geliştirme Fikirleri

1. **Daha fazla model ekleyin:**
   - Logistic Regression
   - Neural Network (basit MLP)
   - SVM (Support Vector Machine)

2. **Optimizasyon teknikleri:**
   - Mini-batch gradient descent
   - Momentum optimizer
   - Adam optimizer

3. **Daha iyi görselleştirme:**
   - Loss grafiği
   - Confusion matrix
   - 3D görselleştirme

4. **Model karşılaştırması:**
   - Eğitim sürelerini ölçün
   - Accuracy karşılaştırması
   - GPU vs CPU performans karşılaştırması

## 📚 Referanslar

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Machine Learning on GPU](https://developer.nvidia.com/deep-learning)

## 📄 Lisans

Bu proje eğitim amaçlıdır ve serbestçe kullanılabilir.

## 👤 Geliştirici Notları

Bu proje, CUDA programlama ve paralel makine öğrenmesinin temellerini öğretmek için tasarlanmıştır. Kodlar **sadelik** ve **anlaşılabilirlik** göz önünde bulundurularak yazılmıştır. Production ortamında daha fazla optimizasyon ve hata kontrolü gerekebilir.

**Keyifli öğrenmeler! 🚀**
