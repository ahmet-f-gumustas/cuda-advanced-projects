/**
 * @file 03_fft_benchmark.cu
 * @brief CPU vs GPU FFT performans karşılaştırması
 * @author cuFFT Tutorial
 * 
 * Bu örnek:
 * 1. CPU'da basit DFT implementasyonu
 * 2. GPU'da cuFFT kullanımı
 * 3. Farklı boyutlar için performans ölçümü
 * 4. Speedup hesaplama
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <time.h>
#include <complex.h>

// Hata kontrolü makroları
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA hatası %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult error = call; \
        if (error != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT hatası %s:%d: %d\n", __FILE__, __LINE__, error); \
            exit(1); \
        } \
    } while(0)

typedef float2 Complex;

// Zaman ölçümü için yardımcı yapı
typedef struct {
    cudaEvent_t start, stop;
    clock_t cpu_start, cpu_end;
} Timer;

/**
 * @brief Timer başlat (GPU için)
 */
void startGPUTimer(Timer* timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
    cudaEventRecord(timer->start, 0);
}

/**
 * @brief GPU timer'ı durdur ve süreyi döndür (ms)
 */
float stopGPUTimer(Timer* timer) {
    cudaEventRecord(timer->stop, 0);
    cudaEventSynchronize(timer->stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, timer->start, timer->stop);
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
    return elapsedTime;
}

/**
 * @brief CPU timer başlat
 */
void startCPUTimer(Timer* timer) {
    timer->cpu_start = clock();
}

/**
 * @brief CPU timer'ı durdur ve süreyi döndür (ms)
 */
float stopCPUTimer(Timer* timer) {
    timer->cpu_end = clock();
    return ((float)(timer->cpu_end - timer->cpu_start)) / CLOCKS_PER_SEC * 1000.0f;
}

/**
 * @brief Basit DFT implementasyonu (CPU'da, eğitim amaçlı)
 * Not: Bu çok yavaş bir implementasyon, sadece karşılaştırma için
 * @param input Girdi sinyali
 * @param output Çıktı spektrumu
 * @param n Sinyal boyutu
 */
void cpuDFT(float complex* input, float complex* output, int n) {
    for (int k = 0; k < n; k++) {
        output[k] = 0;
        for (int j = 0; j < n; j++) {
            float angle = -2.0f * M_PI * k * j / n;
            float complex W = cosf(angle) + I * sinf(angle);
            output[k] += input[j] * W;
        }
    }
}

/**
 * @brief Rastgele sinyal oluştur
 */
void generateRandomSignal(Complex* signal, int size) {
    for (int i = 0; i < size; i++) {
        signal[i].x = (float)rand() / RAND_MAX;
        signal[i].y = 0.0f;
    }
}

/**
 * @brief FFT benchmark fonksiyonu
 * @param size FFT boyutu
 * @param num_iterations Test sayısı
 */
void benchmarkFFT(int size, int num_iterations) {
    printf("\n=== FFT Boyutu: %d ===\n", size);
    
    // Host bellek ayırma
    Complex* h_signal = (Complex*)malloc(sizeof(Complex) * size);
    Complex* h_result_gpu = (Complex*)malloc(sizeof(Complex) * size);
    float complex* h_signal_cpu = (float complex*)malloc(sizeof(float complex) * size);
    float complex* h_result_cpu = (float complex*)malloc(sizeof(float complex) * size);
    
    // Rastgele sinyal oluştur
    generateRandomSignal(h_signal, size);
    
    // CPU için veriyi kopyala
    for (int i = 0; i < size; i++) {
        h_signal_cpu[i] = h_signal[i].x + I * h_signal[i].y;
    }
    
    Timer timer;
    float cpu_time = 0.0f;
    float gpu_time = 0.0f;
    float gpu_time_with_transfer = 0.0f;
    
    // ========== CPU DFT (sadece küçük boyutlar için) ==========
    if (size <= 1024) {  // Büyük boyutlar için çok yavaş
        printf("CPU DFT hesaplanıyor...\n");
        
        startCPUTimer(&timer);
        for (int iter = 0; iter < num_iterations; iter++) {
            cpuDFT(h_signal_cpu, h_result_cpu, size);
        }
        cpu_time = stopCPUTimer(&timer) / num_iterations;
        
        printf("  CPU DFT süresi: %.3f ms\n", cpu_time);
    } else {
        printf("  CPU DFT atlandı (boyut çok büyük)\n");
    }
    
    // ========== GPU cuFFT ==========
    printf("GPU cuFFT hesaplanıyor...\n");
    
    // Device bellek ayırma
    Complex* d_signal;
    Complex* d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_signal, sizeof(Complex) * size));
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(Complex) * size));
    
    // cuFFT planı oluştur
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, size, CUFFT_C2C, 1));
    
    // Veri transferi dahil ölçüm
    startGPUTimer(&timer);
    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemcpy(d_signal, h_signal, sizeof(Complex) * size,
                             cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD));
        CUDA_CHECK(cudaMemcpy(h_result_gpu, d_result, sizeof(Complex) * size,
                             cudaMemcpyDeviceToHost));
    }
    gpu_time_with_transfer = stopGPUTimer(&timer) / num_iterations;
    
    // Sadece FFT hesaplama süresi (transfer hariç)
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal, sizeof(Complex) * size,
                         cudaMemcpyHostToDevice));
    
    startGPUTimer(&timer);
    for (int iter = 0; iter < num_iterations; iter++) {
        CUFFT_CHECK(cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_time = stopGPUTimer(&timer) / num_iterations;
    
    printf("  GPU cuFFT süresi (sadece hesaplama): %.3f ms\n", gpu_time);
    printf("  GPU cuFFT süresi (transfer dahil): %.3f ms\n", gpu_time_with_transfer);
    
    // Speedup hesapla
    if (size <= 1024 && cpu_time > 0) {
        float speedup_compute = cpu_time / gpu_time;
        float speedup_total = cpu_time / gpu_time_with_transfer;
        printf("  Speedup (sadece hesaplama): %.2fx\n", speedup_compute);
        printf("  Speedup (transfer dahil): %.2fx\n", speedup_total);
    }
    
    // Bellek kullanımı
    float memory_mb = (sizeof(Complex) * size * 2) / (1024.0f * 1024.0f);
    printf("  GPU bellek kullanımı: %.2f MB\n", memory_mb);
    
    // Throughput hesapla (GFLOPS)
    // FFT complexity: O(n log n), yaklaşık 5n log2(n) floating point işlem
    float flops = 5.0f * size * log2f((float)size);
    float gflops = (flops * num_iterations) / (gpu_time * 1e6);
    printf("  GPU Throughput: %.2f GFLOPS\n", gflops);
    
    // Temizlik
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_result));
    free(h_signal);
    free(h_result_gpu);
    free(h_signal_cpu);
    free(h_result_cpu);
}

/**
 * @brief GPU özelliklerini göster
 */
void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("CUDA destekli GPU bulunamadı!\n");
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("\n=== GPU Bilgileri ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor sayısı: %d\n", prop.multiProcessorCount);
    printf("Toplam global bellek: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
}

int main() {
    printf("=== cuFFT Performans Benchmark ===\n");
    
    // Rastgele sayı üreteci başlat
    srand(time(NULL));
    
    // GPU bilgilerini göster
    printGPUInfo();
    
    // Test parametreleri
    int test_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_iterations = 100;
    
    printf("\n=== Benchmark Başlıyor ===\n");
    printf("İterasyon sayısı: %d\n", num_iterations);
    
    // Farklı boyutlar için benchmark
    for (int i = 0; i < num_sizes; i++) {
        benchmarkFFT(test_sizes[i], num_iterations);
    }
    
    printf("\n=== Özet ===\n");
    printf("• Küçük boyutlar için CPU ve GPU benzer performans gösterebilir\n");
    printf("• Veri transferi overhead'i küçük boyutlarda belirgin\n");
    printf("• Büyük boyutlarda GPU çok daha hızlı (100x+ speedup mümkün)\n");
    printf("• Batch işlemler GPU performansını daha da artırır\n");
    
    printf("\n=== Benchmark tamamlandı ===\n");
    
    return 0;
}