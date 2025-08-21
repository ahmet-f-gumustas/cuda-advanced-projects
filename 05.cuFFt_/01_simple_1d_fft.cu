/**
 * @file 01_simple_1d_fft.cu
 * @brief Basit 1D FFT örneği - Sinüs dalgasının frekans analizi
 * @author cuFFT Tutorial
 * 
 * Bu örnek:
 * 1. CPU'da bir sinüs dalgası oluşturur
 * 2. Veriyi GPU'ya kopyalar
 * 3. FFT uygular (zaman domeninden frekans domenine)
 * 4. Sonuçları gösterir
 * 5. Inverse FFT ile geri dönüşüm yapar
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

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

// Kompleks sayı yapısı (cuFFT'nin kullandığı)
typedef float2 Complex;

// Sinyal parametreleri
const int SIGNAL_SIZE = 1024;      // Sinyal uzunluğu (2^n tercih edilir)
const float SAMPLE_RATE = 1000.0f; // Örnekleme frekansı (Hz)
const float FREQUENCY1 = 50.0f;    // İlk sinüs dalgası frekansı (Hz)
const float FREQUENCY2 = 120.0f;   // İkinci sinüs dalgası frekansı (Hz)

/**
 * @brief Test sinyali oluşturur (iki sinüs dalgasının toplamı)
 * @param signal Çıktı sinyali
 * @param size Sinyal boyutu
 */
void generateSignal(Complex* signal, int size) {
    for (int i = 0; i < size; i++) {
        float t = i / SAMPLE_RATE; // Zaman (saniye)
        
        // İki sinüs dalgasının toplamı (sadece gerçek kısım)
        signal[i].x = sin(2.0f * M_PI * FREQUENCY1 * t) + 
                     0.5f * sin(2.0f * M_PI * FREQUENCY2 * t);
        signal[i].y = 0.0f; // İmajiner kısım sıfır
    }
}

/**
 * @brief FFT sonucunun magnitude'ünü hesaplar
 * @param complex Kompleks sayı
 * @return Magnitude değeri
 */
float magnitude(Complex c) {
    return sqrtf(c.x * c.x + c.y * c.y);
}

/**
 * @brief En yüksek magnitude'a sahip frekansları bulur ve yazdırır
 * @param spectrum FFT sonucu
 * @param size Spektrum boyutu
 */
void findPeakFrequencies(Complex* spectrum, int size) {
    printf("\n=== Tespit Edilen Frekanslar ===\n");
    
    // Sadece ilk yarıyı kontrol et (Nyquist teoremi)
    int half_size = size / 2;
    
    for (int i = 1; i < half_size; i++) {
        float mag = magnitude(spectrum[i]);
        float freq = (float)i * SAMPLE_RATE / size;
        
        // Önemli bir peak bulduk mu? (threshold: ortalama magnitude'ün 10 katı)
        if (mag > size * 0.1f) {
            printf("Frekans: %.2f Hz, Magnitude: %.2f\n", freq, mag);
        }
    }
}

int main() {
    printf("=== cuFFT 1D FFT Örneği ===\n");
    printf("Sinyal boyutu: %d\n", SIGNAL_SIZE);
    printf("Örnekleme frekansı: %.1f Hz\n", SAMPLE_RATE);
    printf("Test frekansları: %.1f Hz ve %.1f Hz\n\n", FREQUENCY1, FREQUENCY2);
    
    // 1. Host (CPU) bellek ayırma
    Complex* h_signal = (Complex*)malloc(sizeof(Complex) * SIGNAL_SIZE);
    Complex* h_result = (Complex*)malloc(sizeof(Complex) * SIGNAL_SIZE);
    Complex* h_recovered = (Complex*)malloc(sizeof(Complex) * SIGNAL_SIZE);
    
    // 2. Test sinyali oluştur
    generateSignal(h_signal, SIGNAL_SIZE);
    printf("Test sinyali oluşturuldu.\n");
    
    // İlk birkaç örnek değeri göster
    printf("\nİlk 5 sinyal değeri:\n");
    for (int i = 0; i < 5; i++) {
        printf("  signal[%d] = %.4f + %.4fi\n", i, h_signal[i].x, h_signal[i].y);
    }
    
    // 3. Device (GPU) bellek ayırma
    Complex* d_signal;
    Complex* d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_signal, sizeof(Complex) * SIGNAL_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(Complex) * SIGNAL_SIZE));
    
    // 4. Veriyi GPU'ya kopyala
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal, sizeof(Complex) * SIGNAL_SIZE,
                         cudaMemcpyHostToDevice));
    printf("\nVeri GPU'ya kopyalandı.\n");
    
    // 5. cuFFT plan oluştur
    cufftHandle plan_forward, plan_inverse;
    
    // Forward FFT planı (Complex to Complex)
    CUFFT_CHECK(cufftPlan1d(&plan_forward, SIGNAL_SIZE, CUFFT_C2C, 1));
    printf("Forward FFT planı oluşturuldu.\n");
    
    // Inverse FFT planı
    CUFFT_CHECK(cufftPlan1d(&plan_inverse, SIGNAL_SIZE, CUFFT_C2C, 1));
    printf("Inverse FFT planı oluşturuldu.\n");
    
    // 6. Forward FFT uygula (zaman -> frekans)
    CUFFT_CHECK(cufftExecC2C(plan_forward, d_signal, d_result, CUFFT_FORWARD));
    printf("\nForward FFT tamamlandı.\n");
    
    // 7. Sonuçları CPU'ya kopyala
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(Complex) * SIGNAL_SIZE,
                         cudaMemcpyDeviceToHost));
    
    // 8. Frekans spektrumunu analiz et
    findPeakFrequencies(h_result, SIGNAL_SIZE);
    
    // 9. Inverse FFT uygula (frekans -> zaman)
    CUFFT_CHECK(cufftExecC2C(plan_inverse, d_result, d_signal, CUFFT_INVERSE));
    printf("\nInverse FFT tamamlandı.\n");
    
    // 10. Geri dönüştürülmüş sinyali al
    CUDA_CHECK(cudaMemcpy(h_recovered, d_signal, sizeof(Complex) * SIGNAL_SIZE,
                         cudaMemcpyDeviceToHost));
    
    // 11. Normalizasyon (cuFFT inverse FFT'de otomatik normalizasyon yapmaz)
    for (int i = 0; i < SIGNAL_SIZE; i++) {
        h_recovered[i].x /= SIGNAL_SIZE;
        h_recovered[i].y /= SIGNAL_SIZE;
    }
    
    // 12. Orijinal ve geri dönüştürülmüş sinyali karşılaştır
    printf("\n=== Geri Dönüşüm Doğrulaması ===\n");
    float max_error = 0.0f;
    for (int i = 0; i < SIGNAL_SIZE; i++) {
        float error = fabsf(h_signal[i].x - h_recovered[i].x);
        if (error > max_error) max_error = error;
    }
    printf("Maksimum hata: %.6e\n", max_error);
    
    if (max_error < 1e-5) {
        printf("✓ Sinyal başarıyla geri dönüştürüldü!\n");
    } else {
        printf("✗ Geri dönüşümde hata var!\n");
    }
    
    // 13. Temizlik
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUFFT_CHECK(cufftDestroy(plan_inverse));
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_result));
    free(h_signal);
    free(h_result);
    free(h_recovered);
    
    printf("\n=== Program başarıyla tamamlandı ===\n");
    
    return 0;
}