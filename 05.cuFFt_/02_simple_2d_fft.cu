/**
 * @file 02_simple_2d_fft.cu
 * @brief 2D FFT örneği - Görüntü frekans analizi ve filtreleme
 * @author cuFFT Tutorial
 * 
 * Bu örnek:
 * 1. Sentetik bir görüntü oluşturur (çizgiler ve gürültü)
 * 2. 2D FFT uygular
 * 3. Frekans domeninde low-pass filtre uygular
 * 4. Inverse FFT ile görüntüyü geri dönüştürür
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

typedef float2 Complex;

// Görüntü boyutları (kare görüntü)
const int IMAGE_WIDTH = 256;
const int IMAGE_HEIGHT = 256;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

/**
 * @brief Test görüntüsü oluşturur (yatay/dikey çizgiler + gürültü)
 * @param image Çıktı görüntüsü
 * @param width Görüntü genişliği
 * @param height Görüntü yüksekliği
 */
void generateTestImage(Complex* image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // Temel görüntü: degradeyi gösterecek yumuşak geçişler
            float base = 0.5f * (sin(2.0f * M_PI * x / width * 3) + 
                                sin(2.0f * M_PI * y / height * 3));
            
            // Yüksek frekanslı gürültü ekle
            float noise = 0.2f * sin(2.0f * M_PI * x / width * 20) * 
                         sin(2.0f * M_PI * y / height * 20);
            
            // Görüntü değeri (sadece gerçek kısım)
            image[idx].x = base + noise;
            image[idx].y = 0.0f;
        }
    }
}

/**
 * @brief Low-pass filtre kernel'i (GPU'da çalışır)
 * @param spectrum FFT sonucu (frekans domeni)
 * @param width Görüntü genişliği
 * @param height Görüntü yüksekliği
 * @param cutoff_freq Kesim frekansı (0-1 arası)
 */
__global__ void lowPassFilter(Complex* spectrum, int width, int height, float cutoff_freq) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Frekans merkezi (DC component)
    int cx = width / 2;
    int cy = height / 2;
    
    // FFT shift için koordinat düzeltmesi
    int fx = (x < cx) ? x : x - width;
    int fy = (y < cy) ? y : y - height;
    
    // Merkeze olan mesafe (normalize edilmiş)
    float dist = sqrtf((float)(fx * fx + fy * fy)) / sqrtf((float)(cx * cx + cy * cy));
    
    // Low-pass filtre: yüksek frekansları sıfırla
    if (dist > cutoff_freq) {
        spectrum[idx].x = 0.0f;
        spectrum[idx].y = 0.0f;
    }
    // Yumuşak geçiş için Butterworth filtre
    else if (dist > cutoff_freq * 0.8f) {
        float attenuation = 1.0f / (1.0f + powf(dist / cutoff_freq, 6));
        spectrum[idx].x *= attenuation;
        spectrum[idx].y *= attenuation;
    }
}

/**
 * @brief Görüntü istatistiklerini hesaplar ve yazdırır
 * @param image Görüntü verisi
 * @param size Görüntü boyutu
 * @param title Başlık
 */
void printImageStats(Complex* image, int size, const char* title) {
    float min_val = image[0].x;
    float max_val = image[0].x;
    float avg_val = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float val = image[i].x;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        avg_val += val;
    }
    avg_val /= size;
    
    printf("\n%s İstatistikleri:\n", title);
    printf("  Min: %.4f\n", min_val);
    printf("  Max: %.4f\n", max_val);
    printf("  Ortalama: %.4f\n", avg_val);
}

/**
 * @brief Görüntünün küçük bir bölümünü ASCII art olarak yazdırır
 * @param image Görüntü verisi
 * @param width Görüntü genişliği
 * @param height Görüntü yüksekliği
 * @param title Başlık
 */
void printImagePreview(Complex* image, int width, int height, const char* title) {
    printf("\n%s (Sol üst köşe 16x8):\n", title);
    
    const char* chars = " .:-=+*#%@";
    int num_chars = strlen(chars);
    
    for (int y = 0; y < 8 && y < height; y++) {
        for (int x = 0; x < 16 && x < width; x++) {
            float val = image[y * width + x].x;
            // Normalize et [0, 1]
            val = (val + 1.0f) / 2.0f;
            if (val < 0) val = 0;
            if (val > 1) val = 1;
            
            int char_idx = (int)(val * (num_chars - 1));
            printf("%c", chars[char_idx]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== cuFFT 2D FFT Görüntü İşleme Örneği ===\n");
    printf("Görüntü boyutu: %dx%d\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("Toplam piksel: %d\n\n", IMAGE_SIZE);
    
    // 1. Host bellek ayırma
    Complex* h_image = (Complex*)malloc(sizeof(Complex) * IMAGE_SIZE);
    Complex* h_spectrum = (Complex*)malloc(sizeof(Complex) * IMAGE_SIZE);
    Complex* h_filtered = (Complex*)malloc(sizeof(Complex) * IMAGE_SIZE);
    
    // 2. Test görüntüsü oluştur
    generateTestImage(h_image, IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("Test görüntüsü oluşturuldu (düşük frekans + gürültü).\n");
    printImageStats(h_image, IMAGE_SIZE, "Orijinal Görüntü");
    printImagePreview(h_image, IMAGE_WIDTH, IMAGE_HEIGHT, "Orijinal Görüntü");
    
    // 3. Device bellek ayırma
    Complex* d_image;
    Complex* d_spectrum;
    CUDA_CHECK(cudaMalloc((void**)&d_image, sizeof(Complex) * IMAGE_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_spectrum, sizeof(Complex) * IMAGE_SIZE));
    
    // 4. Görüntüyü GPU'ya kopyala
    CUDA_CHECK(cudaMemcpy(d_image, h_image, sizeof(Complex) * IMAGE_SIZE,
                         cudaMemcpyHostToDevice));
    
    // 5. 2D FFT planı oluştur
    cufftHandle plan_forward, plan_inverse;
    
    // 2D FFT planı (Complex to Complex)
    CUFFT_CHECK(cufftPlan2d(&plan_forward, IMAGE_HEIGHT, IMAGE_WIDTH, CUFFT_C2C));
    CUFFT_CHECK(cufftPlan2d(&plan_inverse, IMAGE_HEIGHT, IMAGE_WIDTH, CUFFT_C2C));
    printf("\n2D FFT planları oluşturuldu.\n");
    
    // 6. Forward 2D FFT uygula (uzamsal -> frekans domeni)
    CUFFT_CHECK(cufftExecC2C(plan_forward, d_image, d_spectrum, CUFFT_FORWARD));
    printf("Forward 2D FFT tamamlandı.\n");
    
    // Spektrumu CPU'ya kopyala (analiz için)
    CUDA_CHECK(cudaMemcpy(h_spectrum, d_spectrum, sizeof(Complex) * IMAGE_SIZE,
                         cudaMemcpyDeviceToHost));
    
    // DC bileşeni (ortalama değer)
    float dc_component = magnitude(h_spectrum[0]) / IMAGE_SIZE;
    printf("\nDC bileşeni (ortalama): %.4f\n", dc_component);
    
    // 7. Frekans domeninde low-pass filtre uygula
    printf("\nLow-pass filtre uygulanıyor (cutoff: 0.2)...\n");
    
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x,
                  (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    lowPassFilter<<<gridSize, blockSize>>>(d_spectrum, IMAGE_WIDTH, IMAGE_HEIGHT, 0.2f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 8. Inverse 2D FFT (frekans -> uzamsal domain)
    CUFFT_CHECK(cufftExecC2C(plan_inverse, d_spectrum, d_image, CUFFT_INVERSE));
    printf("Inverse 2D FFT tamamlandı.\n");
    
    // 9. Filtrelenmiş görüntüyü CPU'ya kopyala
    CUDA_CHECK(cudaMemcpy(h_filtered, d_image, sizeof(Complex) * IMAGE_SIZE,
                         cudaMemcpyDeviceToHost));
    
    // 10. Normalizasyon (cuFFT inverse'de otomatik yapmaz)
    for (int i = 0; i < IMAGE_SIZE; i++) {
        h_filtered[i].x /= IMAGE_SIZE;
        h_filtered[i].y /= IMAGE_SIZE;
    }
    
    // 11. Sonuçları göster
    printImageStats(h_filtered, IMAGE_SIZE, "Filtrelenmiş Görüntü");
    printImagePreview(h_filtered, IMAGE_WIDTH, IMAGE_HEIGHT, "Filtrelenmiş Görüntü");
    
    // Gürültü azaltma oranını hesapla
    float noise_reduction = 0.0f;
    for (int i = 0; i < IMAGE_SIZE; i++) {
        float orig_noise = fabsf(h_image[i].x - h_filtered[i].x);
        noise_reduction += orig_noise;
    }
    noise_reduction /= IMAGE_SIZE;
    printf("\nOrtalama gürültü azaltma: %.4f\n", noise_reduction);
    
    // 12. Temizlik
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUFFT_CHECK(cufftDestroy(plan_inverse));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_spectrum));
    free(h_image);
    free(h_spectrum);
    free(h_filtered);
    
    printf("\n=== Program başarıyla tamamlandı ===\n");
    
    return 0;
}