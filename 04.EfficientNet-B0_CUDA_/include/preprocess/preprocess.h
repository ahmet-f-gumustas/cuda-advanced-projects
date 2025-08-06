#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// ImageNet normalizasyon parametreleri
struct NormParams {
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

class Preprocessor {
public:
    Preprocessor(int target_height = 224, int target_width = 224, 
                 bool use_fp16 = false);
    ~Preprocessor();
    
    // HWC uint8 RGB girişi -> NCHW float çıkış (device memory)
    void process(const uint8_t* d_input, int input_height, int input_width,
                 void* d_output, cudaStream_t stream = 0);
    
    // Batch işleme
    void process_batch(const std::vector<const uint8_t*>& d_inputs,
                      const std::vector<std::pair<int, int>>& input_sizes,
                      void* d_output, int batch_size, cudaStream_t stream = 0);
    
    // Çıkış boyutu hesaplama
    size_t get_output_size() const {
        return target_height_ * target_width_ * 3 * (use_fp16_ ? 2 : 4);
    }
    
    size_t get_batch_output_size(int batch_size) const {
        return batch_size * get_output_size();
    }
    
private:
    int target_height_;
    int target_width_;
    bool use_fp16_;
    NormParams norm_params_;
    
    // Ara bufferlar
    void* d_resized_;  // Resize edilmiş görüntü
    size_t resized_size_;
};

// CUDA kernel fonksiyonları (resize_norm.cu'da implement edilecek)
void resize_bilinear_normalize_hwc2nchw(
    const uint8_t* d_input, int input_height, int input_width,
    float* d_output, int output_height, int output_width,
    const float* mean, const float* std,
    cudaStream_t stream);

void resize_bilinear_normalize_hwc2nchw_fp16(
    const uint8_t* d_input, int input_height, int input_width,
    __half* d_output, int output_height, int output_width,
    const float* mean, const float* std,
    cudaStream_t stream);