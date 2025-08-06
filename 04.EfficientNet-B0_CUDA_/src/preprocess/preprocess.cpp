#include "preprocess/preprocess.h"
#include "common/cuda_utils.h"
#include "common/logger.h"
#include <algorithm>

Preprocessor::Preprocessor(int target_height, int target_width, bool use_fp16)
    : target_height_(target_height), target_width_(target_width), use_fp16_(use_fp16) {
    
    // Ara buffer için bellek ayır
    resized_size_ = target_height_ * target_width_ * 3 * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_resized_, resized_size_));
    
    LOG_INFO("Preprocessor initialized: ", target_height_, "x", target_width_,
             use_fp16_ ? " (FP16)" : " (FP32)");
}

Preprocessor::~Preprocessor() {
    cudaFree(d_resized_);
}

void Preprocessor::process(const uint8_t* d_input, int input_height, int input_width,
                          void* d_output, cudaStream_t stream) {
    // Center crop için boyutları hesapla
    float scale = std::max(static_cast<float>(target_height_) / input_height,
                          static_cast<float>(target_width_) / input_width);
    
    int scaled_height = static_cast<int>(input_height * scale);
    int scaled_width = static_cast<int>(input_width * scale);
    
    // Normalizasyon parametrelerini device'a kopyala
    float* d_mean;
    float* d_std;
    CUDA_CHECK(cudaMalloc(&d_mean, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_mean, norm_params_.mean, 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_std, norm_params_.std, 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    
    // Resize + normalize + HWC->NCHW dönüşümü
    if (use_fp16_) {
        resize_bilinear_normalize_hwc2nchw_fp16(
            d_input, input_height, input_width,
            static_cast<__half*>(d_output), target_height_, target_width_,
            d_mean, d_std, stream);
    } else {
        resize_bilinear_normalize_hwc2nchw(
            d_input, input_height, input_width,
            static_cast<float*>(d_output), target_height_, target_width_,
            d_mean, d_std, stream);
    }
    
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_std));
}

void Preprocessor::process_batch(const std::vector<const uint8_t*>& d_inputs,
                               const std::vector<std::pair<int, int>>& input_sizes,
                               void* d_output, int batch_size, cudaStream_t stream) {
    // Batch işleme - her görüntüyü ayrı process et
    size_t single_output_size = get_output_size();
    
    for (int i = 0; i < batch_size; ++i) {
        char* output_ptr = static_cast<char*>(d_output) + i * single_output_size;
        process(d_inputs[i], input_sizes[i].first, input_sizes[i].second,
                output_ptr, stream);
    }
}