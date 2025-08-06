#include "backends/tensorrt/int8_calibrator.h"
#include "common/cuda_utils.h"
#include "common/logger.h"
#include <fstream>
#include <algorithm>

Int8EntropyCalibrator::Int8EntropyCalibrator(
    int batch_size, int input_h, int input_w,
    const std::string& calib_list_file,
    const std::string& cache_file)
    : batch_size_(batch_size),
      input_h_(input_h),
      input_w_(input_w),
      cache_file_(cache_file),
      image_index_(0) {
    
    // Kalibrasyon listesini oku
    std::ifstream file(calib_list_file);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            calib_files_.push_back(line);
        }
    }
    
    LOG_INFO("Loaded", calib_files_.size(), "calibration images");
    
    // CUDA buffer ayır
    size_t input_size = batch_size_ * 3 * input_h_ * input_w_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&device_input_, input_size));
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    cudaFree(device_input_);
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (image_index_ >= calib_files_.size()) {
        return false;
    }
    
    // Batch için görüntüleri yükle ve preprocess et
    std::vector<float> batch_data;
    for (int i = 0; i < batch_size_ && image_index_ < calib_files_.size(); ++i) {
        // Görüntüyü yükle ve preprocess et
        // Bu kısım basitleştirilmiş - gerçek implementasyonda
        // image decoder ve preprocessor kullanılmalı
        LOG_DEBUG("Processing calibration image:", calib_files_[image_index_]);
        image_index_++;
    }
    
    // Device'a kopyala
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    calib_cache_.clear();
    std::ifstream file(cache_file_, std::ios::binary);
    if (file.good()) {
        file.seekg(0, std::ios::end);
        length = file.tellg();
        file.seekg(0, std::ios::beg);
        calib_cache_.resize(length);
        file.read(calib_cache_.data(), length);
        LOG_INFO("Read calibration cache from:", cache_file_);
    }
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream file(cache_file_, std::ios::binary);
    file.write(static_cast<const char*>(cache), length);
    LOG_INFO("Wrote calibration cache to:", cache_file_);
}