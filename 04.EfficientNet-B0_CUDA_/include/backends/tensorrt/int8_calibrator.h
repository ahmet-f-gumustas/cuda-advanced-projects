#pragma once

#include <NvInfer.h>
#include <vector>
#include <string>

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(int batch_size, int input_h, int input_w,
                         const std::string& calib_list_file,
                         const std::string& cache_file);
    ~Int8EntropyCalibrator();
    
    // IInt8Calibrator interface
    int getBatchSize() const noexcept override { return batch_size_; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;
    
private:
    int batch_size_;
    int input_h_;
    int input_w_;
    std::string cache_file_;
    
    std::vector<std::string> calib_files_;
    size_t image_index_;
    
    void* device_input_;
    std::vector<char> calib_cache_;
};