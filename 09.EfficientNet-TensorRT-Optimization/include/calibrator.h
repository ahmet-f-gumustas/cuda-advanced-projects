#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <fstream>
#include <memory>

namespace efficientnet {

// INT8 Entropy Calibrator for TensorRT
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(
        const std::string& calibration_data_path,
        const std::string& cache_file,
        int batch_size,
        int input_width,
        int input_height,
        int input_channels = 3
    );

    ~Int8EntropyCalibrator();

    // Required interface methods
    int getBatchSize() const noexcept override { return batch_size_; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    const void* readCalibrationCache(size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    // Load image list from directory or file
    bool loadImageList(const std::string& path);

    // Load and preprocess single image
    bool loadImage(const std::string& filepath, float* buffer);

    int batch_size_;
    int input_width_;
    int input_height_;
    int input_channels_;
    size_t input_size_;

    std::vector<std::string> image_files_;
    int current_batch_ = 0;

    void* device_input_ = nullptr;
    std::vector<float> host_input_;

    std::string cache_file_;
    std::vector<char> calibration_cache_;
};

// MinMax Calibrator (alternative calibration method)
class Int8MinMaxCalibrator : public nvinfer1::IInt8MinMaxCalibrator {
public:
    Int8MinMaxCalibrator(
        const std::string& calibration_data_path,
        const std::string& cache_file,
        int batch_size,
        int input_width,
        int input_height,
        int input_channels = 3
    );

    ~Int8MinMaxCalibrator();

    int getBatchSize() const noexcept override { return batch_size_; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    const void* readCalibrationCache(size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    bool loadImageList(const std::string& path);
    bool loadImage(const std::string& filepath, float* buffer);

    int batch_size_;
    int input_width_;
    int input_height_;
    int input_channels_;
    size_t input_size_;

    std::vector<std::string> image_files_;
    int current_batch_ = 0;

    void* device_input_ = nullptr;
    std::vector<float> host_input_;

    std::string cache_file_;
    std::vector<char> calibration_cache_;
};

}  // namespace efficientnet

#endif  // CALIBRATOR_H
