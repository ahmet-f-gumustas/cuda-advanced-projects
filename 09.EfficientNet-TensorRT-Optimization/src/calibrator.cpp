#include "calibrator.h"
#include "cuda_preprocess.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace efficientnet {

// ============================================================================
// Int8EntropyCalibrator Implementation
// ============================================================================

Int8EntropyCalibrator::Int8EntropyCalibrator(
    const std::string& calibration_data_path,
    const std::string& cache_file,
    int batch_size,
    int input_width,
    int input_height,
    int input_channels
) : batch_size_(batch_size),
    input_width_(input_width),
    input_height_(input_height),
    input_channels_(input_channels),
    cache_file_(cache_file)
{
    input_size_ = batch_size_ * input_channels_ * input_width_ * input_height_;
    host_input_.resize(input_size_);

    // Allocate device memory
    cudaMalloc(&device_input_, input_size_ * sizeof(float));

    // Load image list
    loadImageList(calibration_data_path);

    std::cout << "Calibrator initialized with " << image_files_.size()
              << " images, batch size " << batch_size_ << std::endl;
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    if (device_input_) {
        cudaFree(device_input_);
    }
}

bool Int8EntropyCalibrator::loadImageList(const std::string& path) {
    namespace fs = std::filesystem;

    if (fs::is_directory(path)) {
        // Load all images from directory
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_files_.push_back(entry.path().string());
                }
            }
        }
    } else if (fs::is_regular_file(path)) {
        // Load image list from text file
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && fs::exists(line)) {
                image_files_.push_back(line);
            }
        }
    }

    // Shuffle for better calibration
    std::random_shuffle(image_files_.begin(), image_files_.end());

    return !image_files_.empty();
}

bool Int8EntropyCalibrator::loadImage(const std::string& filepath, float* buffer) {
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 3);

    if (!data) {
        std::cerr << "Failed to load image: " << filepath << std::endl;
        return false;
    }

    // Resize and normalize on CPU for calibration
    // Using simple bilinear interpolation
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    int plane_size = input_width_ * input_height_;

    for (int y = 0; y < input_height_; y++) {
        for (int x = 0; x < input_width_; x++) {
            float src_x = (x + 0.5f) * width / input_width_ - 0.5f;
            float src_y = (y + 0.5f) * height / input_height_ - 0.5f;

            src_x = std::max(0.0f, std::min(src_x, static_cast<float>(width - 1)));
            src_y = std::max(0.0f, std::min(src_y, static_cast<float>(height - 1)));

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < 3; c++) {
                float v00 = data[(y0 * width + x0) * 3 + c];
                float v01 = data[(y0 * width + x1) * 3 + c];
                float v10 = data[(y1 * width + x0) * 3 + c];
                float v11 = data[(y1 * width + x1) * 3 + c];

                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float val = v0 * (1 - dy) + v1 * dy;

                // Normalize and store in NCHW format
                buffer[c * plane_size + y * input_width_ + x] =
                    (val / 255.0f - mean[c]) / std_dev[c];
            }
        }
    }

    stbi_image_free(data);
    return true;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[],
                                      int nbBindings) noexcept {
    if (current_batch_ >= static_cast<int>(image_files_.size()) / batch_size_) {
        return false;
    }

    int start_idx = current_batch_ * batch_size_;
    int end_idx = std::min(start_idx + batch_size_,
                           static_cast<int>(image_files_.size()));

    int image_size = input_channels_ * input_width_ * input_height_;

    for (int i = start_idx; i < end_idx; i++) {
        float* buffer = host_input_.data() + (i - start_idx) * image_size;
        if (!loadImage(image_files_[i], buffer)) {
            // Fill with zeros if image fails to load
            std::fill(buffer, buffer + image_size, 0.0f);
        }
    }

    // Copy to device
    cudaMemcpy(device_input_, host_input_.data(),
               input_size_ * sizeof(float), cudaMemcpyHostToDevice);

    bindings[0] = device_input_;
    current_batch_++;

    std::cout << "Calibration batch " << current_batch_ << "/"
              << (image_files_.size() / batch_size_) << std::endl;

    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    std::ifstream file(cache_file_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        length = 0;
        return nullptr;
    }

    length = file.tellg();
    file.seekg(0, std::ios::beg);

    calibration_cache_.resize(length);
    file.read(calibration_cache_.data(), length);

    std::cout << "Read calibration cache from: " << cache_file_ << std::endl;
    return calibration_cache_.data();
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache,
                                                    size_t length) noexcept {
    std::ofstream file(cache_file_, std::ios::binary);
    if (file.is_open()) {
        file.write(static_cast<const char*>(cache), length);
        std::cout << "Wrote calibration cache to: " << cache_file_ << std::endl;
    }
}

// ============================================================================
// Int8MinMaxCalibrator Implementation
// ============================================================================

Int8MinMaxCalibrator::Int8MinMaxCalibrator(
    const std::string& calibration_data_path,
    const std::string& cache_file,
    int batch_size,
    int input_width,
    int input_height,
    int input_channels
) : batch_size_(batch_size),
    input_width_(input_width),
    input_height_(input_height),
    input_channels_(input_channels),
    cache_file_(cache_file)
{
    input_size_ = batch_size_ * input_channels_ * input_width_ * input_height_;
    host_input_.resize(input_size_);

    cudaMalloc(&device_input_, input_size_ * sizeof(float));
    loadImageList(calibration_data_path);
}

Int8MinMaxCalibrator::~Int8MinMaxCalibrator() {
    if (device_input_) {
        cudaFree(device_input_);
    }
}

bool Int8MinMaxCalibrator::loadImageList(const std::string& path) {
    namespace fs = std::filesystem;

    if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_files_.push_back(entry.path().string());
                }
            }
        }
    } else if (fs::is_regular_file(path)) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && fs::exists(line)) {
                image_files_.push_back(line);
            }
        }
    }

    std::random_shuffle(image_files_.begin(), image_files_.end());
    return !image_files_.empty();
}

bool Int8MinMaxCalibrator::loadImage(const std::string& filepath, float* buffer) {
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 3);

    if (!data) {
        return false;
    }

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};
    int plane_size = input_width_ * input_height_;

    for (int y = 0; y < input_height_; y++) {
        for (int x = 0; x < input_width_; x++) {
            float src_x = (x + 0.5f) * width / input_width_ - 0.5f;
            float src_y = (y + 0.5f) * height / input_height_ - 0.5f;

            src_x = std::max(0.0f, std::min(src_x, static_cast<float>(width - 1)));
            src_y = std::max(0.0f, std::min(src_y, static_cast<float>(height - 1)));

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < 3; c++) {
                float v00 = data[(y0 * width + x0) * 3 + c];
                float v01 = data[(y0 * width + x1) * 3 + c];
                float v10 = data[(y1 * width + x0) * 3 + c];
                float v11 = data[(y1 * width + x1) * 3 + c];

                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float val = v0 * (1 - dy) + v1 * dy;

                buffer[c * plane_size + y * input_width_ + x] =
                    (val / 255.0f - mean[c]) / std_dev[c];
            }
        }
    }

    stbi_image_free(data);
    return true;
}

bool Int8MinMaxCalibrator::getBatch(void* bindings[], const char* names[],
                                     int nbBindings) noexcept {
    if (current_batch_ >= static_cast<int>(image_files_.size()) / batch_size_) {
        return false;
    }

    int start_idx = current_batch_ * batch_size_;
    int end_idx = std::min(start_idx + batch_size_,
                           static_cast<int>(image_files_.size()));

    int image_size = input_channels_ * input_width_ * input_height_;

    for (int i = start_idx; i < end_idx; i++) {
        float* buffer = host_input_.data() + (i - start_idx) * image_size;
        if (!loadImage(image_files_[i], buffer)) {
            std::fill(buffer, buffer + image_size, 0.0f);
        }
    }

    cudaMemcpy(device_input_, host_input_.data(),
               input_size_ * sizeof(float), cudaMemcpyHostToDevice);

    bindings[0] = device_input_;
    current_batch_++;

    return true;
}

const void* Int8MinMaxCalibrator::readCalibrationCache(size_t& length) noexcept {
    std::ifstream file(cache_file_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        length = 0;
        return nullptr;
    }

    length = file.tellg();
    file.seekg(0, std::ios::beg);

    calibration_cache_.resize(length);
    file.read(calibration_cache_.data(), length);

    return calibration_cache_.data();
}

void Int8MinMaxCalibrator::writeCalibrationCache(const void* cache,
                                                   size_t length) noexcept {
    std::ofstream file(cache_file_, std::ios::binary);
    if (file.is_open()) {
        file.write(static_cast<const char*>(cache), length);
    }
}

}  // namespace efficientnet
