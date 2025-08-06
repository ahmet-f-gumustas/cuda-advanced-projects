#include "io/nvjpeg_decoder.h"
#include "common/cuda_utils.h"
#include "common/logger.h"
#include <fstream>
#include <vector>

#define NVJPEG_CHECK(call)                                                    \
    do {                                                                      \
        nvjpegStatus_t status = call;                                         \
        if (status != NVJPEG_STATUS_SUCCESS) {                              \
            LOG_ERROR("NVJPEG error:", status);                             \
            throw std::runtime_error("NVJPEG error: " + std::to_string(status)); \
        }                                                                     \
    } while (0)

NvJpegDecoder::NvJpegDecoder() {
    NVJPEG_CHECK(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &handle_));
    NVJPEG_CHECK(nvjpegJpegStateCreate(handle_, &state_));
}

NvJpegDecoder::~NvJpegDecoder() {
    for (auto& state : batch_states_) {
        nvjpegJpegStateDestroy(state);
    }
    nvjpegJpegStateDestroy(state_);
    nvjpegDestroy(handle_);
}

std::unique_ptr<Image> NvJpegDecoder::decode(const std::string& path) {
    // JPEG dosyasını oku
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open file:", path);
        return nullptr;
    }
    
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> jpeg_data(file_size);
    file.read(reinterpret_cast<char*>(jpeg_data.data()), file_size);
    
    // JPEG bilgilerini al
    int width, height, channels;
    nvjpegChromaSubsampling_t subsampling;
    NVJPEG_CHECK(nvjpegGetImageInfo(handle_, jpeg_data.data(), file_size,
                                    &channels, &subsampling, &width, &height));
    
    // Çıkış görüntüsü için bellek ayır
    auto image = std::make_unique<Image>();
    image->width = width;
    image->height = height;
    image->channels = 3;  // RGB'ye zorla
    image->data.resize(width * height * 3);
    
    // NVJPEG çıkış formatını ayarla
    nvjpegImage_t nv_image;
    nv_image.channel[0] = image->data.data();
    nv_image.pitch[0] = width * 3;
    nv_image.channel[1] = nullptr;
    nv_image.channel[2] = nullptr;
    
    // Decode (host memory'ye)
    NVJPEG_CHECK(nvjpegDecode(handle_, state_, jpeg_data.data(), file_size,
                             NVJPEG_OUTPUT_RGBI, &nv_image, nullptr));
    
    return image;
}

bool NvJpegDecoder::decode_to_device(const std::string& path, void* d_output,
                                    int& width, int& height, cudaStream_t stream) {
    // JPEG dosyasını oku
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open file:", path);
        return false;
    }
    
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> jpeg_data(file_size);
    file.read(reinterpret_cast<char*>(jpeg_data.data()), file_size);
    
    // JPEG bilgilerini al
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    NVJPEG_CHECK(nvjpegGetImageInfo(handle_, jpeg_data.data(), file_size,
                                    &channels, &subsampling, &width, &height));
    
    // NVJPEG çıkış formatını ayarla (device memory)
    nvjpegImage_t nv_image;
    nv_image.channel[0] = static_cast<unsigned char*>(d_output);
    nv_image.pitch[0] = width * 3;
    nv_image.channel[1] = nullptr;
    nv_image.channel[2] = nullptr;
    
    // Decode (device memory'ye)
    NVJPEG_CHECK(nvjpegDecode(handle_, state_, jpeg_data.data(), file_size,
                             NVJPEG_OUTPUT_RGBI, &nv_image, stream));
    
    return true;
}

std::vector<std::unique_ptr<Image>> NvJpegDecoder::decode_batch(
    const std::vector<std::string>& paths) {
    // Batch decoding implementasyonu
    ensure_batch_states(paths.size());
    
    std::vector<std::unique_ptr<Image>> images;
    images.reserve(paths.size());
    
    // Her görüntüyü ayrı decode et (basit versiyon)
    // Gerçek batch decoding daha karmaşık
    for (const auto& path : paths) {
        images.push_back(decode(path));
    }
    
    return images;
}

void NvJpegDecoder::ensure_batch_states(size_t batch_size) {
    while (batch_states_.size() < batch_size) {
        nvjpegJpegState_t state;
        NVJPEG_CHECK(nvjpegJpegStateCreate(handle_, &state));
        batch_states_.push_back(state);
    }
}