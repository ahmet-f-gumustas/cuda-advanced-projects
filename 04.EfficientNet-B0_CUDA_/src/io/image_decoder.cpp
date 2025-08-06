#include "io/image_decoder.h"
#include "io/nvjpeg_decoder.h"
#include "common/logger.h"

#ifdef HAVE_NVJPEG
#include <cuda_runtime.h>
#endif

// STB_IMAGE implementasyonu
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// CPU tabanlı decoder (fallback)
class StbImageDecoder : public ImageDecoder {
public:
    std::unique_ptr<Image> decode(const std::string& path) override {
        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
        
        if (!data) {
            LOG_ERROR("Failed to load image:", path, "Error:", stbi_failure_reason());
            return nullptr;
        }
        
        auto image = std::make_unique<Image>();
        image->width = width;
        image->height = height;
        image->channels = 3;  // RGB'ye zorla
        image->data.assign(data, data + width * height * 3);
        
        stbi_image_free(data);
        return image;
    }
};

std::unique_ptr<ImageDecoder> create_image_decoder(bool prefer_gpu) {
#ifdef HAVE_NVJPEG
    if (prefer_gpu) {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            try {
                return std::make_unique<NvJpegDecoder>();
            } catch (const std::exception& e) {
                LOG_WARNING("Failed to create NVJPEG decoder:", e.what(), "- falling back to CPU");
            }
        }
    }
#endif
    return std::make_unique<StbImageDecoder>();
}