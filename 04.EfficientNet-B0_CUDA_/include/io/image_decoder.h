#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

// Görüntü verisi için basit yapı
struct Image {
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;
    
    size_t size() const { return width * height * channels; }
};

// Görüntü decoder arayüzü
class ImageDecoder {
public:
    virtual ~ImageDecoder() = default;
    
    // CPU bellekte decode et
    virtual std::unique_ptr<Image> decode(const std::string& path) = 0;
    
    // GPU bellekte decode et (destekleniyorsa)
    virtual bool decode_to_device(const std::string& path, void* d_output, 
                                  int& width, int& height, cudaStream_t stream = 0) {
        return false;  // Varsayılan: desteklenmiyor
    }
    
    // Batch decode (opsiyonel)
    virtual std::vector<std::unique_ptr<Image>> decode_batch(
        const std::vector<std::string>& paths) {
        std::vector<std::unique_ptr<Image>> images;
        for (const auto& path : paths) {
            images.push_back(decode(path));
        }
        return images;
    }
};

// Factory metodu
std::unique_ptr<ImageDecoder> create_image_decoder(bool prefer_gpu = true);