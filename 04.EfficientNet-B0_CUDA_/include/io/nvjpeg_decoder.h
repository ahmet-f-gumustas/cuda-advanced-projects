#pragma once

#include "image_decoder.h"
#include <nvjpeg.h>
#include <cuda_runtime.h>

class NvJpegDecoder : public ImageDecoder {
public:
    NvJpegDecoder();
    ~NvJpegDecoder();
    
    std::unique_ptr<Image> decode(const std::string& path) override;
    
    bool decode_to_device(const std::string& path, void* d_output, 
                         int& width, int& height, cudaStream_t stream = 0) override;
    
    std::vector<std::unique_ptr<Image>> decode_batch(
        const std::vector<std::string>& paths) override;
    
private:
    nvjpegHandle_t handle_;
    nvjpegJpegState_t state_;
    
    // Batch decode için ek state'ler
    std::vector<nvjpegJpegState_t> batch_states_;
    
    void ensure_batch_states(size_t batch_size);
};