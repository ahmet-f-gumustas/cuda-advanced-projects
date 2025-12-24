#ifndef CUDA_PREPROCESS_H
#define CUDA_PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>

namespace efficientnet {

// Image preprocessing parameters
struct PreprocessParams {
    int src_width;
    int src_height;
    int dst_width = 224;   // EfficientNet-B0 input size
    int dst_height = 224;
    float mean[3] = {0.485f, 0.456f, 0.406f};  // ImageNet mean
    float std[3] = {0.229f, 0.224f, 0.225f};   // ImageNet std
};

// Resize and normalize image (HWC uint8 -> NCHW float)
// Input: HWC format, uint8 [0-255]
// Output: NCHW format, float normalized
cudaError_t preprocessImage(
    const uint8_t* src,         // Device pointer to source image
    float* dst,                 // Device pointer to destination tensor
    const PreprocessParams& params,
    cudaStream_t stream = nullptr
);

// Resize image using bilinear interpolation
cudaError_t resizeBilinear(
    const uint8_t* src,
    uint8_t* dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    cudaStream_t stream = nullptr
);

// Normalize image (uint8 -> float with mean/std normalization)
cudaError_t normalizeImage(
    const uint8_t* src,
    float* dst,
    int width, int height,
    const float* mean,  // Device pointer to mean values
    const float* std,   // Device pointer to std values
    cudaStream_t stream = nullptr
);

// Convert HWC to NCHW format
cudaError_t hwcToNchw(
    const float* src,   // HWC format
    float* dst,         // NCHW format
    int height, int width, int channels,
    cudaStream_t stream = nullptr
);

// Combined preprocessing: resize + normalize + HWC->NCHW
// This is the most efficient version
cudaError_t preprocessImageFused(
    const uint8_t* src,         // Device pointer to source image (HWC, uint8)
    float* dst,                 // Device pointer to destination tensor (NCHW, float)
    int src_width, int src_height,
    int dst_width, int dst_height,
    const float* mean,          // Host pointer to mean [3]
    const float* std,           // Host pointer to std [3]
    cudaStream_t stream = nullptr
);

// Batch preprocessing
cudaError_t preprocessBatch(
    const uint8_t* const* src_batch,  // Array of device pointers
    float* dst,                        // Batched output tensor
    int batch_size,
    const PreprocessParams& params,
    cudaStream_t stream = nullptr
);

// Softmax on GPU
cudaError_t softmax(
    const float* input,
    float* output,
    int batch_size,
    int num_classes,
    cudaStream_t stream = nullptr
);

// ArgMax on GPU
cudaError_t argmax(
    const float* input,
    int* output,
    float* max_values,
    int batch_size,
    int num_classes,
    cudaStream_t stream = nullptr
);

// TopK on GPU
cudaError_t topK(
    const float* input,
    int* indices,
    float* values,
    int batch_size,
    int num_classes,
    int k,
    cudaStream_t stream = nullptr
);

}  // namespace efficientnet

#endif  // CUDA_PREPROCESS_H
