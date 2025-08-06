#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Bilinear interpolation kernel
__device__ float bilinear_interpolate(const uint8_t* input, int height, int width,
                                     float y, float x, int c) {
    // Sınırları kontrol et
    if (y < 0) y = 0;
    if (x < 0) x = 0;
    
    int y0 = static_cast<int>(y);
    int x0 = static_cast<int>(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    
    if (y1 >= height) y1 = height - 1;
    if (x1 >= width) x1 = width - 1;
    
    float fy = y - y0;
    float fx = x - x0;
    
    // 4 komşu pikseli al
    float p00 = input[(y0 * width + x0) * 3 + c];
    float p01 = input[(y0 * width + x1) * 3 + c];
    float p10 = input[(y1 * width + x0) * 3 + c];
    float p11 = input[(y1 * width + x1) * 3 + c];
    
    // Bilinear interpolation
    float p0 = p00 * (1 - fx) + p01 * fx;
    float p1 = p10 * (1 - fx) + p11 * fx;
    
    return p0 * (1 - fy) + p1 * fy;
}

// Resize + normalize + HWC->NCHW kernel (FP32)
__global__ void resize_normalize_hwc2nchw_kernel(
    const uint8_t* input, int input_height, int input_width,
    float* output, int output_height, int output_width,
    const float* mean, const float* std) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    // Ölçek faktörlerini hesapla
    float scale_y = static_cast<float>(input_height) / output_height;
    float scale_x = static_cast<float>(input_width) / output_width;
    
    // Kaynak koordinatları
    float src_y = y * scale_y;
    float src_x = x * scale_x;
    
    // Her kanal için
    for (int c = 0; c < 3; ++c) {
        // Bilinear interpolation
        float value = bilinear_interpolate(input, input_height, input_width,
                                         src_y, src_x, c);
        
        // Normalize: [0,255] -> [0,1] -> standardize
        value = (value / 255.0f - mean[c]) / std[c];
        
        // NCHW formatında yaz
        int out_idx = c * output_height * output_width + y * output_width + x;
        output[out_idx] = value;
    }
}

// FP16 versiyonu
__global__ void resize_normalize_hwc2nchw_kernel_fp16(
    const uint8_t* input, int input_height, int input_width,
    __half* output, int output_height, int output_width,
    const float* mean, const float* std) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    float scale_y = static_cast<float>(input_height) / output_height;
    float scale_x = static_cast<float>(input_width) / output_width;
    
    float src_y = y * scale_y;
    float src_x = x * scale_x;
    
    for (int c = 0; c < 3; ++c) {
        float value = bilinear_interpolate(input, input_height, input_width,
                                         src_y, src_x, c);
        
        value = (value / 255.0f - mean[c]) / std[c];
        
        int out_idx = c * output_height * output_width + y * output_width + x;
        output[out_idx] = __float2half(value);
    }
}

// Host fonksiyonları
void resize_bilinear_normalize_hwc2nchw(
    const uint8_t* d_input, int input_height, int input_width,
    float* d_output, int output_height, int output_width,
    const float* mean, const float* std,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid((output_width + block.x - 1) / block.x,
              (output_height + block.y - 1) / block.y);
    
    resize_normalize_hwc2nchw_kernel<<<grid, block, 0, stream>>>(
        d_input, input_height, input_width,
        d_output, output_height, output_width,
        mean, std);
}

void resize_bilinear_normalize_hwc2nchw_fp16(
    const uint8_t* d_input, int input_height, int input_width,
    __half* d_output, int output_height, int output_width,
    const float* mean, const float* std,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid((output_width + block.x - 1) / block.x,
              (output_height + block.y - 1) / block.y);
    
    resize_normalize_hwc2nchw_kernel_fp16<<<grid, block, 0, stream>>>(
        d_input, input_height, input_width,
        d_output, output_height, output_width,
        mean, std);
}