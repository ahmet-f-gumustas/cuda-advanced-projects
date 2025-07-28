#include "cuda_denoiser.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace rtid {

// Constant memory for Gaussian parameters
__constant__ float c_gaussian_sigma;
__constant__ int c_gaussian_radius;
__constant__ float c_gaussian_kernel[64]; // Max kernel size 15x15

// Device helper functions
__device__ __forceinline__ float gaussian2D(int x, int y, float sigma) {
    float sigma2 = sigma * sigma;
    return expf(-(x * x + y * y) / (2.0f * sigma2)) / (2.0f * M_PI * sigma2);
}

// Separable Gaussian filter - horizontal pass
__global__ void gaussianHorizontalKernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int width, int height) {
    extern __shared__ float shared_row[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    
    if (y >= height) return;
    
    // Load data into shared memory with halo
    int shared_width = blockDim.x + 2 * c_gaussian_radius;
    
    // Load left halo
    if (tx < c_gaussian_radius) {
        int src_x = blockIdx.x * blockDim.x + tx - c_gaussian_radius;
        src_x = max(0, src_x); // Clamp to boundary
        shared_row[tx] = input[y * width + src_x];
    }
    
    // Load main data
    if (x < width) {
        shared_row[tx + c_gaussian_radius] = input[y * width + x];
    } else {
        shared_row[tx + c_gaussian_radius] = 0.0f;
    }
    
    // Load right halo
    if (tx < c_gaussian_radius) {
        int src_x = blockIdx.x * blockDim.x + blockDim.x + tx;
        src_x = min(width - 1, src_x); // Clamp to boundary
        shared_row[tx + blockDim.x + c_gaussian_radius] = input[y * width + src_x];
    }
    
    __syncthreads();
    
    if (x >= width) return;
    
    // Apply horizontal Gaussian filter
    float sum = 0.0f;
    for (int i = -c_gaussian_radius; i <= c_gaussian_radius; i++) {
        int kernel_idx = i + c_gaussian_radius;
        int shared_idx = tx + c_gaussian_radius + i;
        sum += c_gaussian_kernel[kernel_idx] * shared_row[shared_idx];
    }
    
    output[y * width + x] = sum;
}

// Separable Gaussian filter - vertical pass
__global__ void gaussianVerticalKernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int width, int height) {
    extern __shared__ float shared_col[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = threadIdx.y;
    
    if (x >= width) return;
    
    // Load data into shared memory with halo
    int shared_height = blockDim.y + 2 * c_gaussian_radius;
    
    // Load top halo
    if (ty < c_gaussian_radius) {
        int src_y = blockIdx.y * blockDim.y + ty - c_gaussian_radius;
        src_y = max(0, src_y); // Clamp to boundary
        shared_col[ty] = input[src_y * width + x];
    }
    
    // Load main data
    if (y < height) {
        shared_col[ty + c_gaussian_radius] = input[y * width + x];
    } else {
        shared_col[ty + c_gaussian_radius] = 0.0f;
    }
    
    // Load bottom halo
    if (ty < c_gaussian_radius) {
        int src_y = blockIdx.y * blockDim.y + blockDim.y + ty;
        src_y = min(height - 1, src_y); // Clamp to boundary
        shared_col[ty + blockDim.y + c_gaussian_radius] = input[src_y * width + x];
    }
    
    __syncthreads();
    
    if (y >= height) return;
    
    // Apply vertical Gaussian filter
    float sum = 0.0f;
    for (int i = -c_gaussian_radius; i <= c_gaussian_radius; i++) {
        int kernel_idx = i + c_gaussian_radius;
        int shared_idx = ty + c_gaussian_radius + i;
        sum += c_gaussian_kernel[kernel_idx] * shared_col[shared_idx];
    }
    
    output[y * width + x] = sum;
}

// Direct 2D Gaussian filter kernel
__global__ void gaussian2DKernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply 2D Gaussian filter
    for (int dy = -c_gaussian_radius; dy <= c_gaussian_radius; dy++) {
        for (int dx = -c_gaussian_radius; dx <= c_gaussian_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Boundary check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float weight = gaussian2D(dx, dy, c_gaussian_sigma);
                sum += weight * input[ny * width + nx];
                weight_sum += weight;
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (sum / weight_sum) : input[idx];
}

// Optimized 2D Gaussian with shared memory
__global__ void gaussian2DSharedKernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int width, int height) {
    extern __shared__ float shared_data[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int shared_width = blockDim.x + 2 * c_gaussian_radius;
    int shared_height = blockDim.y + 2 * c_gaussian_radius;
    
    // Load data into shared memory with halo
    for (int dy = 0; dy < shared_height; dy += blockDim.y) {
        for (int dx = 0; dx < shared_width; dx += blockDim.x) {
            int sx = tx + dx;
            int sy = ty + dy;
            
            if (sx < shared_width && sy < shared_height) {
                int gx = blockIdx.x * blockDim.x + sx - c_gaussian_radius;
                int gy = blockIdx.y * blockDim.y + sy - c_gaussian_radius;
                
                // Handle boundaries with clamping
                gx = max(0, min(gx, width - 1));
                gy = max(0, min(gy, height - 1));
                
                shared_data[sy * shared_width + sx] = input[gy * width + gx];
            }
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    int shared_x = tx + c_gaussian_radius;
    int shared_y = ty + c_gaussian_radius;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply 2D Gaussian filter using shared memory
    for (int dy = -c_gaussian_radius; dy <= c_gaussian_radius; dy++) {
        for (int dx = -c_gaussian_radius; dx <= c_gaussian_radius; dx++) {
            int sx = shared_x + dx;
            int sy = shared_y + dy;
            
            float weight = gaussian2D(dx, dy, c_gaussian_sigma);
            sum += weight * shared_data[sy * shared_width + sx];
            weight_sum += weight;
        }
    }
    
    // Normalize and store result
    int idx = y * width + x;
    output[idx] = (weight_sum > 0.0f) ? (sum / weight_sum) : shared_data[shared_y * shared_width + shared_x];
}

// Adaptive Gaussian filter with edge preservation
__global__ void adaptiveGaussianKernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center_value = input[idx];
    
    // Calculate local gradient magnitude for edge detection
    float gradient_mag = 0.0f;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = input[y * width + (x + 1)] - input[y * width + (x - 1)];
        float gy = input[(y + 1) * width + x] - input[(y - 1) * width + x];
        gradient_mag = sqrtf(gx * gx + gy * gy);
    }
    
    // Adaptive sigma based on gradient magnitude
    float adaptive_sigma = c_gaussian_sigma * (1.0f + gradient_mag);
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply adaptive Gaussian filter
    for (int dy = -c_gaussian_radius; dy <= c_gaussian_radius; dy++) {
        for (int dx = -c_gaussian_radius; dx <= c_gaussian_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Boundary check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float spatial_dist = sqrtf(dx * dx + dy * dy);
                float weight = expf(-(spatial_dist * spatial_dist) / (2.0f * adaptive_sigma * adaptive_sigma));
                
                sum += weight * input[ny * width + nx];
                weight_sum += weight;
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (sum / weight_sum) : center_value;
}

// Recursive Gaussian approximation (single pass)
__global__ void recursiveGaussianKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       float* __restrict__ temp,
                                       int width, int height,
                                       float a0, float a1, float a2, float a3,
                                       float b1, float b2, float coeff) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= height) return;
    
    // Forward pass
    float w1 = 0.0f, w2 = 0.0f;
    for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        float w0 = a0 * input[idx] + a1 * (x > 0 ? input[idx - 1] : input[idx]) +
                   a2 * (x > 1 ? input[idx - 2] : input[idx]) +
                   a3 * (x > 2 ? input[idx - 3] : input[idx]) +
                   b1 * w1 + b2 * w2;
        
        temp[idx] = w0;
        w2 = w1;
        w1 = w0;
    }
    
    // Backward pass
    w1 = w2 = 0.0f;
    for (int x = width - 1; x >= 0; x--) {
        int idx = y * width + x;
        float w0 = a0 * temp[idx] + a1 * (x < width - 1 ? temp[idx + 1] : temp[idx]) +
                   a2 * (x < width - 2 ? temp[idx + 2] : temp[idx]) +
                   a3 * (x < width - 3 ? temp[idx + 3] : temp[idx]) +
                   b1 * w1 + b2 * w2;
        
        output[idx] = coeff * w0;
        w2 = w1;
        w1 = w0;
    }
}

// Host function to generate Gaussian kernel
void generateGaussianKernel(float* kernel, int radius, float sigma) {
    float sum = 0.0f;
    
    for (int i = -radius; i <= radius; i++) {
        float value = expf(-(i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = value;
        sum += value;
    }
    
    // Normalize kernel
    for (int i = 0; i < 2 * radius + 1; i++) {
        kernel[i] /= sum;
    }
}

// Host function implementations
extern "C" {

bool launchGaussianFilter(float* d_input, float* d_output, int width, int height,
                         float sigma, int radius, cudaStream_t stream) {
    // Generate and copy Gaussian kernel to constant memory
    float h_kernel[64];
    generateGaussianKernel(h_kernel, radius, sigma);
    
    cudaMemcpyToSymbol(c_gaussian_kernel, h_kernel, (2 * radius + 1) * sizeof(float));
    cudaMemcpyToSymbol(c_gaussian_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(c_gaussian_radius, &radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Launch 2D Gaussian kernel
    gaussian2DKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchSeparableGaussianFilter(float* d_input, float* d_temp, float* d_output,
                                  int width, int height, float sigma, int radius,
                                  cudaStream_t stream) {
    // Generate and copy Gaussian kernel to constant memory
    float h_kernel[64];
    generateGaussianKernel(h_kernel, radius, sigma);
    
    cudaMemcpyToSymbol(c_gaussian_kernel, h_kernel, (2 * radius + 1) * sizeof(float));
    cudaMemcpyToSymbol(c_gaussian_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(c_gaussian_radius, &radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size_h(256, 1);
    dim3 grid_size_h((width + block_size_h.x - 1) / block_size_h.x, height);
    
    dim3 block_size_v(1, 256);
    dim3 grid_size_v(width, (height + block_size_v.y - 1) / block_size_v.y);
    
    // Calculate shared memory size
    size_t shared_mem_h = (block_size_h.x + 2 * radius) * sizeof(float);
    size_t shared_mem_v = (block_size_v.y + 2 * radius) * sizeof(float);
    
    // Launch horizontal pass
    gaussianHorizontalKernel<<<grid_size_h, block_size_h, shared_mem_h, stream>>>(
        d_input, d_temp, width, height);
    
    // Launch vertical pass
    gaussianVerticalKernel<<<grid_size_v, block_size_v, shared_mem_v, stream>>>(
        d_temp, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchGaussianFilterShared(float* d_input, float* d_output, int width, int height,
                               float sigma, int radius, cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_gaussian_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(c_gaussian_radius, &radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Calculate shared memory size
    int shared_width = block_size.x + 2 * radius;
    int shared_height = block_size.y + 2 * radius;
    size_t shared_mem_size = shared_width * shared_height * sizeof(float);
    
    // Launch kernel with shared memory
    gaussian2DSharedKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchAdaptiveGaussianFilter(float* d_input, float* d_output, int width, int height,
                                 float sigma, int radius, cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_gaussian_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(c_gaussian_radius, &radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Launch adaptive Gaussian kernel
    adaptiveGaussianKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

} // extern "C"

} // namespace rtid