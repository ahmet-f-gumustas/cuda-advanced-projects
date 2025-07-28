#include "cuda_denoiser.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace rtid {

// Constant memory for filter parameters
__constant__ float c_sigma_color;
__constant__ float c_sigma_space;
__constant__ int c_kernel_radius;

// Device helper functions
__device__ __forceinline__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

__device__ __forceinline__ float colorDistance(float a, float b) {
    float diff = a - b;
    return diff * diff;
}

__device__ __forceinline__ float spatialDistance(int x1, int y1, int x2, int y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

// Standard bilateral filter kernel
__global__ void bilateralFilterKernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center_value = input[idx];
    
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply bilateral filter
    for (int dy = -c_kernel_radius; dy <= c_kernel_radius; dy++) {
        for (int dx = -c_kernel_radius; dx <= c_kernel_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Boundary check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_idx = ny * width + nx;
                float neighbor_value = input[neighbor_idx];
                
                // Calculate spatial weight
                float spatial_dist = spatialDistance(x, y, nx, ny);
                float spatial_weight = gaussian(spatial_dist, c_sigma_space);
                
                // Calculate color weight
                float color_dist = sqrtf(colorDistance(center_value, neighbor_value));
                float color_weight = gaussian(color_dist, c_sigma_color);
                
                // Combined weight
                float weight = spatial_weight * color_weight;
                
                weighted_sum += weight * neighbor_value;
                weight_sum += weight;
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : center_value;
}

// Optimized bilateral filter using shared memory
__global__ void bilateralFilterSharedKernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height) {
    // Shared memory for tile + halo
    extern __shared__ float shared_data[];
    
    int block_size = blockDim.x;
    int shared_size = block_size + 2 * c_kernel_radius;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load data into shared memory with halo
    for (int dy = 0; dy < shared_size; dy += blockDim.y) {
        for (int dx = 0; dx < shared_size; dx += blockDim.x) {
            int sx = tx + dx;
            int sy = ty + dy;
            
            if (sx < shared_size && sy < shared_size) {
                int gx = blockIdx.x * blockDim.x + sx - c_kernel_radius;
                int gy = blockIdx.y * blockDim.y + sy - c_kernel_radius;
                
                // Handle boundaries with clamping
                gx = max(0, min(gx, width - 1));
                gy = max(0, min(gy, height - 1));
                
                shared_data[sy * shared_size + sx] = input[gy * width + gx];
            }
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    int shared_x = tx + c_kernel_radius;
    int shared_y = ty + c_kernel_radius;
    
    float center_value = shared_data[shared_y * shared_size + shared_x];
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply bilateral filter using shared memory
    for (int dy = -c_kernel_radius; dy <= c_kernel_radius; dy++) {
        for (int dx = -c_kernel_radius; dx <= c_kernel_radius; dx++) {
            int sx = shared_x + dx;
            int sy = shared_y + dy;
            
            float neighbor_value = shared_data[sy * shared_size + sx];
            
            // Calculate spatial weight
            float spatial_dist = spatialDistance(0, 0, dx, dy);
            float spatial_weight = gaussian(spatial_dist, c_sigma_space);
            
            // Calculate color weight
            float color_dist = sqrtf(colorDistance(center_value, neighbor_value));
            float color_weight = gaussian(color_dist, c_sigma_color);
            
            // Combined weight
            float weight = spatial_weight * color_weight;
            
            weighted_sum += weight * neighbor_value;
            weight_sum += weight;
        }
    }
    
    // Normalize and store result
    int idx = y * width + x;
    output[idx] = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : center_value;
}

// Adaptive bilateral filter kernel
__global__ void adaptiveBilateralKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center_value = input[idx];
    
    // Calculate local variance for adaptive parameters
    float local_mean = 0.0f;
    float local_variance = 0.0f;
    int count = 0;
    
    // Calculate local statistics
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float value = input[ny * width + nx];
                local_mean += value;
                count++;
            }
        }
    }
    
    if (count > 0) {
        local_mean /= count;
        
        // Calculate variance
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float value = input[ny * width + nx];
                    float diff = value - local_mean;
                    local_variance += diff * diff;
                }
            }
        }
        local_variance /= count;
    }
    
    // Adaptive sigma based on local variance
    float adaptive_sigma_color = c_sigma_color * (1.0f + sqrtf(local_variance));
    float adaptive_sigma_space = c_sigma_space;
    
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply adaptive bilateral filter
    for (int dy = -c_kernel_radius; dy <= c_kernel_radius; dy++) {
        for (int dx = -c_kernel_radius; dx <= c_kernel_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_idx = ny * width + nx;
                float neighbor_value = input[neighbor_idx];
                
                // Calculate spatial weight
                float spatial_dist = spatialDistance(x, y, nx, ny);
                float spatial_weight = gaussian(spatial_dist, adaptive_sigma_space);
                
                // Calculate color weight
                float color_dist = sqrtf(colorDistance(center_value, neighbor_value));
                float color_weight = gaussian(color_dist, adaptive_sigma_color);
                
                // Combined weight
                float weight = spatial_weight * color_weight;
                
                weighted_sum += weight * neighbor_value;
                weight_sum += weight;
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : center_value;
}

// Host function implementations
extern "C" {

bool launchBilateralFilter(float* d_input, float* d_output, int width, int height,
                          float sigma_color, float sigma_space, int kernel_radius,
                          cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_sigma_color, &sigma_color, sizeof(float));
    cudaMemcpyToSymbol(c_sigma_space, &sigma_space, sizeof(float));
    cudaMemcpyToSymbol(c_kernel_radius, &kernel_radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Launch kernel
    bilateralFilterKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchBilateralFilterShared(float* d_input, float* d_output, int width, int height,
                                float sigma_color, float sigma_space, int kernel_radius,
                                cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_sigma_color, &sigma_color, sizeof(float));
    cudaMemcpyToSymbol(c_sigma_space, &sigma_space, sizeof(float));
    cudaMemcpyToSymbol(c_kernel_radius, &kernel_radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Calculate shared memory size
    int shared_size = (block_size.x + 2 * kernel_radius) * (block_size.y + 2 * kernel_radius);
    size_t shared_mem_size = shared_size * sizeof(float);
    
    // Launch kernel with shared memory
    bilateralFilterSharedKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchAdaptiveBilateralFilter(float* d_input, float* d_output, int width, int height,
                                  float sigma_color, float sigma_space, int kernel_radius,
                                  cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_sigma_color, &sigma_color, sizeof(float));
    cudaMemcpyToSymbol(c_sigma_space, &sigma_space, sizeof(float));
    cudaMemcpyToSymbol(c_kernel_radius, &kernel_radius, sizeof(int));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Launch kernel
    adaptiveBilateralKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

} // extern "C"

} // namespace rtid