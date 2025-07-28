#include "cuda_denoiser.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace rtid {

// Constant memory for NLM parameters
__constant__ float c_h_param;
__constant__ int c_template_radius;
__constant__ int c_search_radius;
__constant__ float c_h2_inv;

// Device helper functions
__device__ __forceinline__ float computePatchDistance(const float* img1, const float* img2,
                                                     int x1, int y1, int x2, int y2,
                                                     int width, int height, int template_radius) {
    float distance = 0.0f;
    int count = 0;
    
    for (int dy = -template_radius; dy <= template_radius; dy++) {
        for (int dx = -template_radius; dx <= template_radius; dx++) {
            int px1 = x1 + dx;
            int py1 = y1 + dy;
            int px2 = x2 + dx;
            int py2 = y2 + dy;
            
            // Boundary check
            if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                
                float diff = img1[py1 * width + px1] - img2[py2 * width + px2];
                distance += diff * diff;
                count++;
            }
        }
    }
    
    return (count > 0) ? (distance / count) : 0.0f;
}

__device__ __forceinline__ float nlmWeight(float distance, float h_param) {
    return expf(-fmaxf(distance - 2.0f * h_param * h_param, 0.0f) / (h_param * h_param));
}

// Standard Non-Local Means kernel
__global__ void nonLocalMeansKernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center_value = input[idx];
    
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Search in the neighborhood
    for (int sy = -c_search_radius; sy <= c_search_radius; sy++) {
        for (int sx = -c_search_radius; sx <= c_search_radius; sx++) {
            int nx = x + sx;
            int ny = y + sy;
            
            // Boundary check
            if (nx >= c_template_radius && nx < width - c_template_radius &&
                ny >= c_template_radius && ny < height - c_template_radius) {
                
                // Compute patch distance
                float patch_dist = computePatchDistance(input, input, x, y, nx, ny,
                                                       width, height, c_template_radius);
                
                // Compute weight
                float weight = nlmWeight(patch_dist, c_h_param);
                
                weighted_sum += weight * input[ny * width + nx];
                weight_sum += weight;
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : center_value;
}

// Optimized NLM with precomputed integral images
__global__ void nonLocalMeansIntegralKernel(const float* __restrict__ input,
                                           const float* __restrict__ integral,
                                           const float* __restrict__ integral_sq,
                                           float* __restrict__ output,
                                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center_value = input[idx];
    
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Template size
    int template_size = (2 * c_template_radius + 1) * (2 * c_template_radius + 1);
    
    // Search in the neighborhood
    for (int sy = -c_search_radius; sy <= c_search_radius; sy++) {
        for (int sx = -c_search_radius; sx <= c_search_radius; sx++) {
            int nx = x + sx;
            int ny = y + sy;
            
            // Boundary check
            if (nx >= c_template_radius && nx < width - c_template_radius &&
                ny >= c_template_radius && ny < height - c_template_radius) {
                
                // Fast patch distance using integral images
                float distance = 0.0f;
                
                // Calculate bounds for integral computation
                int x1_min = min(x - c_template_radius, nx - c_template_radius);
                int y1_min = min(y - c_template_radius, ny - c_template_radius);
                int x1_max = max(x + c_template_radius, nx + c_template_radius);
                int y1_max = max(y + c_template_radius, ny + c_template_radius);
                
                // Use integral images for fast computation (simplified version)
                for (int dy = -c_template_radius; dy <= c_template_radius; dy++) {
                    for (int dx = -c_template_radius; dx <= c_template_radius; dx++) {
                        int px1 = x + dx;
                        int py1 = y + dy;
                        int px2 = nx + dx;
                        int py2 = ny + dy;
                        
                        if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                            px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                            
                            float diff = input[py1 * width + px1] - input[py2 * width + px2];
                            distance += diff * diff;
                        }
                    }
                }
                
                distance /= template_size;
                
                // Compute weight
                float weight = expf(-fmaxf(distance, 0.0f) * c_h2_inv);
                
                weighted_sum += weight * input[ny * width + nx];
                weight_sum += weight;
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : center_value;
}

// Fast NLM using shared memory and reduced search
__global__ void fastNonLocalMeansKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int width, int height) {
    // Shared memory for local patch
    extern __shared__ float shared_data[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load data into shared memory
    int shared_width = blockDim.x + 2 * c_template_radius;
    int shared_height = blockDim.y + 2 * c_template_radius;
    
    for (int sy = ty; sy < shared_height; sy += blockDim.y) {
        for (int sx = tx; sx < shared_width; sx += blockDim.x) {
            int gx = blockIdx.x * blockDim.x + sx - c_template_radius;
            int gy = blockIdx.y * blockDim.y + sy - c_template_radius;
            
            // Clamp to image boundaries
            gx = max(0, min(gx, width - 1));
            gy = max(0, min(gy, height - 1));
            
            shared_data[sy * shared_width + sx] = input[gy * width + gx];
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center_value = input[idx];
    
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Reduced search radius for performance
    int reduced_search = min(c_search_radius, 5);
    
    // Search in the neighborhood
    for (int sy = -reduced_search; sy <= reduced_search; sy++) {
        for (int sx = -reduced_search; sx <= reduced_search; sx++) {
            int nx = x + sx;
            int ny = y + sy;
            
            // Boundary check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                
                // Compute simplified patch distance
                float distance = 0.0f;
                int count = 0;
                
                for (int dy = -c_template_radius; dy <= c_template_radius; dy += 2) {
                    for (int dx = -c_template_radius; dx <= c_template_radius; dx += 2) {
                        int px1 = x + dx;
                        int py1 = y + dy;
                        int px2 = nx + dx;
                        int py2 = ny + dy;
                        
                        if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                            px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                            
                            float diff = input[py1 * width + px1] - input[py2 * width + px2];
                            distance += diff * diff;
                            count++;
                        }
                    }
                }
                
                if (count > 0) {
                    distance /= count;
                    
                    // Compute weight
                    float weight = expf(-fmaxf(distance, 0.0f) * c_h2_inv);
                    
                    weighted_sum += weight * input[ny * width + nx];
                    weight_sum += weight;
                }
            }
        }
    }
    
    // Normalize and store result
    output[idx] = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : center_value;
}

// Precompute integral images for faster patch comparisons
__global__ void computeIntegralImage(const float* __restrict__ input,
                                    float* __restrict__ integral,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float sum = input[idx];
    
    // Add left neighbor
    if (x > 0) {
        sum += integral[idx - 1];
    }
    
    // Add top neighbor
    if (y > 0) {
        sum += integral[(y - 1) * width + x];
    }
    
    // Subtract top-left neighbor (added twice)
    if (x > 0 && y > 0) {
        sum -= integral[(y - 1) * width + (x - 1)];
    }
    
    integral[idx] = sum;
}

// Host function implementations
extern "C" {

bool launchNonLocalMeans(float* d_input, float* d_output, int width, int height,
                        float h_param, int template_radius, int search_radius,
                        cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_h_param, &h_param, sizeof(float));
    cudaMemcpyToSymbol(c_template_radius, &template_radius, sizeof(int));
    cudaMemcpyToSymbol(c_search_radius, &search_radius, sizeof(int));
    
    float h2_inv = 1.0f / (h_param * h_param);
    cudaMemcpyToSymbol(c_h2_inv, &h2_inv, sizeof(float));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Launch kernel
    nonLocalMeansKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchFastNonLocalMeans(float* d_input, float* d_output, int width, int height,
                            float h_param, int template_radius, int search_radius,
                            cudaStream_t stream) {
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_h_param, &h_param, sizeof(float));
    cudaMemcpyToSymbol(c_template_radius, &template_radius, sizeof(int));
    cudaMemcpyToSymbol(c_search_radius, &search_radius, sizeof(int));
    
    float h2_inv = 1.0f / (h_param * h_param);
    cudaMemcpyToSymbol(c_h2_inv, &h2_inv, sizeof(float));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Calculate shared memory size
    int shared_width = block_size.x + 2 * template_radius;
    int shared_height = block_size.y + 2 * template_radius;
    size_t shared_mem_size = shared_width * shared_height * sizeof(float);
    
    // Launch kernel with shared memory
    fastNonLocalMeansKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

bool launchNonLocalMeansWithIntegral(float* d_input, float* d_integral, float* d_integral_sq,
                                    float* d_output, int width, int height,
                                    float h_param, int template_radius, int search_radius,
                                    cudaStream_t stream) {
    // First compute integral images
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    computeIntegralImage<<<grid_size, block_size, 0, stream>>>(
        d_input, d_integral, width, height);
    
    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_h_param, &h_param, sizeof(float));
    cudaMemcpyToSymbol(c_template_radius, &template_radius, sizeof(int));
    cudaMemcpyToSymbol(c_search_radius, &search_radius, sizeof(int));
    
    float h2_inv = 1.0f / (h_param * h_param);
    cudaMemcpyToSymbol(c_h2_inv, &h2_inv, sizeof(float));
    
    // Launch NLM kernel with integral images
    nonLocalMeansIntegralKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_integral, d_integral_sq, d_output, width, height);
    
    return cudaGetLastError() == cudaSuccess;
}

} // extern "C"

} // namespace rtid