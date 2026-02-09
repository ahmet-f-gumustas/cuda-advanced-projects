#include "gaussian_pyramid.cuh"

namespace cuda_stabilizer {

// Gaussian kernel coefficients for sigma = 1.0 (5x5)
__constant__ float c_gauss_kernel[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

// 1D Gaussian kernel for separable convolution
__constant__ float c_gauss_1d[7] = {
    0.00443f, 0.05400f, 0.24203f, 0.39905f, 0.24203f, 0.05400f, 0.00443f
};

void initGaussianPyramid(GaussianPyramid& pyramid, int width, int height, int num_levels) {
    pyramid.num_levels = num_levels;
    pyramid.d_levels = new float*[num_levels];
    pyramid.widths = new int[num_levels];
    pyramid.heights = new int[num_levels];

    int w = width;
    int h = height;

    for (int i = 0; i < num_levels; i++) {
        pyramid.widths[i] = w;
        pyramid.heights[i] = h;
        CUDA_CHECK(cudaMalloc(&pyramid.d_levels[i], w * h * sizeof(float)));
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

void releaseGaussianPyramid(GaussianPyramid& pyramid) {
    if (pyramid.d_levels != nullptr) {
        for (int i = 0; i < pyramid.num_levels; i++) {
            if (pyramid.d_levels[i] != nullptr) {
                cudaFree(pyramid.d_levels[i]);
            }
        }
        delete[] pyramid.d_levels;
        pyramid.d_levels = nullptr;
    }
    if (pyramid.widths != nullptr) {
        delete[] pyramid.widths;
        pyramid.widths = nullptr;
    }
    if (pyramid.heights != nullptr) {
        delete[] pyramid.heights;
        pyramid.heights = nullptr;
    }
    pyramid.num_levels = 0;
}

// Gaussian blur kernel (2D convolution)
__global__ void gaussianBlurKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int kernel_radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);

            float w = c_gauss_kernel[kx + kernel_radius] * c_gauss_kernel[ky + kernel_radius];
            sum += input[ny * width + nx] * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = sum / weight_sum;
}

void gaussianBlur(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    float sigma
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int kernel_radius = 2;  // For 5x5 kernel

    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, width, height, kernel_radius);
    CUDA_CHECK(cudaGetLastError());
}

// Horizontal Gaussian blur kernel (separable)
__global__ void gaussianBlurHorizontalKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int kernel_radius = 3;

    #pragma unroll
    for (int k = -kernel_radius; k <= kernel_radius; k++) {
        int nx = min(max(x + k, 0), width - 1);
        sum += input[y * width + nx] * c_gauss_1d[k + kernel_radius];
    }

    output[y * width + x] = sum;
}

// Vertical Gaussian blur kernel (separable)
__global__ void gaussianBlurVerticalKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int kernel_radius = 3;

    #pragma unroll
    for (int k = -kernel_radius; k <= kernel_radius; k++) {
        int ny = min(max(y + k, 0), height - 1);
        sum += input[ny * width + x] * c_gauss_1d[k + kernel_radius];
    }

    output[y * width + x] = sum;
}

void gaussianBlurSeparable(
    const float* d_input,
    float* d_output,
    float* d_temp,
    int width,
    int height,
    float sigma
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Horizontal pass
    gaussianBlurHorizontalKernel<<<grid, block>>>(d_input, d_temp, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Vertical pass
    gaussianBlurVerticalKernel<<<grid, block>>>(d_temp, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// Downsample by 2 kernel
__global__ void downsample2xKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_width,
    int in_height,
    int out_width,
    int out_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    // Apply Gaussian filter while downsampling
    int sx = x * 2;
    int sy = y * 2;

    float sum = 0.0f;
    float weight = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(sx + dx, 0), in_width - 1);
            int ny = min(max(sy + dy, 0), in_height - 1);
            float w = (dx == 0 && dy == 0) ? 0.25f : (dx == 0 || dy == 0) ? 0.125f : 0.0625f;
            sum += input[ny * in_width + nx] * w;
            weight += w;
        }
    }

    output[y * out_width + x] = sum / weight;
}

void downsample2x(
    const float* d_input,
    float* d_output,
    int in_width,
    int in_height
) {
    int out_width = (in_width + 1) / 2;
    int out_height = (in_height + 1) / 2;

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);

    downsample2xKernel<<<grid, block>>>(d_input, d_output, in_width, in_height, out_width, out_height);
    CUDA_CHECK(cudaGetLastError());
}

// Upsample by 2 kernel
__global__ void upsample2xKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_width,
    int in_height,
    int out_width,
    int out_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    float fx = x / 2.0f;
    float fy = y / 2.0f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, in_width - 1);
    int y1 = min(y0 + 1, in_height - 1);

    float wx = fx - x0;
    float wy = fy - y0;

    float v00 = input[y0 * in_width + x0];
    float v01 = input[y0 * in_width + x1];
    float v10 = input[y1 * in_width + x0];
    float v11 = input[y1 * in_width + x1];

    float v0 = v00 * (1.0f - wx) + v01 * wx;
    float v1 = v10 * (1.0f - wx) + v11 * wx;

    output[y * out_width + x] = v0 * (1.0f - wy) + v1 * wy;
}

void upsample2x(
    const float* d_input,
    float* d_output,
    int in_width,
    int in_height
) {
    int out_width = in_width * 2;
    int out_height = in_height * 2;

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);

    upsample2xKernel<<<grid, block>>>(d_input, d_output, in_width, in_height, out_width, out_height);
    CUDA_CHECK(cudaGetLastError());
}

void buildGaussianPyramid(
    const float* d_image,
    GaussianPyramid& pyramid,
    int width,
    int height
) {
    // Copy first level
    CUDA_CHECK(cudaMemcpy(pyramid.d_levels[0], d_image, width * height * sizeof(float), cudaMemcpyDeviceToDevice));

    // Build subsequent levels
    for (int i = 1; i < pyramid.num_levels; i++) {
        // Blur and downsample
        float* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, pyramid.widths[i-1] * pyramid.heights[i-1] * sizeof(float)));

        gaussianBlur(pyramid.d_levels[i-1], d_temp, pyramid.widths[i-1], pyramid.heights[i-1], 1.0f);
        downsample2x(d_temp, pyramid.d_levels[i], pyramid.widths[i-1], pyramid.heights[i-1]);

        CUDA_CHECK(cudaFree(d_temp));
    }
}

// Laplacian subtraction kernel
__global__ void laplacianSubtractKernel(
    const float* __restrict__ original,
    const float* __restrict__ upsampled,
    float* __restrict__ laplacian,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    laplacian[idx] = original[idx] - upsampled[idx];
}

void buildLaplacianPyramid(
    const float* d_image,
    GaussianPyramid& gaussian_pyramid,
    GaussianPyramid& laplacian_pyramid,
    int width,
    int height
) {
    // First build Gaussian pyramid
    buildGaussianPyramid(d_image, gaussian_pyramid, width, height);

    // Initialize Laplacian pyramid with same structure
    initGaussianPyramid(laplacian_pyramid, width, height, gaussian_pyramid.num_levels);

    // Build Laplacian pyramid
    for (int i = 0; i < gaussian_pyramid.num_levels - 1; i++) {
        // Upsample next level
        float* d_upsampled;
        CUDA_CHECK(cudaMalloc(&d_upsampled, gaussian_pyramid.widths[i] * gaussian_pyramid.heights[i] * sizeof(float)));

        upsample2x(gaussian_pyramid.d_levels[i+1], d_upsampled,
                   gaussian_pyramid.widths[i+1], gaussian_pyramid.heights[i+1]);

        // Subtract
        dim3 block(16, 16);
        dim3 grid((gaussian_pyramid.widths[i] + block.x - 1) / block.x,
                  (gaussian_pyramid.heights[i] + block.y - 1) / block.y);

        laplacianSubtractKernel<<<grid, block>>>(
            gaussian_pyramid.d_levels[i], d_upsampled, laplacian_pyramid.d_levels[i],
            gaussian_pyramid.widths[i], gaussian_pyramid.heights[i]
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_upsampled));
    }

    // Top level of Laplacian is same as top level of Gaussian
    CUDA_CHECK(cudaMemcpy(
        laplacian_pyramid.d_levels[gaussian_pyramid.num_levels - 1],
        gaussian_pyramid.d_levels[gaussian_pyramid.num_levels - 1],
        gaussian_pyramid.widths[gaussian_pyramid.num_levels - 1] *
        gaussian_pyramid.heights[gaussian_pyramid.num_levels - 1] * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));
}

// Laplacian addition kernel
__global__ void laplacianAddKernel(
    const float* __restrict__ upsampled,
    const float* __restrict__ laplacian,
    float* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = upsampled[idx] + laplacian[idx];
}

void reconstructFromLaplacian(
    const GaussianPyramid& laplacian_pyramid,
    float* d_output,
    int width,
    int height
) {
    int top_level = laplacian_pyramid.num_levels - 1;

    // Start with top level
    float* d_current;
    CUDA_CHECK(cudaMalloc(&d_current,
        laplacian_pyramid.widths[top_level] * laplacian_pyramid.heights[top_level] * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_current, laplacian_pyramid.d_levels[top_level],
        laplacian_pyramid.widths[top_level] * laplacian_pyramid.heights[top_level] * sizeof(float),
        cudaMemcpyDeviceToDevice));

    // Reconstruct from top to bottom
    for (int i = top_level - 1; i >= 0; i--) {
        float* d_upsampled;
        CUDA_CHECK(cudaMalloc(&d_upsampled,
            laplacian_pyramid.widths[i] * laplacian_pyramid.heights[i] * sizeof(float)));

        upsample2x(d_current, d_upsampled,
                   laplacian_pyramid.widths[i+1], laplacian_pyramid.heights[i+1]);

        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaMalloc(&d_current,
            laplacian_pyramid.widths[i] * laplacian_pyramid.heights[i] * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((laplacian_pyramid.widths[i] + block.x - 1) / block.x,
                  (laplacian_pyramid.heights[i] + block.y - 1) / block.y);

        laplacianAddKernel<<<grid, block>>>(
            d_upsampled, laplacian_pyramid.d_levels[i], d_current,
            laplacian_pyramid.widths[i], laplacian_pyramid.heights[i]
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_upsampled));
    }

    // Copy result
    CUDA_CHECK(cudaMemcpy(d_output, d_current, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_current));
}

} // namespace cuda_stabilizer
