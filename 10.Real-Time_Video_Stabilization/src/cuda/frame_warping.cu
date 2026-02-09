#include "frame_warping.cuh"

namespace cuda_stabilizer {

// Static resources
static float* d_warp_temp = nullptr;
static int warp_allocated_width = 0;
static int warp_allocated_height = 0;

void initFrameWarping(int width, int height) {
    if (d_warp_temp != nullptr) {
        releaseFrameWarping();
    }

    CUDA_CHECK(cudaMalloc(&d_warp_temp, width * height * 3 * sizeof(float)));
    warp_allocated_width = width;
    warp_allocated_height = height;
}

void releaseFrameWarping() {
    if (d_warp_temp) { cudaFree(d_warp_temp); d_warp_temp = nullptr; }
    warp_allocated_width = 0;
    warp_allocated_height = 0;
}

// Bilinear interpolation device function
__device__ inline float bilinearInterpolateDev(
    const unsigned char* image,
    float x,
    float y,
    int width,
    int height,
    int channel,
    int channels
) {
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp coordinates
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    float fx = x - floorf(x);
    float fy = y - floorf(y);

    float v00 = image[(y0 * width + x0) * channels + channel];
    float v01 = image[(y0 * width + x1) * channels + channel];
    float v10 = image[(y1 * width + x0) * channels + channel];
    float v11 = image[(y1 * width + x1) * channels + channel];

    float v0 = v00 * (1.0f - fx) + v01 * fx;
    float v1 = v10 * (1.0f - fx) + v11 * fx;

    return v0 * (1.0f - fy) + v1 * fy;
}

// Affine warp kernel
__global__ void warpAffineKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    float dx,
    float dy,
    float angle,
    float scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Center of image
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    // Transform coordinates (inverse mapping)
    float cos_a = cosf(-angle);
    float sin_a = sinf(-angle);
    float inv_scale = 1.0f / scale;

    // Translate to origin
    float px = x - cx;
    float py = y - cy;

    // Apply inverse scale
    px *= inv_scale;
    py *= inv_scale;

    // Apply inverse rotation
    float rx = px * cos_a - py * sin_a;
    float ry = px * sin_a + py * cos_a;

    // Apply inverse translation and translate back
    float src_x = rx + cx - dx;
    float src_y = ry + cy - dy;

    int out_idx = (y * width + x) * channels;

    if (src_x < 0 || src_x >= width - 1 || src_y < 0 || src_y >= height - 1) {
        // Out of bounds - fill with black
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = 0;
        }
    } else {
        // Bilinear interpolation
        for (int c = 0; c < channels; c++) {
            float val = bilinearInterpolateDev(input, src_x, src_y, width, height, c, channels);
            output[out_idx + c] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
        }
    }
}

void warpFrameAffine(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    const TransformParams& transform
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    warpAffineKernel<<<grid, block>>>(
        d_input, d_output,
        width, height, channels,
        transform.dx, transform.dy, transform.da, transform.ds
    );
    CUDA_CHECK(cudaGetLastError());
}

// Affine warp with crop kernel
__global__ void warpAffineCropKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    float dx,
    float dy,
    float angle,
    float scale,
    float crop_ratio
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Center of image
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    // Crop scaling
    float crop_scale = 1.0f / crop_ratio;

    // Map output coordinates to cropped region
    float ox = (x - cx) * crop_scale + cx;
    float oy = (y - cy) * crop_scale + cy;

    // Transform coordinates (inverse mapping)
    float cos_a = cosf(-angle);
    float sin_a = sinf(-angle);
    float inv_scale = 1.0f / scale;

    // Translate to origin
    float px = ox - cx;
    float py = oy - cy;

    // Apply inverse scale
    px *= inv_scale;
    py *= inv_scale;

    // Apply inverse rotation
    float rx = px * cos_a - py * sin_a;
    float ry = px * sin_a + py * cos_a;

    // Apply inverse translation and translate back
    float src_x = rx + cx - dx;
    float src_y = ry + cy - dy;

    int out_idx = (y * width + x) * channels;

    if (src_x < 0 || src_x >= width - 1 || src_y < 0 || src_y >= height - 1) {
        // Out of bounds - use edge pixel
        src_x = fminf(fmaxf(src_x, 0.0f), width - 1.0f);
        src_y = fminf(fmaxf(src_y, 0.0f), height - 1.0f);
    }

    // Bilinear interpolation
    for (int c = 0; c < channels; c++) {
        float val = bilinearInterpolateDev(input, src_x, src_y, width, height, c, channels);
        output[out_idx + c] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }
}

void warpFrameAffineCrop(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    const TransformParams& transform,
    float crop_ratio
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    warpAffineCropKernel<<<grid, block>>>(
        d_input, d_output,
        width, height, channels,
        transform.dx, transform.dy, transform.da, transform.ds,
        crop_ratio
    );
    CUDA_CHECK(cudaGetLastError());
}

// Perspective warp kernel
__global__ void warpPerspectiveKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    const float* __restrict__ H  // 3x3 homography matrix
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Compute inverse homography coordinates
    float w = H[6] * x + H[7] * y + H[8];
    if (fabsf(w) < 1e-8f) w = 1e-8f;

    float src_x = (H[0] * x + H[1] * y + H[2]) / w;
    float src_y = (H[3] * x + H[4] * y + H[5]) / w;

    int out_idx = (y * width + x) * channels;

    if (src_x < 0 || src_x >= width - 1 || src_y < 0 || src_y >= height - 1) {
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = 0;
        }
    } else {
        for (int c = 0; c < channels; c++) {
            float val = bilinearInterpolateDev(input, src_x, src_y, width, height, c, channels);
            output[out_idx + c] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
        }
    }
}

void warpFramePerspective(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    const float* d_homography
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    warpPerspectiveKernel<<<grid, block>>>(
        d_input, d_output,
        width, height, channels,
        d_homography
    );
    CUDA_CHECK(cudaGetLastError());
}

// BGR to grayscale kernel
__global__ void bgr2grayKernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int in_idx = (y * width + x) * channels;
    int out_idx = y * width + x;

    if (channels >= 3) {
        // BGR to grayscale: 0.114 * B + 0.587 * G + 0.299 * R
        float b = input[in_idx];
        float g = input[in_idx + 1];
        float r = input[in_idx + 2];
        output[out_idx] = 0.114f * b + 0.587f * g + 0.299f * r;
    } else {
        output[out_idx] = input[in_idx];
    }
}

void convertToGrayscale(
    const unsigned char* d_input,
    float* d_output,
    int width,
    int height,
    int channels
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    bgr2grayKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}

// Grayscale to float kernel
__global__ void gray2floatKernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = (float)input[idx];
}

void convertToFloat(
    const unsigned char* d_input,
    float* d_output,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    gray2floatKernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// Border filling kernel
__global__ void fillBorderKernel(
    unsigned char* __restrict__ image,
    int width,
    int height,
    int channels,
    int border_size,
    BorderMode mode
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    bool is_border = (x < border_size || x >= width - border_size ||
                      y < border_size || y >= height - border_size);

    if (!is_border) return;

    int idx = (y * width + x) * channels;

    if (mode == BorderMode::CONSTANT) {
        for (int c = 0; c < channels; c++) {
            image[idx + c] = 0;
        }
    } else if (mode == BorderMode::REPLICATE) {
        int src_x = min(max(x, border_size), width - border_size - 1);
        int src_y = min(max(y, border_size), height - border_size - 1);
        int src_idx = (src_y * width + src_x) * channels;
        for (int c = 0; c < channels; c++) {
            image[idx + c] = image[src_idx + c];
        }
    } else if (mode == BorderMode::REFLECT) {
        int src_x = x < border_size ? border_size + (border_size - x) :
                    x >= width - border_size ? width - border_size - 2 - (x - width + border_size) : x;
        int src_y = y < border_size ? border_size + (border_size - y) :
                    y >= height - border_size ? height - border_size - 2 - (y - height + border_size) : y;
        src_x = max(0, min(src_x, width - 1));
        src_y = max(0, min(src_y, height - 1));
        int src_idx = (src_y * width + src_x) * channels;
        for (int c = 0; c < channels; c++) {
            image[idx + c] = image[src_idx + c];
        }
    }
}

void applyBorder(
    unsigned char* d_image,
    int width,
    int height,
    int channels,
    int border_size,
    BorderMode mode
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    fillBorderKernel<<<grid, block>>>(d_image, width, height, channels, border_size, mode);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_stabilizer
