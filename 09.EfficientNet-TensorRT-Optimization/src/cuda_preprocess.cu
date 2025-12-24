#include "cuda_preprocess.h"
#include <cmath>
#include <cfloat>

namespace efficientnet {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return err; \
        } \
    } while(0)

// Fused resize + normalize + HWC->NCHW kernel
__global__ void preprocessFusedKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // Calculate source coordinates (bilinear interpolation)
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    // Clamp to valid range
    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    // Bilinear interpolation for each channel
    float rgb[3];
    for (int c = 0; c < 3; c++) {
        float v00 = src[(y0 * src_width + x0) * 3 + c];
        float v01 = src[(y0 * src_width + x1) * 3 + c];
        float v10 = src[(y1 * src_width + x0) * 3 + c];
        float v11 = src[(y1 * src_width + x1) * 3 + c];

        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        rgb[c] = v0 * (1.0f - dy) + v1 * dy;
    }

    // Normalize and convert to float
    float r = (rgb[0] / 255.0f - mean_r) / std_r;
    float g = (rgb[1] / 255.0f - mean_g) / std_g;
    float b = (rgb[2] / 255.0f - mean_b) / std_b;

    // Write in NCHW format
    int dst_idx = dst_y * dst_width + dst_x;
    int plane_size = dst_width * dst_height;
    dst[0 * plane_size + dst_idx] = r;  // R channel
    dst[1 * plane_size + dst_idx] = g;  // G channel
    dst[2 * plane_size + dst_idx] = b;  // B channel
}

cudaError_t preprocessImageFused(
    const uint8_t* src,
    float* dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    const float* mean,
    const float* std,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x,
              (dst_height + block.y - 1) / block.y);

    preprocessFusedKernel<<<grid, block, 0, stream>>>(
        src, dst,
        src_width, src_height,
        dst_width, dst_height,
        mean[0], mean[1], mean[2],
        std[0], std[1], std[2]
    );

    return cudaGetLastError();
}

// Bilinear resize kernel
__global__ void resizeBilinearKernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    for (int c = 0; c < 3; c++) {
        float v00 = src[(y0 * src_width + x0) * 3 + c];
        float v01 = src[(y0 * src_width + x1) * 3 + c];
        float v10 = src[(y1 * src_width + x0) * 3 + c];
        float v11 = src[(y1 * src_width + x1) * 3 + c];

        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        float val = v0 * (1.0f - dy) + v1 * dy;

        dst[(dst_y * dst_width + dst_x) * 3 + c] = static_cast<uint8_t>(val + 0.5f);
    }
}

cudaError_t resizeBilinear(
    const uint8_t* src,
    uint8_t* dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x,
              (dst_height + block.y - 1) / block.y);

    resizeBilinearKernel<<<grid, block, 0, stream>>>(
        src, dst, src_width, src_height, dst_width, dst_height
    );

    return cudaGetLastError();
}

// Normalize kernel
__global__ void normalizeKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int width, int height,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    dst[idx + 0] = (src[idx + 0] / 255.0f - mean_r) / std_r;
    dst[idx + 1] = (src[idx + 1] / 255.0f - mean_g) / std_g;
    dst[idx + 2] = (src[idx + 2] / 255.0f - mean_b) / std_b;
}

cudaError_t normalizeImage(
    const uint8_t* src,
    float* dst,
    int width, int height,
    const float* mean,
    const float* std,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    normalizeKernel<<<grid, block, 0, stream>>>(
        src, dst, width, height,
        mean[0], mean[1], mean[2],
        std[0], std[1], std[2]
    );

    return cudaGetLastError();
}

// HWC to NCHW conversion kernel
__global__ void hwcToNchwKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int height, int width, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int plane_size = width * height;
    int src_idx = (y * width + x) * channels;
    int dst_idx = y * width + x;

    for (int c = 0; c < channels; c++) {
        dst[c * plane_size + dst_idx] = src[src_idx + c];
    }
}

cudaError_t hwcToNchw(
    const float* src,
    float* dst,
    int height, int width, int channels,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    hwcToNchwKernel<<<grid, block, 0, stream>>>(
        src, dst, height, width, channels
    );

    return cudaGetLastError();
}

// Softmax kernel
__global__ void softmaxKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    const float* in = input + batch_idx * num_classes;
    float* out = output + batch_idx * num_classes;

    // Find max for numerical stability
    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + 1;

    if (threadIdx.x == 0) {
        max_val[0] = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_val[0] = fmaxf(max_val[0], in[i]);
        }
    }
    __syncthreads();

    // Compute exp and sum
    if (threadIdx.x == 0) {
        sum_exp[0] = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            out[i] = expf(in[i] - max_val[0]);
            sum_exp[0] += out[i];
        }
    }
    __syncthreads();

    // Normalize
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_classes; i++) {
            out[i] /= sum_exp[0];
        }
    }
}

cudaError_t softmax(
    const float* input,
    float* output,
    int batch_size,
    int num_classes,
    cudaStream_t stream
) {
    softmaxKernel<<<batch_size, 1, 2 * sizeof(float), stream>>>(
        input, output, num_classes
    );
    return cudaGetLastError();
}

// ArgMax kernel
__global__ void argmaxKernel(
    const float* __restrict__ input,
    int* __restrict__ output,
    float* __restrict__ max_values,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    const float* in = input + batch_idx * num_classes;

    int max_idx = 0;
    float max_val = in[0];

    for (int i = 1; i < num_classes; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }

    output[batch_idx] = max_idx;
    if (max_values) {
        max_values[batch_idx] = max_val;
    }
}

cudaError_t argmax(
    const float* input,
    int* output,
    float* max_values,
    int batch_size,
    int num_classes,
    cudaStream_t stream
) {
    argmaxKernel<<<batch_size, 1, 0, stream>>>(input, output, max_values, num_classes);
    return cudaGetLastError();
}

// TopK kernel (simple implementation for small K)
__global__ void topKKernel(
    const float* __restrict__ input,
    int* __restrict__ indices,
    float* __restrict__ values,
    int num_classes,
    int k
) {
    int batch_idx = blockIdx.x;
    const float* in = input + batch_idx * num_classes;
    int* out_idx = indices + batch_idx * k;
    float* out_val = values + batch_idx * k;

    // Initialize with minimum values
    for (int i = 0; i < k; i++) {
        out_val[i] = -FLT_MAX;
        out_idx[i] = -1;
    }

    // Find top K
    for (int i = 0; i < num_classes; i++) {
        float val = in[i];

        // Check if this value should be in top K
        if (val > out_val[k - 1]) {
            // Find insertion position
            int pos = k - 1;
            while (pos > 0 && val > out_val[pos - 1]) {
                out_val[pos] = out_val[pos - 1];
                out_idx[pos] = out_idx[pos - 1];
                pos--;
            }
            out_val[pos] = val;
            out_idx[pos] = i;
        }
    }
}

cudaError_t topK(
    const float* input,
    int* indices,
    float* values,
    int batch_size,
    int num_classes,
    int k,
    cudaStream_t stream
) {
    topKKernel<<<batch_size, 1, 0, stream>>>(input, indices, values, num_classes, k);
    return cudaGetLastError();
}

// Batch preprocessing
__global__ void preprocessBatchKernel(
    const uint8_t* const* __restrict__ src_batch,
    float* __restrict__ dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    const uint8_t* src = src_batch[batch_idx];
    int plane_size = dst_width * dst_height;
    int batch_offset = batch_idx * 3 * plane_size;

    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    float rgb[3];
    for (int c = 0; c < 3; c++) {
        float v00 = src[(y0 * src_width + x0) * 3 + c];
        float v01 = src[(y0 * src_width + x1) * 3 + c];
        float v10 = src[(y1 * src_width + x0) * 3 + c];
        float v11 = src[(y1 * src_width + x1) * 3 + c];

        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        rgb[c] = v0 * (1.0f - dy) + v1 * dy;
    }

    float r = (rgb[0] / 255.0f - mean_r) / std_r;
    float g = (rgb[1] / 255.0f - mean_g) / std_g;
    float b = (rgb[2] / 255.0f - mean_b) / std_b;

    int dst_idx = dst_y * dst_width + dst_x;
    dst[batch_offset + 0 * plane_size + dst_idx] = r;
    dst[batch_offset + 1 * plane_size + dst_idx] = g;
    dst[batch_offset + 2 * plane_size + dst_idx] = b;
}

cudaError_t preprocessBatch(
    const uint8_t* const* src_batch,
    float* dst,
    int batch_size,
    const PreprocessParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((params.dst_width + block.x - 1) / block.x,
              (params.dst_height + block.y - 1) / block.y,
              batch_size);

    preprocessBatchKernel<<<grid, block, 0, stream>>>(
        src_batch, dst,
        params.src_width, params.src_height,
        params.dst_width, params.dst_height,
        params.mean[0], params.mean[1], params.mean[2],
        params.std[0], params.std[1], params.std[2]
    );

    return cudaGetLastError();
}

}  // namespace efficientnet
