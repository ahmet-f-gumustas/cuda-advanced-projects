#include "optical_flow.cuh"

namespace cuda_stabilizer {

// Texture references for fast memory access
static float* d_grad_x = nullptr;
static float* d_grad_y = nullptr;
static float* d_grad_t = nullptr;
static float* d_temp_buffer = nullptr;
static int allocated_width = 0;
static int allocated_height = 0;

// Sobel kernel constants
__constant__ float c_sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float c_sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// Gaussian kernel for smoothing (5x5)
__constant__ float c_gaussian[25] = {
    0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f,
    0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
    0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
    0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
    0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f
};

void initOpticalFlow(int width, int height, int pyramid_levels) {
    if (d_grad_x != nullptr) {
        releaseOpticalFlow();
    }

    size_t total_size = width * height;

    CUDA_CHECK(cudaMalloc(&d_grad_x, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_t, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, total_size * sizeof(float)));

    allocated_width = width;
    allocated_height = height;
}

void releaseOpticalFlow() {
    if (d_grad_x) { cudaFree(d_grad_x); d_grad_x = nullptr; }
    if (d_grad_y) { cudaFree(d_grad_y); d_grad_y = nullptr; }
    if (d_grad_t) { cudaFree(d_grad_t); d_grad_t = nullptr; }
    if (d_temp_buffer) { cudaFree(d_temp_buffer); d_temp_buffer = nullptr; }
    allocated_width = 0;
    allocated_height = 0;
}

// Sobel gradient computation kernel
__global__ void sobelGradientKernel(
    const float* __restrict__ input,
    float* __restrict__ grad_x,
    float* __restrict__ grad_y,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float gx = 0.0f, gy = 0.0f;

    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            float val = input[ny * width + nx];
            int kidx = (ky + 1) * 3 + (kx + 1);
            gx += val * c_sobel_x[kidx];
            gy += val * c_sobel_y[kidx];
        }
    }

    int idx = y * width + x;
    grad_x[idx] = gx;
    grad_y[idx] = gy;
}

void computeGradients(
    const float* d_image,
    float* d_grad_x_out,
    float* d_grad_y_out,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobelGradientKernel<<<grid, block>>>(d_image, d_grad_x_out, d_grad_y_out, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// Temporal gradient kernel
__global__ void temporalGradientKernel(
    const float* __restrict__ prev_frame,
    const float* __restrict__ curr_frame,
    float* __restrict__ grad_t,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    grad_t[idx] = curr_frame[idx] - prev_frame[idx];
}

void computeTemporalGradient(
    const float* d_prev_frame,
    const float* d_curr_frame,
    float* d_grad_t_out,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    temporalGradientKernel<<<grid, block>>>(d_prev_frame, d_curr_frame, d_grad_t_out, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// Lucas-Kanade optical flow kernel
__global__ void lucasKanadeKernel(
    const float* __restrict__ grad_x,
    const float* __restrict__ grad_y,
    const float* __restrict__ grad_t,
    float* __restrict__ flow_x,
    float* __restrict__ flow_y,
    int width,
    int height,
    int window_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_win = window_size / 2;

    // Compute structure tensor elements
    float sum_Ix2 = 0.0f;
    float sum_Iy2 = 0.0f;
    float sum_IxIy = 0.0f;
    float sum_IxIt = 0.0f;
    float sum_IyIt = 0.0f;

    for (int wy = -half_win; wy <= half_win; wy++) {
        for (int wx = -half_win; wx <= half_win; wx++) {
            int nx = min(max(x + wx, 0), width - 1);
            int ny = min(max(y + wy, 0), height - 1);
            int idx = ny * width + nx;

            float Ix = grad_x[idx];
            float Iy = grad_y[idx];
            float It = grad_t[idx];

            sum_Ix2 += Ix * Ix;
            sum_Iy2 += Iy * Iy;
            sum_IxIy += Ix * Iy;
            sum_IxIt += Ix * It;
            sum_IyIt += Iy * It;
        }
    }

    // Solve 2x2 system using Cramer's rule
    float det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;
    int out_idx = y * width + x;

    if (fabsf(det) > 1e-6f) {
        flow_x[out_idx] = (sum_IxIy * sum_IyIt - sum_Iy2 * sum_IxIt) / det;
        flow_y[out_idx] = (sum_IxIy * sum_IxIt - sum_Ix2 * sum_IyIt) / det;
    } else {
        flow_x[out_idx] = 0.0f;
        flow_y[out_idx] = 0.0f;
    }
}

void computeOpticalFlowLK(
    const float* d_prev_frame,
    const float* d_curr_frame,
    float* d_flow_x,
    float* d_flow_y,
    int width,
    int height,
    const OpticalFlowParams& params
) {
    // Compute spatial gradients
    computeGradients(d_prev_frame, d_grad_x, d_grad_y, width, height);

    // Compute temporal gradient
    computeTemporalGradient(d_prev_frame, d_curr_frame, d_grad_t, width, height);

    // Compute optical flow
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    lucasKanadeKernel<<<grid, block>>>(
        d_grad_x, d_grad_y, d_grad_t,
        d_flow_x, d_flow_y,
        width, height, params.window_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// Horn-Schunck iterative kernel
__global__ void hornSchunckKernel(
    const float* __restrict__ grad_x,
    const float* __restrict__ grad_y,
    const float* __restrict__ grad_t,
    const float* __restrict__ flow_x_prev,
    const float* __restrict__ flow_y_prev,
    float* __restrict__ flow_x,
    float* __restrict__ flow_y,
    int width,
    int height,
    float alpha
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Compute local average of flow
    float avg_u = 0.0f, avg_v = 0.0f;
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                avg_u += flow_x_prev[nidx];
                avg_v += flow_y_prev[nidx];
                count++;
            }
        }
    }

    if (count > 0) {
        avg_u /= count;
        avg_v /= count;
    }

    float Ix = grad_x[idx];
    float Iy = grad_y[idx];
    float It = grad_t[idx];

    float denom = alpha * alpha + Ix * Ix + Iy * Iy;
    float P = (Ix * avg_u + Iy * avg_v + It);

    flow_x[idx] = avg_u - Ix * P / denom;
    flow_y[idx] = avg_v - Iy * P / denom;
}

void computeOpticalFlowHS(
    const float* d_prev_frame,
    const float* d_curr_frame,
    float* d_flow_x,
    float* d_flow_y,
    int width,
    int height,
    float alpha,
    int iterations
) {
    // Compute gradients
    computeGradients(d_prev_frame, d_grad_x, d_grad_y, width, height);
    computeTemporalGradient(d_prev_frame, d_curr_frame, d_grad_t, width, height);

    // Initialize flow to zero
    CUDA_CHECK(cudaMemset(d_flow_x, 0, width * height * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flow_y, 0, width * height * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Allocate temp buffers for ping-pong
    float* d_flow_x_temp;
    float* d_flow_y_temp;
    CUDA_CHECK(cudaMalloc(&d_flow_x_temp, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flow_y_temp, width * height * sizeof(float)));

    float* d_flow_x_curr = d_flow_x;
    float* d_flow_y_curr = d_flow_y;
    float* d_flow_x_next = d_flow_x_temp;
    float* d_flow_y_next = d_flow_y_temp;

    for (int i = 0; i < iterations; i++) {
        hornSchunckKernel<<<grid, block>>>(
            d_grad_x, d_grad_y, d_grad_t,
            d_flow_x_curr, d_flow_y_curr,
            d_flow_x_next, d_flow_y_next,
            width, height, alpha
        );
        CUDA_CHECK(cudaGetLastError());

        // Swap buffers
        std::swap(d_flow_x_curr, d_flow_x_next);
        std::swap(d_flow_y_curr, d_flow_y_next);
    }

    // Copy result back if needed
    if (d_flow_x_curr != d_flow_x) {
        CUDA_CHECK(cudaMemcpy(d_flow_x, d_flow_x_curr, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_flow_y, d_flow_y_curr, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_flow_x_temp));
    CUDA_CHECK(cudaFree(d_flow_y_temp));
}

} // namespace cuda_stabilizer
