#include "motion_estimation.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <curand_kernel.h>

namespace cuda_stabilizer {

// Static resources
static MotionVector* d_motion_vectors_buffer = nullptr;
static float* d_harris_response = nullptr;
static int me_allocated_width = 0;
static int me_allocated_height = 0;

void initMotionEstimation(int width, int height) {
    if (d_motion_vectors_buffer != nullptr) {
        releaseMotionEstimation();
    }

    int num_blocks_x = (width + 15) / 16;
    int num_blocks_y = (height + 15) / 16;

    CUDA_CHECK(cudaMalloc(&d_motion_vectors_buffer, num_blocks_x * num_blocks_y * sizeof(MotionVector)));
    CUDA_CHECK(cudaMalloc(&d_harris_response, width * height * sizeof(float)));

    me_allocated_width = width;
    me_allocated_height = height;
}

void releaseMotionEstimation() {
    if (d_motion_vectors_buffer) { cudaFree(d_motion_vectors_buffer); d_motion_vectors_buffer = nullptr; }
    if (d_harris_response) { cudaFree(d_harris_response); d_harris_response = nullptr; }
    me_allocated_width = 0;
    me_allocated_height = 0;
}

// Sum of Absolute Differences (SAD) for block matching
__device__ float computeSAD(
    const float* __restrict__ prev_frame,
    const float* __restrict__ curr_frame,
    int px, int py,
    int cx, int cy,
    int block_size,
    int width,
    int height
) {
    float sad = 0.0f;
    int half_block = block_size / 2;

    for (int dy = -half_block; dy <= half_block; dy++) {
        for (int dx = -half_block; dx <= half_block; dx++) {
            int prev_x = min(max(px + dx, 0), width - 1);
            int prev_y = min(max(py + dy, 0), height - 1);
            int curr_x = min(max(cx + dx, 0), width - 1);
            int curr_y = min(max(cy + dy, 0), height - 1);

            float diff = prev_frame[prev_y * width + prev_x] - curr_frame[curr_y * width + curr_x];
            sad += fabsf(diff);
        }
    }

    return sad;
}

// Block matching kernel
__global__ void blockMatchingKernel(
    const float* __restrict__ prev_frame,
    const float* __restrict__ curr_frame,
    MotionVector* __restrict__ motion_vectors,
    int width,
    int height,
    int block_size,
    int search_range
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int num_blocks_x = gridDim.x;

    int center_x = bx * block_size + block_size / 2;
    int center_y = by * block_size + block_size / 2;

    if (center_x >= width || center_y >= height) return;

    float min_sad = 1e10f;
    int best_dx = 0;
    int best_dy = 0;

    // Three-step search for efficiency
    int step_sizes[] = {search_range / 2, search_range / 4, 1};

    int curr_x = center_x;
    int curr_y = center_y;

    for (int step_idx = 0; step_idx < 3; step_idx++) {
        int step = step_sizes[step_idx];
        int local_best_dx = 0;
        int local_best_dy = 0;
        float local_min_sad = min_sad;

        for (int dy = -step; dy <= step; dy += step) {
            for (int dx = -step; dx <= step; dx += step) {
                if (step_idx > 0 && dx == 0 && dy == 0) continue;

                int search_x = curr_x + dx;
                int search_y = curr_y + dy;

                if (search_x < 0 || search_x >= width || search_y < 0 || search_y >= height) continue;

                float sad = computeSAD(prev_frame, curr_frame, center_x, center_y, search_x, search_y, block_size, width, height);

                if (sad < local_min_sad) {
                    local_min_sad = sad;
                    local_best_dx = dx;
                    local_best_dy = dy;
                }
            }
        }

        curr_x += local_best_dx;
        curr_y += local_best_dy;
        min_sad = local_min_sad;
        best_dx = curr_x - center_x;
        best_dy = curr_y - center_y;
    }

    int mv_idx = by * num_blocks_x + bx;
    motion_vectors[mv_idx].dx = static_cast<float>(best_dx);
    motion_vectors[mv_idx].dy = static_cast<float>(best_dy);
}

void computeBlockMotion(
    const float* d_prev_frame,
    const float* d_curr_frame,
    MotionVector* d_motion_vectors,
    int width,
    int height,
    const MotionEstimationParams& params
) {
    int num_blocks_x = (width + params.block_size - 1) / params.block_size;
    int num_blocks_y = (height + params.block_size - 1) / params.block_size;

    dim3 grid(num_blocks_x, num_blocks_y);
    dim3 block(1, 1);

    blockMatchingKernel<<<grid, block>>>(
        d_prev_frame, d_curr_frame,
        d_motion_vectors,
        width, height,
        params.block_size, params.search_range
    );
    CUDA_CHECK(cudaGetLastError());
}

// Compute global motion from motion vectors on CPU
TransformParams computeGlobalMotion(
    const MotionVector* d_motion_vectors,
    int num_blocks_x,
    int num_blocks_y,
    int block_size,
    int width,
    int height
) {
    int num_vectors = num_blocks_x * num_blocks_y;

    // Copy motion vectors to host
    std::vector<MotionVector> h_motion_vectors(num_vectors);
    CUDA_CHECK(cudaMemcpy(h_motion_vectors.data(), d_motion_vectors,
                          num_vectors * sizeof(MotionVector), cudaMemcpyDeviceToHost));

    // Compute weighted average motion
    float sum_dx = 0.0f, sum_dy = 0.0f;
    float sum_rotation = 0.0f;
    float total_weight = 0.0f;

    float cx = width / 2.0f;
    float cy = height / 2.0f;

    for (int by = 0; by < num_blocks_y; by++) {
        for (int bx = 0; bx < num_blocks_x; bx++) {
            int idx = by * num_blocks_x + bx;
            const MotionVector& mv = h_motion_vectors[idx];

            float block_cx = bx * block_size + block_size / 2.0f;
            float block_cy = by * block_size + block_size / 2.0f;

            // Weight based on distance from center (center blocks are more reliable)
            float dist_from_center = sqrtf((block_cx - cx) * (block_cx - cx) + (block_cy - cy) * (block_cy - cy));
            float max_dist = sqrtf(cx * cx + cy * cy);
            float weight = 1.0f - (dist_from_center / max_dist) * 0.5f;

            // Filter out outliers (large motions are likely errors)
            float motion_mag = sqrtf(mv.dx * mv.dx + mv.dy * mv.dy);
            if (motion_mag > block_size * 2) {
                weight *= 0.1f;
            }

            sum_dx += mv.dx * weight;
            sum_dy += mv.dy * weight;

            // Compute rotation contribution
            float rx = block_cx - cx;
            float ry = block_cy - cy;
            float r2 = rx * rx + ry * ry;
            if (r2 > 1.0f) {
                float rot = (rx * mv.dy - ry * mv.dx) / r2;
                sum_rotation += rot * weight;
            }

            total_weight += weight;
        }
    }

    TransformParams params;
    if (total_weight > 0.0f) {
        params.dx = sum_dx / total_weight;
        params.dy = sum_dy / total_weight;
        params.da = sum_rotation / total_weight;
        params.ds = 1.0f;  // Scale estimation not implemented
    }

    return params;
}

// Harris corner detection kernel
__global__ void harrisCornerKernel(
    const float* __restrict__ grad_x,
    const float* __restrict__ grad_y,
    float* __restrict__ response,
    int width,
    int height,
    int window_size,
    float k
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_win = window_size / 2;

    float sum_Ix2 = 0.0f;
    float sum_Iy2 = 0.0f;
    float sum_IxIy = 0.0f;

    for (int wy = -half_win; wy <= half_win; wy++) {
        for (int wx = -half_win; wx <= half_win; wx++) {
            int nx = min(max(x + wx, 0), width - 1);
            int ny = min(max(y + wy, 0), height - 1);
            int idx = ny * width + nx;

            float Ix = grad_x[idx];
            float Iy = grad_y[idx];

            sum_Ix2 += Ix * Ix;
            sum_Iy2 += Iy * Iy;
            sum_IxIy += Ix * Iy;
        }
    }

    // Harris response: det(M) - k * trace(M)^2
    float det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;
    float trace = sum_Ix2 + sum_Iy2;
    float R = det - k * trace * trace;

    response[y * width + x] = R;
}

void detectFeaturePoints(
    const float* d_image,
    float* d_corners,
    int* d_num_corners,
    int width,
    int height,
    float threshold,
    int max_corners
) {
    // Compute gradients
    float* d_grad_x;
    float* d_grad_y;
    CUDA_CHECK(cudaMalloc(&d_grad_x, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, width * height * sizeof(float)));

    // Compute Sobel gradients (using external function)
    extern void computeGradients(const float*, float*, float*, int, int);
    computeGradients(d_image, d_grad_x, d_grad_y, width, height);

    // Compute Harris response
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    harrisCornerKernel<<<grid, block>>>(
        d_grad_x, d_grad_y,
        d_harris_response,
        width, height,
        5,      // window size
        0.04f   // k parameter
    );
    CUDA_CHECK(cudaGetLastError());

    // Non-maximum suppression and thresholding would be done here
    // For simplicity, using host-side processing

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
}

// Lucas-Kanade feature tracking kernel
__global__ void trackFeaturesKernel(
    const float* __restrict__ prev_frame,
    const float* __restrict__ curr_frame,
    const float* __restrict__ prev_points,
    float* __restrict__ curr_points,
    unsigned char* __restrict__ status,
    int num_points,
    int width,
    int height,
    int window_size,
    int max_iterations,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float px = prev_points[idx * 2];
    float py = prev_points[idx * 2 + 1];

    if (px < 0 || px >= width - 1 || py < 0 || py >= height - 1) {
        status[idx] = 0;
        return;
    }

    int half_win = window_size / 2;
    float u = 0.0f, v = 0.0f;

    for (int iter = 0; iter < max_iterations; iter++) {
        float sum_Ix2 = 0.0f;
        float sum_Iy2 = 0.0f;
        float sum_IxIy = 0.0f;
        float sum_IxIt = 0.0f;
        float sum_IyIt = 0.0f;

        for (int wy = -half_win; wy <= half_win; wy++) {
            for (int wx = -half_win; wx <= half_win; wx++) {
                float sx = px + wx;
                float sy = py + wy;
                float dx = px + u + wx;
                float dy = py + v + wy;

                if (sx < 0 || sx >= width - 1 || sy < 0 || sy >= height - 1 ||
                    dx < 0 || dx >= width - 1 || dy < 0 || dy >= height - 1) continue;

                // Bilinear interpolation for prev frame
                int sx0 = (int)sx, sy0 = (int)sy;
                float fx = sx - sx0, fy = sy - sy0;
                float I_prev = (1-fx)*(1-fy)*prev_frame[sy0*width+sx0] +
                               fx*(1-fy)*prev_frame[sy0*width+sx0+1] +
                               (1-fx)*fy*prev_frame[(sy0+1)*width+sx0] +
                               fx*fy*prev_frame[(sy0+1)*width+sx0+1];

                // Bilinear interpolation for curr frame
                int dx0 = (int)dx, dy0 = (int)dy;
                float fdx = dx - dx0, fdy = dy - dy0;
                float I_curr = (1-fdx)*(1-fdy)*curr_frame[dy0*width+dx0] +
                               fdx*(1-fdy)*curr_frame[dy0*width+dx0+1] +
                               (1-fdx)*fdy*curr_frame[(dy0+1)*width+dx0] +
                               fdx*fdy*curr_frame[(dy0+1)*width+dx0+1];

                // Compute gradients
                float Ix = 0.5f * (prev_frame[sy0*width+min(sx0+1, width-1)] - prev_frame[sy0*width+max(sx0-1, 0)]);
                float Iy = 0.5f * (prev_frame[min(sy0+1, height-1)*width+sx0] - prev_frame[max(sy0-1, 0)*width+sx0]);
                float It = I_curr - I_prev;

                sum_Ix2 += Ix * Ix;
                sum_Iy2 += Iy * Iy;
                sum_IxIy += Ix * Iy;
                sum_IxIt += Ix * It;
                sum_IyIt += Iy * It;
            }
        }

        float det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;
        if (fabsf(det) < 1e-6f) {
            status[idx] = 0;
            return;
        }

        float du = (sum_IxIy * sum_IyIt - sum_Iy2 * sum_IxIt) / det;
        float dv = (sum_IxIy * sum_IxIt - sum_Ix2 * sum_IyIt) / det;

        u += du;
        v += dv;

        if (fabsf(du) < epsilon && fabsf(dv) < epsilon) break;
    }

    curr_points[idx * 2] = px + u;
    curr_points[idx * 2 + 1] = py + v;
    status[idx] = 1;
}

void trackFeaturePoints(
    const float* d_prev_frame,
    const float* d_curr_frame,
    const float* d_prev_points,
    float* d_curr_points,
    unsigned char* d_status,
    int num_points,
    int width,
    int height,
    int window_size
) {
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    trackFeaturesKernel<<<grid_size, block_size>>>(
        d_prev_frame, d_curr_frame,
        d_prev_points, d_curr_points,
        d_status, num_points,
        width, height,
        window_size, 30, 0.01f
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_stabilizer
