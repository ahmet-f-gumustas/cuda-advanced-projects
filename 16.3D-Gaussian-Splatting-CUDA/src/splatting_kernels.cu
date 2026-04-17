#include "splatting_kernels.cuh"
#include <cub/cub.cuh>

// ============================================================
// Device math helpers
// ============================================================

// Transform 3D point by 4x4 column-major matrix
__device__ inline float3 transformPoint(const float* M, float3 p) {
    return make_float3(
        M[0] * p.x + M[4] * p.y + M[8]  * p.z + M[12],
        M[1] * p.x + M[5] * p.y + M[9]  * p.z + M[13],
        M[2] * p.x + M[6] * p.y + M[10] * p.z + M[14]
    );
}

// Convert quaternion (w, x, y, z) to 3x3 rotation matrix (row-major)
__device__ inline void quatToMat3(const float* q, float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1.0f - 2.0f * (y*y + z*z);  R[1] = 2.0f * (x*y - w*z);         R[2] = 2.0f * (x*z + w*y);
    R[3] = 2.0f * (x*y + w*z);         R[4] = 1.0f - 2.0f * (x*x + z*z);  R[5] = 2.0f * (y*z - w*x);
    R[6] = 2.0f * (x*z - w*y);         R[7] = 2.0f * (y*z + w*x);         R[8] = 1.0f - 2.0f * (x*x + y*y);
}

// Compute 3D covariance from log-scale and rotation quaternion.
// cov3D packed as 6 floats: [xx, xy, xz, yy, yz, zz]
__device__ inline void computeCov3D(const float* scales_log, const float* rot_q, float cov3D[6]) {
    float sx = expf(scales_log[0]);
    float sy = expf(scales_log[1]);
    float sz = expf(scales_log[2]);

    float R[9];
    quatToMat3(rot_q, R);

    // M = R * diag(s)
    float M[9];
    M[0] = R[0] * sx;  M[1] = R[1] * sy;  M[2] = R[2] * sz;
    M[3] = R[3] * sx;  M[4] = R[4] * sy;  M[5] = R[5] * sz;
    M[6] = R[6] * sx;  M[7] = R[7] * sy;  M[8] = R[8] * sz;

    // Cov3D = M * M^T
    cov3D[0] = M[0]*M[0] + M[1]*M[1] + M[2]*M[2];   // xx
    cov3D[1] = M[0]*M[3] + M[1]*M[4] + M[2]*M[5];   // xy
    cov3D[2] = M[0]*M[6] + M[1]*M[7] + M[2]*M[8];   // xz
    cov3D[3] = M[3]*M[3] + M[4]*M[4] + M[5]*M[5];   // yy
    cov3D[4] = M[3]*M[6] + M[4]*M[7] + M[5]*M[8];   // yz
    cov3D[5] = M[6]*M[6] + M[7]*M[7] + M[8]*M[8];   // zz
}

// EWA splatting: project 3D covariance to 2D screen covariance.
// Returns (xx, xy, yy) of the 2x2 covariance with low-pass filter applied.
__device__ inline float3 computeCov2D(
    float3 t,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    const float cov3D[6],
    const float* viewmatrix
) {
    // Clamp view-space x/y to avoid extreme values near frustum edge
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    float tx = fminf(limx, fmaxf(-limx, txtz)) * t.z;
    float ty = fminf(limy, fmaxf(-limy, tytz)) * t.z;

    // Jacobian J (2x3) of perspective projection at (tx, ty, t.z)
    float J00 = focal_x / t.z;
    float J02 = -(focal_x * tx) / (t.z * t.z);
    float J11 = focal_y / t.z;
    float J12 = -(focal_y * ty) / (t.z * t.z);

    // W = view matrix upper-left 3x3 (column-major storage: W[row + col*3])
    // W_ij = viewmatrix[col*4 + row] where col in 0..2
    float W00 = viewmatrix[0], W01 = viewmatrix[4], W02 = viewmatrix[8];
    float W10 = viewmatrix[1], W11 = viewmatrix[5], W12 = viewmatrix[9];
    float W20 = viewmatrix[2], W21 = viewmatrix[6], W22 = viewmatrix[10];

    // T = J * W (2x3 * 3x3 = 2x3)
    float T00 = J00 * W00 + J02 * W20;
    float T01 = J00 * W01 + J02 * W21;
    float T02 = J00 * W02 + J02 * W22;
    float T10 = J11 * W10 + J12 * W20;
    float T11 = J11 * W11 + J12 * W21;
    float T12 = J11 * W12 + J12 * W22;

    // Cov2D = T * Cov3D * T^T (2x2)
    // Cov3D full matrix: [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    float cxx = cov3D[0], cxy = cov3D[1], cxz = cov3D[2];
    float cyy = cov3D[3], cyz = cov3D[4], czz = cov3D[5];

    // M = T * Cov3D (2x3)
    float M00 = T00*cxx + T01*cxy + T02*cxz;
    float M01 = T00*cxy + T01*cyy + T02*cyz;
    float M02 = T00*cxz + T01*cyz + T02*czz;
    float M10 = T10*cxx + T11*cxy + T12*cxz;
    float M11 = T10*cxy + T11*cyy + T12*cyz;
    float M12 = T10*cxz + T11*cyz + T12*czz;

    // Cov2D = M * T^T (2x2)
    float cov_xx = M00*T00 + M01*T01 + M02*T02;
    float cov_xy = M00*T10 + M01*T11 + M02*T12;
    float cov_yy = M10*T10 + M11*T11 + M12*T12;

    // Low-pass filter (prevents degeneration at sub-pixel Gaussians)
    cov_xx += 0.3f;
    cov_yy += 0.3f;

    return make_float3(cov_xx, cov_xy, cov_yy);
}

// Evaluate spherical harmonics at given direction, return RGB in [0, 1].
__device__ inline float3 evaluateSH(int sh_degree, float3 dir, const float* sh) {
    // Degree 0 (DC term)
    float3 rgb = make_float3(
        0.28209479177387814f * sh[0],
        0.28209479177387814f * sh[1],
        0.28209479177387814f * sh[2]
    );

    if (sh_degree >= 1) {
        float x = dir.x, y = dir.y, z = dir.z;
        float c;
        c = -0.48860251190291987f * y;
        rgb.x += c * sh[3];  rgb.y += c * sh[4];  rgb.z += c * sh[5];
        c =  0.48860251190291987f * z;
        rgb.x += c * sh[6];  rgb.y += c * sh[7];  rgb.z += c * sh[8];
        c = -0.48860251190291987f * x;
        rgb.x += c * sh[9];  rgb.y += c * sh[10]; rgb.z += c * sh[11];

        if (sh_degree >= 2) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, xz = x*z, yz = y*z;
            c =  1.0925484305920792f * xy;
            rgb.x += c * sh[12]; rgb.y += c * sh[13]; rgb.z += c * sh[14];
            c = -1.0925484305920792f * yz;
            rgb.x += c * sh[15]; rgb.y += c * sh[16]; rgb.z += c * sh[17];
            c =  0.31539156525252005f * (2.0f*zz - xx - yy);
            rgb.x += c * sh[18]; rgb.y += c * sh[19]; rgb.z += c * sh[20];
            c = -1.0925484305920792f * xz;
            rgb.x += c * sh[21]; rgb.y += c * sh[22]; rgb.z += c * sh[23];
            c =  0.5462742152960396f * (xx - yy);
            rgb.x += c * sh[24]; rgb.y += c * sh[25]; rgb.z += c * sh[26];

            if (sh_degree >= 3) {
                c = -0.5900435899266435f * y * (3.0f*xx - yy);
                rgb.x += c * sh[27]; rgb.y += c * sh[28]; rgb.z += c * sh[29];
                c =  2.890611442640554f * xy * z;
                rgb.x += c * sh[30]; rgb.y += c * sh[31]; rgb.z += c * sh[32];
                c = -0.4570457994644658f * y * (4.0f*zz - xx - yy);
                rgb.x += c * sh[33]; rgb.y += c * sh[34]; rgb.z += c * sh[35];
                c =  0.3731763325901154f * z * (2.0f*zz - 3.0f*xx - 3.0f*yy);
                rgb.x += c * sh[36]; rgb.y += c * sh[37]; rgb.z += c * sh[38];
                c = -0.4570457994644658f * x * (4.0f*zz - xx - yy);
                rgb.x += c * sh[39]; rgb.y += c * sh[40]; rgb.z += c * sh[41];
                c =  1.445305721320277f * z * (xx - yy);
                rgb.x += c * sh[42]; rgb.y += c * sh[43]; rgb.z += c * sh[44];
                c = -0.5900435899266435f * x * (xx - 3.0f*yy);
                rgb.x += c * sh[45]; rgb.y += c * sh[46]; rgb.z += c * sh[47];
            }
        }
    }

    rgb.x = fmaxf(0.0f, rgb.x + 0.5f);
    rgb.y = fmaxf(0.0f, rgb.y + 0.5f);
    rgb.z = fmaxf(0.0f, rgb.z + 0.5f);
    return rgb;
}

__device__ inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Largest-eigenvalue-based 3σ radius in pixels
__device__ inline int computeScreenRadius(float3 cov2D) {
    float mid = 0.5f * (cov2D.x + cov2D.z);
    float disc = sqrtf(fmaxf(0.1f, mid * mid - (cov2D.x * cov2D.z - cov2D.y * cov2D.y)));
    float lambda_max = fmaxf(mid + disc, mid - disc);
    return (int)ceilf(3.0f * sqrtf(fmaxf(lambda_max, 0.0f)));
}

// ============================================================
// Preprocess kernel
// ============================================================

__global__ void preprocess_kernel(
    int N,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    const float* __restrict__ sh_coeffs,
    const float* __restrict__ opacities,
    const float* __restrict__ viewmatrix,
    float3 cam_pos,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    int W, int H,
    int grid_w, int grid_h,
    int sh_degree,
    int num_sh_coeffs,
    float2* __restrict__ means2D,
    float4* __restrict__ conic_opacity,
    float3* __restrict__ colors,
    float* __restrict__ depths,
    int* __restrict__ radii,
    int* __restrict__ tiles_touched
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Default outputs
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Load position, transform to view space
    float3 p_world = make_float3(positions[idx*3+0], positions[idx*3+1], positions[idx*3+2]);
    float3 p_view = transformPoint(viewmatrix, p_world);

    // Frustum cull (near-plane)
    if (p_view.z < 0.2f) return;

    // Compute 3D covariance
    float cov3D[6];
    computeCov3D(&scales[idx*3], &rotations[idx*4], cov3D);

    // Project to 2D covariance
    float3 cov2D = computeCov2D(p_view, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // Invert 2D covariance (conic form for efficient Mahalanobis distance)
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0f) return;
    float inv_det = 1.0f / det;
    float3 conic = make_float3( cov2D.z * inv_det, -cov2D.y * inv_det, cov2D.x * inv_det );

    // Project to screen pixel coordinates
    float px = focal_x * p_view.x / p_view.z + (float)W * 0.5f;
    float py = focal_y * p_view.y / p_view.z + (float)H * 0.5f;

    // Screen radius (3 sigma)
    int radius = computeScreenRadius(cov2D);
    if (radius == 0) return;

    // Tile bbox
    int tile_min_x = max(0, min(grid_w, (int)((px - radius) / TILE_SIZE)));
    int tile_min_y = max(0, min(grid_h, (int)((py - radius) / TILE_SIZE)));
    int tile_max_x = max(0, min(grid_w, (int)((px + radius + TILE_SIZE - 1) / TILE_SIZE)));
    int tile_max_y = max(0, min(grid_h, (int)((py + radius + TILE_SIZE - 1) / TILE_SIZE)));
    int num_tiles = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y);
    if (num_tiles == 0) return;

    // Evaluate SH color (view direction = from camera to Gaussian)
    float3 dir = normalize(p_world - cam_pos);
    const float* sh_ptr = sh_coeffs + (size_t)idx * num_sh_coeffs * 3;
    float3 color = evaluateSH(sh_degree, dir, sh_ptr);

    // Opacity
    float opa = sigmoid_f(opacities[idx]);

    // Commit outputs
    means2D[idx] = make_float2(px, py);
    conic_opacity[idx] = make_float4(conic.x, conic.y, conic.z, opa);
    colors[idx] = color;
    depths[idx] = p_view.z;
    radii[idx] = radius;
    tiles_touched[idx] = num_tiles;
}

// ============================================================
// Duplicate-with-keys kernel
// ============================================================

__global__ void duplicate_with_keys_kernel(
    int N,
    const float2* __restrict__ means2D,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const uint32_t* __restrict__ offsets,
    int grid_w, int grid_h,
    uint64_t* __restrict__ keys_out,
    uint32_t* __restrict__ values_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int radius = radii[idx];
    if (radius == 0) return;

    uint32_t off = (idx == 0) ? 0u : offsets[idx - 1];
    float2 mean = means2D[idx];

    int tile_min_x = max(0, min(grid_w, (int)((mean.x - radius) / TILE_SIZE)));
    int tile_min_y = max(0, min(grid_h, (int)((mean.y - radius) / TILE_SIZE)));
    int tile_max_x = max(0, min(grid_w, (int)((mean.x + radius + TILE_SIZE - 1) / TILE_SIZE)));
    int tile_max_y = max(0, min(grid_h, (int)((mean.y + radius + TILE_SIZE - 1) / TILE_SIZE)));

    uint32_t depth_bits = __float_as_uint(depths[idx]);

    for (int ty = tile_min_y; ty < tile_max_y; ty++) {
        for (int tx = tile_min_x; tx < tile_max_x; tx++) {
            uint32_t tile_id = (uint32_t)(ty * grid_w + tx);
            uint64_t key = ((uint64_t)tile_id << 32) | (uint64_t)depth_bits;
            keys_out[off] = key;
            values_out[off] = (uint32_t)idx;
            off++;
        }
    }
}

// ============================================================
// Tile range identification kernel
// ============================================================

__global__ void get_tile_ranges_kernel(
    int num_rendered,
    const uint64_t* __restrict__ keys_sorted,
    uint2* __restrict__ tile_ranges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rendered) return;

    uint32_t curr_tile = (uint32_t)(keys_sorted[idx] >> 32);

    if (idx == 0) {
        tile_ranges[curr_tile].x = 0;
    } else {
        uint32_t prev_tile = (uint32_t)(keys_sorted[idx - 1] >> 32);
        if (prev_tile != curr_tile) {
            tile_ranges[prev_tile].y = (uint32_t)idx;
            tile_ranges[curr_tile].x = (uint32_t)idx;
        }
    }

    if (idx == num_rendered - 1) {
        tile_ranges[curr_tile].y = (uint32_t)num_rendered;
    }
}

// ============================================================
// Rasterize kernel
// ============================================================

__global__ void __launch_bounds__(BLOCK_SIZE)
rasterize_kernel(
    const uint2* __restrict__ tile_ranges,
    const uint32_t* __restrict__ gaussian_list,
    const float2* __restrict__ means2D,
    const float4* __restrict__ conic_opacity,
    const float3* __restrict__ colors,
    int W, int H,
    int grid_w,
    float3 bg_color,
    float3* __restrict__ framebuffer
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_id = tile_y * grid_w + tile_x;

    int pix_x = tile_x * TILE_SIZE + threadIdx.x;
    int pix_y = tile_y * TILE_SIZE + threadIdx.y;
    bool inside = (pix_x < W) && (pix_y < H);
    int thread_id = threadIdx.y * TILE_SIZE + threadIdx.x;

    uint2 range = tile_ranges[tile_id];
    int num_gauss = (int)range.y - (int)range.x;
    int num_batches = (num_gauss + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ float2 s_means[BLOCK_SIZE];
    __shared__ float4 s_conic_opa[BLOCK_SIZE];
    __shared__ float3 s_colors[BLOCK_SIZE];

    float T = 1.0f;
    float3 C = make_float3(0.0f, 0.0f, 0.0f);
    bool done = !inside;

    for (int b = 0; b < num_batches; b++) {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE) break;

        // Collaborative load of one batch
        int load_idx = (int)range.x + b * BLOCK_SIZE + thread_id;
        if (load_idx < (int)range.y) {
            uint32_t gi = gaussian_list[load_idx];
            s_means[thread_id]    = means2D[gi];
            s_conic_opa[thread_id] = conic_opacity[gi];
            s_colors[thread_id]    = colors[gi];
        }
        __syncthreads();

        int batch_size = min(BLOCK_SIZE, num_gauss - b * BLOCK_SIZE);
        for (int j = 0; j < batch_size && !done; j++) {
            float2 xy = s_means[j];
            float4 co = s_conic_opa[j];
            float dx = xy.x - (float)pix_x;
            float dy = xy.y - (float)pix_y;

            // power = -0.5 * (conic_xx*dx² + conic_yy*dy²) - conic_xy*dx*dy
            float power = -0.5f * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
            if (power > 0.0f) continue;

            float alpha = fminf(0.99f, co.w * expf(power));
            if (alpha < (1.0f / 255.0f)) continue;

            float new_T = T * (1.0f - alpha);
            if (new_T < 1e-4f) { done = true; break; }

            float3 col = s_colors[j];
            C.x += T * alpha * col.x;
            C.y += T * alpha * col.y;
            C.z += T * alpha * col.z;
            T = new_T;
        }
        __syncthreads();
    }

    if (inside) {
        int pix_idx = pix_y * W + pix_x;
        framebuffer[pix_idx] = make_float3(
            C.x + T * bg_color.x,
            C.y + T * bg_color.y,
            C.z + T * bg_color.z
        );
    }
}

// ============================================================
// Launch wrappers
// ============================================================

void launchPreprocess(
    int num_gaussians,
    const float* d_positions,
    const float* d_scales,
    const float* d_rotations,
    const float* d_sh_coeffs,
    const float* d_opacities,
    const float* d_viewmatrix,
    float3 cam_pos,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    int image_width, int image_height,
    int grid_width, int grid_height,
    int sh_degree,
    int num_sh_coeffs,
    float2* d_means2D,
    float4* d_conic_opacity,
    float3* d_colors,
    float* d_depths,
    int* d_radii,
    int* d_tiles_touched,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    preprocess_kernel<<<blocks, threads, 0, stream>>>(
        num_gaussians,
        d_positions, d_scales, d_rotations, d_sh_coeffs, d_opacities,
        d_viewmatrix, cam_pos, focal_x, focal_y, tan_fovx, tan_fovy,
        image_width, image_height, grid_width, grid_height,
        sh_degree, num_sh_coeffs,
        d_means2D, d_conic_opacity, d_colors, d_depths, d_radii, d_tiles_touched
    );
    CUDA_CHECK_LAST_ERROR();
}

void launchInclusiveSum(const int* d_in, uint32_t* d_out, int N, cudaStream_t stream) {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_in, d_out, N, stream);
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, N, stream);
    CUDA_CHECK(cudaFree(d_temp));
}

void launchDuplicateWithKeys(
    int num_gaussians,
    const float2* d_means2D,
    const float* d_depths,
    const int* d_radii,
    const uint32_t* d_offsets,
    int grid_width, int grid_height,
    uint64_t* d_keys,
    uint32_t* d_values,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    duplicate_with_keys_kernel<<<blocks, threads, 0, stream>>>(
        num_gaussians, d_means2D, d_depths, d_radii, d_offsets,
        grid_width, grid_height, d_keys, d_values
    );
    CUDA_CHECK_LAST_ERROR();
}

void launchRadixSortPairs(
    uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    uint32_t* d_values_in,
    uint32_t* d_values_out,
    int N,
    cudaStream_t stream
) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        N, 0, 64, stream);
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        N, 0, 64, stream);
    CUDA_CHECK(cudaFree(d_temp));
}

void launchGetTileRanges(
    int num_rendered,
    const uint64_t* d_keys_sorted,
    int num_tiles,
    uint2* d_tile_ranges,
    cudaStream_t stream
) {
    // Zero-initialize ranges (tiles with no gaussians stay {0, 0})
    CUDA_CHECK(cudaMemsetAsync(d_tile_ranges, 0, (size_t)num_tiles * sizeof(uint2), stream));

    if (num_rendered == 0) return;

    int threads = 256;
    int blocks = (num_rendered + threads - 1) / threads;
    get_tile_ranges_kernel<<<blocks, threads, 0, stream>>>(
        num_rendered, d_keys_sorted, d_tile_ranges
    );
    CUDA_CHECK_LAST_ERROR();
}

void launchRasterize(
    const uint2* d_tile_ranges,
    const uint32_t* d_gaussian_list,
    const float2* d_means2D,
    const float4* d_conic_opacity,
    const float3* d_colors,
    int image_width, int image_height,
    int grid_width, int grid_height,
    float3 bg_color,
    float3* d_framebuffer,
    cudaStream_t stream
) {
    dim3 grid(grid_width, grid_height);
    dim3 block(TILE_SIZE, TILE_SIZE);
    rasterize_kernel<<<grid, block, 0, stream>>>(
        d_tile_ranges, d_gaussian_list,
        d_means2D, d_conic_opacity, d_colors,
        image_width, image_height, grid_width,
        bg_color, d_framebuffer
    );
    CUDA_CHECK_LAST_ERROR();
}
