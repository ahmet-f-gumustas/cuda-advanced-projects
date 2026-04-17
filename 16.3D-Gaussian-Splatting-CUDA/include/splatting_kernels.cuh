#ifndef SPLATTING_KERNELS_CUH
#define SPLATTING_KERNELS_CUH

#include "cuda_utils.h"
#include <cstdint>

// ============================================================
// Constants
// ============================================================

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = TILE_SIZE * TILE_SIZE; // 256 threads per tile

// ============================================================
// Launch wrappers
// ============================================================

// Per-Gaussian preprocessing:
// - Transform to view space (raster convention: z > 0 for visible)
// - Compute 3D covariance from scale+rotation
// - Project to 2D covariance via EWA splatting
// - Invert 2D covariance to get conic
// - Evaluate spherical harmonics for RGB
// - Compute screen-space radius and tiles touched
void launchPreprocess(
    int num_gaussians,
    const float* d_positions,       // (N x 3)
    const float* d_scales,          // (N x 3)  log-scale
    const float* d_rotations,       // (N x 4)  quaternion (w,x,y,z)
    const float* d_sh_coeffs,       // (N x C x 3)
    const float* d_opacities,       // (N)      pre-sigmoid
    const float* d_viewmatrix,      // 16 floats, column-major (raster convention)
    float3 cam_pos,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    int image_width, int image_height,
    int grid_width, int grid_height,
    int sh_degree,
    int num_sh_coeffs,
    // outputs:
    float2* d_means2D,              // (N)
    float4* d_conic_opacity,        // (N)  (inv_cov_xx, inv_cov_xy, inv_cov_yy, opacity)
    float3* d_colors,               // (N)
    float* d_depths,                // (N)  view-space z (positive)
    int* d_radii,                   // (N)
    int* d_tiles_touched,           // (N)
    cudaStream_t stream = 0
);

// Inclusive scan on tiles_touched → offsets (using CUB)
void launchInclusiveSum(
    const int* d_in,
    uint32_t* d_out,
    int N,
    cudaStream_t stream = 0
);

// Emit (tile_id, depth) keys and gaussian index values for each touched tile
void launchDuplicateWithKeys(
    int num_gaussians,
    const float2* d_means2D,
    const float* d_depths,
    const int* d_radii,
    const uint32_t* d_offsets,
    int grid_width, int grid_height,
    // outputs:
    uint64_t* d_keys,
    uint32_t* d_values,
    cudaStream_t stream = 0
);

// Sort (key, value) pairs by key (using CUB radix sort)
void launchRadixSortPairs(
    uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    uint32_t* d_values_in,
    uint32_t* d_values_out,
    int N,
    cudaStream_t stream = 0
);

// Identify [start, end) ranges of gaussians per tile in sorted key array
void launchGetTileRanges(
    int num_rendered,
    const uint64_t* d_keys_sorted,
    int num_tiles,
    uint2* d_tile_ranges,
    cudaStream_t stream = 0
);

// Per-tile rasterization with front-to-back alpha compositing
void launchRasterize(
    const uint2* d_tile_ranges,
    const uint32_t* d_gaussian_list,  // sorted values
    const float2* d_means2D,
    const float4* d_conic_opacity,
    const float3* d_colors,
    int image_width, int image_height,
    int grid_width, int grid_height,
    float3 bg_color,
    float3* d_framebuffer,             // (H x W)
    cudaStream_t stream = 0
);

#endif // SPLATTING_KERNELS_CUH
