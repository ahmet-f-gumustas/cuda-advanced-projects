#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"
#include "splatting_kernels.cuh"

// Save float3 framebuffer as PPM (Y-down image)
static void savePPM(const char* filename, const float3* h_fb, int W, int H) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", filename); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<uint8_t> row(W * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float3 c = h_fb[y * W + x];
            row[x*3+0] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, c.x)) * 255.0f);
            row[x*3+1] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, c.y)) * 255.0f);
            row[x*3+2] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, c.z)) * 255.0f);
        }
        fwrite(row.data(), 1, W * 3, f);
    }
    fclose(f);
    printf("Saved %s (%dx%d)\n", filename, W, H);
}

int main() {
    printf("=== 3D Gaussian Splatting — Phase 2 End-to-End Render ===\n\n");
    printDeviceInfo();

    // --- Setup camera ---
    Camera camera;
    CameraConfig cam_cfg;
    cam_cfg.width = 800;
    cam_cfg.height = 600;
    cam_cfg.fov_y = 60.0f;
    camera.setConfig(cam_cfg);
    camera.printInfo();

    int W = cam_cfg.width;
    int H = cam_cfg.height;
    int grid_w = (W + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (H + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles = grid_w * grid_h;
    printf("Tile grid: %d x %d = %d tiles\n\n", grid_w, grid_h, num_tiles);

    // --- Generate scene ---
    GaussianModel model;
    int N = 50000;
    float3 bmin = {-3.0f, -3.0f, -3.0f};
    float3 bmax = { 3.0f,  3.0f,  3.0f};
    model.generateRandom(N, 3, bmin, bmax);

    // --- Upload view matrix (raster convention) to device ---
    Mat4 view_raster = camera.getRasterViewMatrix();
    float* d_viewmatrix = cudaMallocDevice<float>(16);
    cudaMemcpyH2D(d_viewmatrix, view_raster.m, 16);

    // --- Allocate per-Gaussian output buffers ---
    float2* d_means2D        = cudaMallocDevice<float2>(N);
    float4* d_conic_opacity  = cudaMallocDevice<float4>(N);
    float3* d_colors         = cudaMallocDevice<float3>(N);
    float*  d_depths         = cudaMallocDevice<float>(N);
    int*    d_radii          = cudaMallocDevice<int>(N);
    int*    d_tiles_touched  = cudaMallocDevice<int>(N);
    uint32_t* d_offsets      = cudaMallocDevice<uint32_t>(N);

    // --- Framebuffer ---
    float3* d_framebuffer = cudaMallocDevice<float3>((size_t)W * H);
    std::vector<float3> h_framebuffer(W * H);

    CudaTimer timer;

    // ========================================================
    // Step 1: Preprocess
    // ========================================================
    timer.start();
    launchPreprocess(
        N,
        model.getData().d_positions,
        model.getData().d_scales,
        model.getData().d_rotations,
        model.getData().d_sh_coeffs,
        model.getData().d_opacities,
        d_viewmatrix,
        camera.getPosition(),
        camera.getFocalX(), camera.getFocalY(),
        camera.getTanFovX(), camera.getTanFovY(),
        W, H, grid_w, grid_h,
        model.getSHDegree(), model.getNumSHCoeffs(),
        d_means2D, d_conic_opacity, d_colors, d_depths, d_radii, d_tiles_touched
    );
    timer.stop();
    float t_preprocess = timer.elapsed_ms();

    // ========================================================
    // Step 2: Inclusive prefix sum → offsets
    // ========================================================
    timer.start();
    launchInclusiveSum(d_tiles_touched, d_offsets, N);
    timer.stop();
    float t_scan = timer.elapsed_ms();

    // Read total number of (gaussian, tile) pairs
    uint32_t num_rendered = 0;
    CUDA_CHECK(cudaMemcpy(&num_rendered, d_offsets + (N - 1),
                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("Preprocess: %.3f ms\n", t_preprocess);
    printf("Prefix sum: %.3f ms  (total rendered pairs: %u)\n", t_scan, num_rendered);

    if (num_rendered == 0) {
        fprintf(stderr, "No Gaussians pass frustum test!\n");
        return 1;
    }

    // ========================================================
    // Step 3: Duplicate with keys
    // ========================================================
    uint64_t* d_keys_unsorted   = cudaMallocDevice<uint64_t>(num_rendered);
    uint32_t* d_values_unsorted = cudaMallocDevice<uint32_t>(num_rendered);
    uint64_t* d_keys_sorted     = cudaMallocDevice<uint64_t>(num_rendered);
    uint32_t* d_values_sorted   = cudaMallocDevice<uint32_t>(num_rendered);

    timer.start();
    launchDuplicateWithKeys(
        N, d_means2D, d_depths, d_radii, d_offsets,
        grid_w, grid_h, d_keys_unsorted, d_values_unsorted
    );
    timer.stop();
    float t_dup = timer.elapsed_ms();
    printf("Duplicate-with-keys: %.3f ms\n", t_dup);

    // ========================================================
    // Step 4: Radix sort
    // ========================================================
    timer.start();
    launchRadixSortPairs(
        d_keys_unsorted, d_keys_sorted,
        d_values_unsorted, d_values_sorted,
        (int)num_rendered
    );
    timer.stop();
    float t_sort = timer.elapsed_ms();
    printf("Radix sort: %.3f ms\n", t_sort);

    // ========================================================
    // Step 5: Tile ranges
    // ========================================================
    uint2* d_tile_ranges = cudaMallocDevice<uint2>(num_tiles);
    timer.start();
    launchGetTileRanges((int)num_rendered, d_keys_sorted, num_tiles, d_tile_ranges);
    timer.stop();
    float t_ranges = timer.elapsed_ms();
    printf("Tile ranges: %.3f ms\n", t_ranges);

    // ========================================================
    // Step 6: Rasterize
    // ========================================================
    float3 bg_color = make_float3(0.05f, 0.05f, 0.08f);
    timer.start();
    launchRasterize(
        d_tile_ranges, d_values_sorted,
        d_means2D, d_conic_opacity, d_colors,
        W, H, grid_w, grid_h,
        bg_color, d_framebuffer
    );
    timer.stop();
    float t_raster = timer.elapsed_ms();
    printf("Rasterize: %.3f ms\n\n", t_raster);

    float t_total = t_preprocess + t_scan + t_dup + t_sort + t_ranges + t_raster;
    printf("TOTAL: %.3f ms  (%.1f FPS)\n\n", t_total, 1000.0f / t_total);

    // --- Download framebuffer and save ---
    cudaMemcpyD2H(h_framebuffer.data(), d_framebuffer, (size_t)W * H);
    savePPM("render.ppm", h_framebuffer.data(), W, H);

    // --- Simple sanity check on output ---
    float avg_r = 0, avg_g = 0, avg_b = 0;
    for (int i = 0; i < W * H; i++) {
        avg_r += h_framebuffer[i].x;
        avg_g += h_framebuffer[i].y;
        avg_b += h_framebuffer[i].z;
    }
    avg_r /= (W * H); avg_g /= (W * H); avg_b /= (W * H);
    printf("\nFramebuffer mean RGB: (%.3f, %.3f, %.3f)\n", avg_r, avg_g, avg_b);

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_viewmatrix));
    CUDA_CHECK(cudaFree(d_means2D));
    CUDA_CHECK(cudaFree(d_conic_opacity));
    CUDA_CHECK(cudaFree(d_colors));
    CUDA_CHECK(cudaFree(d_depths));
    CUDA_CHECK(cudaFree(d_radii));
    CUDA_CHECK(cudaFree(d_tiles_touched));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_keys_unsorted));
    CUDA_CHECK(cudaFree(d_values_unsorted));
    CUDA_CHECK(cudaFree(d_keys_sorted));
    CUDA_CHECK(cudaFree(d_values_sorted));
    CUDA_CHECK(cudaFree(d_tile_ranges));
    CUDA_CHECK(cudaFree(d_framebuffer));

    printf("\n=== Phase 2 Complete ===\n");
    return 0;
}
