#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"
#include "splatting_kernels.cuh"

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("TEST: %s ... ", name)
#define PASS() do { printf("PASSED\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAILED: %s\n", msg); tests_failed++; } while(0)

// ============================================================
// Phase 1 tests
// ============================================================

void test_gaussian_allocation() {
    TEST("Gaussian allocation and free");
    GaussianModel model;
    float3 bmin = {-1, -1, -1}, bmax = {1, 1, 1};
    model.generateRandom(100, 2, bmin, bmax);
    if (model.getCount() == 100 && model.getSHDegree() == 2 &&
        model.getNumSHCoeffs() == 9 && model.isAllocated()) {
        model.free();
        if (!model.isAllocated()) { PASS(); return; }
    }
    FAIL("unexpected state");
}

void test_camera_matrices() {
    TEST("Camera view/projection matrices");
    Camera cam;
    CameraConfig cfg;
    cfg.width = 800; cfg.height = 600;
    cam.setConfig(cfg);
    Mat4 view = cam.getViewMatrix();
    Mat4 proj = cam.getProjectionMatrix();
    bool view_ok = (view.at(0,0) != 0 && view.at(1,1) != 0 && view.at(2,2) != 0);
    float aspect = 800.0f / 600.0f;
    bool proj_ok = (fabsf(proj.at(0,0) / proj.at(1,1) - 1.0f / aspect) < 0.01f);
    if (view_ok && proj_ok) { PASS(); } else { FAIL("matrix values incorrect"); }
}

void test_camera_focal() {
    TEST("Camera focal length");
    Camera cam;
    CameraConfig cfg;
    cfg.width = 1920; cfg.height = 1080; cfg.fov_y = 60.0f;
    cam.setConfig(cfg);
    float fy = cam.getFocalY();
    float expected_fy = 1080.0f / (2.0f * tanf(30.0f * 3.14159265f / 180.0f));
    if (fabsf(fy - expected_fy) < 0.1f) { PASS(); } else { FAIL("focal length mismatch"); }
}

void test_camera_orbit() {
    TEST("Camera orbit control");
    Camera cam;
    float3 pos_before = cam.getPosition();
    cam.orbit(90.0f, 0.0f);
    float3 pos_after = cam.getPosition();
    bool changed = (fabsf(pos_before.x - pos_after.x) > 0.01f ||
                    fabsf(pos_before.z - pos_after.z) > 0.01f);
    if (changed) { PASS(); } else { FAIL("position did not change"); }
}

// ============================================================
// Phase 2: kernel tests
// ============================================================

void test_raster_view_matrix() {
    TEST("Raster view matrix (point in front → z>0)");
    Camera cam;                   // default: pos (0,0,5), target (0,0,0)
    CameraConfig cfg;
    cfg.width = 800; cfg.height = 600;
    cam.setConfig(cfg);
    Mat4 view = cam.getRasterViewMatrix();

    // Transform origin (0,0,0) to view space: should have z > 0
    float x = view.at(0,0)*0 + view.at(0,1)*0 + view.at(0,2)*0 + view.at(0,3);
    float y = view.at(1,0)*0 + view.at(1,1)*0 + view.at(1,2)*0 + view.at(1,3);
    float z = view.at(2,0)*0 + view.at(2,1)*0 + view.at(2,2)*0 + view.at(2,3);

    if (z > 0 && fabsf(x) < 0.01f && fabsf(y) < 0.01f) {
        PASS();
    } else {
        printf("got (%.3f, %.3f, %.3f) ", x, y, z);
        FAIL("origin should have z>0 and x,y=0");
    }
}

void test_preprocess_kernel() {
    TEST("Preprocess kernel (small scene)");
    Camera cam;
    CameraConfig cfg;
    cfg.width = 256; cfg.height = 256; cfg.fov_y = 60.0f;
    cam.setConfig(cfg);

    GaussianModel model;
    float3 bmin = {-1, -1, -1}, bmax = {1, 1, 1};
    model.generateRandom(100, 2, bmin, bmax, 42);

    int W = 256, H = 256;
    int grid_w = (W + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (H + TILE_SIZE - 1) / TILE_SIZE;
    int N = model.getCount();

    Mat4 view_raster = cam.getRasterViewMatrix();
    float* d_view = cudaMallocDevice<float>(16);
    cudaMemcpyH2D(d_view, view_raster.m, 16);

    float2* d_means2D = cudaMallocDevice<float2>(N);
    float4* d_conic  = cudaMallocDevice<float4>(N);
    float3* d_colors = cudaMallocDevice<float3>(N);
    float*  d_depth  = cudaMallocDevice<float>(N);
    int*    d_radii  = cudaMallocDevice<int>(N);
    int*    d_tiles  = cudaMallocDevice<int>(N);

    launchPreprocess(
        N,
        model.getData().d_positions,
        model.getData().d_scales,
        model.getData().d_rotations,
        model.getData().d_sh_coeffs,
        model.getData().d_opacities,
        d_view, cam.getPosition(),
        cam.getFocalX(), cam.getFocalY(),
        cam.getTanFovX(), cam.getTanFovY(),
        W, H, grid_w, grid_h,
        model.getSHDegree(), model.getNumSHCoeffs(),
        d_means2D, d_conic, d_colors, d_depth, d_radii, d_tiles
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check at least some Gaussians passed the frustum test
    std::vector<int> h_radii(N);
    std::vector<int> h_tiles(N);
    cudaMemcpyD2H(h_radii.data(), d_radii, N);
    cudaMemcpyD2H(h_tiles.data(), d_tiles, N);

    int visible = 0;
    int total_tiles = 0;
    for (int i = 0; i < N; i++) {
        if (h_radii[i] > 0) visible++;
        total_tiles += h_tiles[i];
    }

    cudaFree(d_view);
    cudaFree(d_means2D); cudaFree(d_conic); cudaFree(d_colors);
    cudaFree(d_depth); cudaFree(d_radii); cudaFree(d_tiles);

    if (visible > 0 && total_tiles > 0) {
        printf("(%d visible, %d tile-slots) ", visible, total_tiles);
        PASS();
    } else {
        FAIL("no visible Gaussians");
    }
}

void test_full_pipeline() {
    TEST("Full render pipeline (no crash, non-empty output)");
    Camera cam;
    CameraConfig cfg;
    cfg.width = 256; cfg.height = 256; cfg.fov_y = 60.0f;
    cam.setConfig(cfg);

    GaussianModel model;
    float3 bmin = {-2, -2, -2}, bmax = {2, 2, 2};
    model.generateRandom(500, 3, bmin, bmax, 123);

    int W = 256, H = 256;
    int grid_w = (W + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (H + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles = grid_w * grid_h;
    int N = model.getCount();

    Mat4 view = cam.getRasterViewMatrix();
    float* d_view = cudaMallocDevice<float>(16);
    cudaMemcpyH2D(d_view, view.m, 16);

    float2* d_means2D = cudaMallocDevice<float2>(N);
    float4* d_conic  = cudaMallocDevice<float4>(N);
    float3* d_colors = cudaMallocDevice<float3>(N);
    float*  d_depth  = cudaMallocDevice<float>(N);
    int*    d_radii  = cudaMallocDevice<int>(N);
    int*    d_tiles  = cudaMallocDevice<int>(N);
    uint32_t* d_offs = cudaMallocDevice<uint32_t>(N);

    launchPreprocess(N,
        model.getData().d_positions, model.getData().d_scales,
        model.getData().d_rotations, model.getData().d_sh_coeffs,
        model.getData().d_opacities, d_view, cam.getPosition(),
        cam.getFocalX(), cam.getFocalY(),
        cam.getTanFovX(), cam.getTanFovY(),
        W, H, grid_w, grid_h,
        model.getSHDegree(), model.getNumSHCoeffs(),
        d_means2D, d_conic, d_colors, d_depth, d_radii, d_tiles);

    launchInclusiveSum(d_tiles, d_offs, N);
    uint32_t num_rendered = 0;
    cudaMemcpy(&num_rendered, d_offs + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (num_rendered == 0) { FAIL("no render pairs"); return; }

    uint64_t* d_keys_u = cudaMallocDevice<uint64_t>(num_rendered);
    uint32_t* d_vals_u = cudaMallocDevice<uint32_t>(num_rendered);
    uint64_t* d_keys_s = cudaMallocDevice<uint64_t>(num_rendered);
    uint32_t* d_vals_s = cudaMallocDevice<uint32_t>(num_rendered);

    launchDuplicateWithKeys(N, d_means2D, d_depth, d_radii, d_offs,
                             grid_w, grid_h, d_keys_u, d_vals_u);
    launchRadixSortPairs(d_keys_u, d_keys_s, d_vals_u, d_vals_s, (int)num_rendered);

    uint2* d_tile_ranges = cudaMallocDevice<uint2>(num_tiles);
    launchGetTileRanges((int)num_rendered, d_keys_s, num_tiles, d_tile_ranges);

    float3* d_fb = cudaMallocDevice<float3>((size_t)W * H);
    float3 bg = make_float3(0.0f, 0.0f, 0.0f);
    launchRasterize(d_tile_ranges, d_vals_s, d_means2D, d_conic, d_colors,
                    W, H, grid_w, grid_h, bg, d_fb);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float3> h_fb(W * H);
    cudaMemcpyD2H(h_fb.data(), d_fb, (size_t)W * H);

    float max_val = 0;
    int nonzero = 0;
    for (int i = 0; i < W * H; i++) {
        float v = fmaxf(fmaxf(h_fb[i].x, h_fb[i].y), h_fb[i].z);
        if (v > 0.001f) nonzero++;
        max_val = fmaxf(max_val, v);
    }

    cudaFree(d_view); cudaFree(d_means2D); cudaFree(d_conic); cudaFree(d_colors);
    cudaFree(d_depth); cudaFree(d_radii); cudaFree(d_tiles); cudaFree(d_offs);
    cudaFree(d_keys_u); cudaFree(d_vals_u); cudaFree(d_keys_s); cudaFree(d_vals_s);
    cudaFree(d_tile_ranges); cudaFree(d_fb);

    if (nonzero > 100 && max_val > 0.0f) {
        printf("(%d non-bg pixels, max=%.3f) ", nonzero, max_val);
        PASS();
    } else {
        FAIL("framebuffer mostly empty");
    }
}

void test_sort_ordering() {
    TEST("Radix sort preserves (tile, depth) ordering");
    const int N = 10;
    std::vector<uint64_t> h_keys(N);
    std::vector<uint32_t> h_vals(N);
    // Unsorted keys with 3 tiles, varying depths
    uint64_t raw[10] = {
        ((uint64_t)2 << 32) | 0x30000000,
        ((uint64_t)0 << 32) | 0x10000000,
        ((uint64_t)1 << 32) | 0x20000000,
        ((uint64_t)0 << 32) | 0x05000000,
        ((uint64_t)2 << 32) | 0x10000000,
        ((uint64_t)1 << 32) | 0x15000000,
        ((uint64_t)0 << 32) | 0x20000000,
        ((uint64_t)2 << 32) | 0x20000000,
        ((uint64_t)1 << 32) | 0x10000000,
        ((uint64_t)1 << 32) | 0x05000000,
    };
    for (int i = 0; i < N; i++) { h_keys[i] = raw[i]; h_vals[i] = i; }

    uint64_t* d_keys_in  = cudaMallocDevice<uint64_t>(N);
    uint64_t* d_keys_out = cudaMallocDevice<uint64_t>(N);
    uint32_t* d_vals_in  = cudaMallocDevice<uint32_t>(N);
    uint32_t* d_vals_out = cudaMallocDevice<uint32_t>(N);
    cudaMemcpyH2D(d_keys_in, h_keys.data(), N);
    cudaMemcpyH2D(d_vals_in, h_vals.data(), N);

    launchRadixSortPairs(d_keys_in, d_keys_out, d_vals_in, d_vals_out, N);
    std::vector<uint64_t> sorted(N);
    cudaMemcpyD2H(sorted.data(), d_keys_out, N);

    bool ordered = true;
    for (int i = 1; i < N; i++) {
        if (sorted[i] < sorted[i-1]) { ordered = false; break; }
    }

    cudaFree(d_keys_in); cudaFree(d_keys_out);
    cudaFree(d_vals_in); cudaFree(d_vals_out);

    if (ordered) { PASS(); } else { FAIL("keys not sorted"); }
}

// ============================================================
int main() {
    printf("=== 3D Gaussian Splatting — Tests (Phases 1+2) ===\n\n");
    printDeviceInfo();

    test_gaussian_allocation();
    test_camera_matrices();
    test_camera_focal();
    test_camera_orbit();
    test_raster_view_matrix();
    test_preprocess_kernel();
    test_sort_ordering();
    test_full_pipeline();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
