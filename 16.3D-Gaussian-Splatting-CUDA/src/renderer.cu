#include "renderer.h"

// ============================================================
// Lifecycle
// ============================================================

Renderer::Renderer() {
    d_viewmatrix_ = cudaMallocDevice<float>(16);
}

Renderer::~Renderer() {
    freeGaussianBuffers();
    freeRenderedBuffers();
    freeImageBuffers();
    if (d_viewmatrix_) { CUDA_CHECK(cudaFree(d_viewmatrix_)); d_viewmatrix_ = nullptr; }
    if (d_scan_temp_)  { CUDA_CHECK(cudaFree(d_scan_temp_));  d_scan_temp_  = nullptr; }
    if (d_sort_temp_)  { CUDA_CHECK(cudaFree(d_sort_temp_));  d_sort_temp_  = nullptr; }
}

void Renderer::init(int width, int height) {
    resize(width, height);
}

void Renderer::resize(int width, int height) {
    if (width == width_ && height == height_) return;
    freeImageBuffers();
    allocateImageBuffers(width, height);
}

// ============================================================
// Buffer management
// ============================================================

void Renderer::allocateImageBuffers(int width, int height) {
    width_    = width;
    height_   = height;
    grid_w_   = (width  + TILE_SIZE - 1) / TILE_SIZE;
    grid_h_   = (height + TILE_SIZE - 1) / TILE_SIZE;
    num_tiles_ = grid_w_ * grid_h_;
    d_framebuffer_ = cudaMallocDevice<float3>((size_t)width * height);
    d_tile_ranges_ = cudaMallocDevice<uint2>(num_tiles_);
}

void Renderer::freeImageBuffers() {
    if (d_framebuffer_) { CUDA_CHECK(cudaFree(d_framebuffer_)); d_framebuffer_ = nullptr; }
    if (d_tile_ranges_) { CUDA_CHECK(cudaFree(d_tile_ranges_)); d_tile_ranges_ = nullptr; }
    width_ = height_ = grid_w_ = grid_h_ = num_tiles_ = 0;
}

void Renderer::ensureGaussianBuffers(int N) {
    if (N <= capacity_gaussians_) return;
    freeGaussianBuffers();
    // Add headroom to reduce re-allocation churn
    int cap = (N * 3) / 2;
    d_means2D_       = cudaMallocDevice<float2>(cap);
    d_conic_opacity_ = cudaMallocDevice<float4>(cap);
    d_colors_        = cudaMallocDevice<float3>(cap);
    d_depths_        = cudaMallocDevice<float>(cap);
    d_radii_         = cudaMallocDevice<int>(cap);
    d_tiles_touched_ = cudaMallocDevice<int>(cap);
    d_offsets_       = cudaMallocDevice<uint32_t>(cap);
    capacity_gaussians_ = cap;
}

void Renderer::freeGaussianBuffers() {
    if (d_means2D_)       { CUDA_CHECK(cudaFree(d_means2D_));       d_means2D_ = nullptr; }
    if (d_conic_opacity_) { CUDA_CHECK(cudaFree(d_conic_opacity_)); d_conic_opacity_ = nullptr; }
    if (d_colors_)        { CUDA_CHECK(cudaFree(d_colors_));        d_colors_ = nullptr; }
    if (d_depths_)        { CUDA_CHECK(cudaFree(d_depths_));        d_depths_ = nullptr; }
    if (d_radii_)         { CUDA_CHECK(cudaFree(d_radii_));         d_radii_ = nullptr; }
    if (d_tiles_touched_) { CUDA_CHECK(cudaFree(d_tiles_touched_)); d_tiles_touched_ = nullptr; }
    if (d_offsets_)       { CUDA_CHECK(cudaFree(d_offsets_));       d_offsets_ = nullptr; }
    capacity_gaussians_ = 0;
}

void Renderer::ensureRenderedBuffers(int N) {
    if (N <= capacity_rendered_) return;
    freeRenderedBuffers();
    int cap = (N * 3) / 2;
    d_keys_unsorted_   = cudaMallocDevice<uint64_t>(cap);
    d_values_unsorted_ = cudaMallocDevice<uint32_t>(cap);
    d_keys_sorted_     = cudaMallocDevice<uint64_t>(cap);
    d_values_sorted_   = cudaMallocDevice<uint32_t>(cap);
    capacity_rendered_ = cap;
}

void Renderer::freeRenderedBuffers() {
    if (d_keys_unsorted_)   { CUDA_CHECK(cudaFree(d_keys_unsorted_));   d_keys_unsorted_ = nullptr; }
    if (d_values_unsorted_) { CUDA_CHECK(cudaFree(d_values_unsorted_)); d_values_unsorted_ = nullptr; }
    if (d_keys_sorted_)     { CUDA_CHECK(cudaFree(d_keys_sorted_));     d_keys_sorted_ = nullptr; }
    if (d_values_sorted_)   { CUDA_CHECK(cudaFree(d_values_sorted_));   d_values_sorted_ = nullptr; }
    capacity_rendered_ = 0;
}

void Renderer::ensureScanTempStorage(int N) {
    size_t required = getInclusiveSumTempBytes(N);
    if (required > scan_temp_bytes_) {
        if (d_scan_temp_) CUDA_CHECK(cudaFree(d_scan_temp_));
        CUDA_CHECK(cudaMalloc(&d_scan_temp_, required));
        scan_temp_bytes_ = required;
    }
}

void Renderer::ensureSortTempStorage(int N) {
    size_t required = getRadixSortPairsTempBytes(N);
    if (required > sort_temp_bytes_) {
        if (d_sort_temp_) CUDA_CHECK(cudaFree(d_sort_temp_));
        CUDA_CHECK(cudaMalloc(&d_sort_temp_, required));
        sort_temp_bytes_ = required;
    }
}

// ============================================================
// Main render entry point
// ============================================================

float3* Renderer::render(const GaussianModel& model, const Camera& camera) {
    int N = model.getCount();
    if (N == 0 || width_ == 0 || height_ == 0) return d_framebuffer_;

    ensureGaussianBuffers(N);
    ensureScanTempStorage(N);

    // Upload view matrix (raster convention) to device
    Mat4 view_raster = camera.getRasterViewMatrix();
    cudaMemcpyH2D(d_viewmatrix_, view_raster.m, 16);

    CudaTimer timer;

    // --- Preprocess ---
    timer.start();
    launchPreprocess(
        N,
        model.getData().d_positions,
        model.getData().d_scales,
        model.getData().d_rotations,
        model.getData().d_sh_coeffs,
        model.getData().d_opacities,
        d_viewmatrix_, camera.getPosition(),
        camera.getFocalX(), camera.getFocalY(),
        camera.getTanFovX(), camera.getTanFovY(),
        width_, height_, grid_w_, grid_h_,
        model.getSHDegree(), model.getNumSHCoeffs(),
        d_means2D_, d_conic_opacity_, d_colors_, d_depths_,
        d_radii_, d_tiles_touched_
    );
    timer.stop();
    timings_.preprocess_ms = timer.elapsed_ms();

    // --- Inclusive scan → offsets ---
    timer.start();
    launchInclusiveSumPreAlloc(
        d_scan_temp_, scan_temp_bytes_,
        d_tiles_touched_, d_offsets_, N
    );
    timer.stop();
    timings_.scan_ms = timer.elapsed_ms();

    // --- Read total num_rendered (host-device sync) ---
    uint32_t num_rendered = 0;
    CUDA_CHECK(cudaMemcpy(&num_rendered, d_offsets_ + (N - 1),
                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
    timings_.num_rendered = (int)num_rendered;

    // Clear tile ranges + framebuffer background early so empty scenes return bg
    CUDA_CHECK(cudaMemset(d_tile_ranges_, 0, (size_t)num_tiles_ * sizeof(uint2)));

    if (num_rendered == 0) {
        // No visible gaussians — fill framebuffer with bg and return
        std::vector<float3> h_bg((size_t)width_ * height_, config_.bg_color);
        cudaMemcpyH2D(d_framebuffer_, h_bg.data(), (size_t)width_ * height_);
        timings_.duplicate_ms = 0;
        timings_.sort_ms = 0;
        timings_.ranges_ms = 0;
        timings_.rasterize_ms = 0;
        timings_.total_ms = timings_.preprocess_ms + timings_.scan_ms;
        return d_framebuffer_;
    }

    ensureRenderedBuffers((int)num_rendered);
    ensureSortTempStorage((int)num_rendered);

    // --- Duplicate with keys ---
    timer.start();
    launchDuplicateWithKeys(
        N, d_means2D_, d_depths_, d_radii_, d_offsets_,
        grid_w_, grid_h_, d_keys_unsorted_, d_values_unsorted_
    );
    timer.stop();
    timings_.duplicate_ms = timer.elapsed_ms();

    // --- Radix sort ---
    timer.start();
    launchRadixSortPairsPreAlloc(
        d_sort_temp_, sort_temp_bytes_,
        d_keys_unsorted_, d_keys_sorted_,
        d_values_unsorted_, d_values_sorted_,
        (int)num_rendered
    );
    timer.stop();
    timings_.sort_ms = timer.elapsed_ms();

    // --- Tile ranges ---
    timer.start();
    launchGetTileRanges((int)num_rendered, d_keys_sorted_, num_tiles_, d_tile_ranges_);
    timer.stop();
    timings_.ranges_ms = timer.elapsed_ms();

    // --- Rasterize ---
    timer.start();
    launchRasterize(
        d_tile_ranges_, d_values_sorted_,
        d_means2D_, d_conic_opacity_, d_colors_,
        width_, height_, grid_w_, grid_h_,
        config_.bg_color, d_framebuffer_
    );
    timer.stop();
    timings_.rasterize_ms = timer.elapsed_ms();

    timings_.total_ms =
        timings_.preprocess_ms + timings_.scan_ms + timings_.duplicate_ms +
        timings_.sort_ms + timings_.ranges_ms + timings_.rasterize_ms;

    return d_framebuffer_;
}

// ============================================================
// Framebuffer utilities
// ============================================================

void Renderer::downloadFramebuffer(std::vector<float3>& h_fb) const {
    h_fb.resize((size_t)width_ * height_);
    cudaMemcpyD2H(h_fb.data(), d_framebuffer_, (size_t)width_ * height_);
}

void Renderer::saveFramebufferPPM(const std::string& path) const {
    std::vector<float3> h_fb;
    downloadFramebuffer(h_fb);

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path.c_str()); return; }
    fprintf(f, "P6\n%d %d\n255\n", width_, height_);

    std::vector<uint8_t> row((size_t)width_ * 3);
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            float3 c = h_fb[y * width_ + x];
            row[x*3+0] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, c.x)) * 255.0f);
            row[x*3+1] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, c.y)) * 255.0f);
            row[x*3+2] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, c.z)) * 255.0f);
        }
        fwrite(row.data(), 1, (size_t)width_ * 3, f);
    }
    fclose(f);
    printf("Saved %s (%dx%d)\n", path.c_str(), width_, height_);
}
