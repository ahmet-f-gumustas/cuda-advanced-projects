#ifndef RENDERER_H
#define RENDERER_H

#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"
#include "splatting_kernels.cuh"
#include <string>

// ============================================================
// Renderer configuration
// ============================================================

struct RendererConfig {
    float3 bg_color = make_float3(0.0f, 0.0f, 0.0f);
};

// ============================================================
// Renderer — orchestrates the full 3DGS render pipeline
// ============================================================

class Renderer {
public:
    struct Timings {
        float preprocess_ms = 0.0f;
        float scan_ms       = 0.0f;
        float duplicate_ms  = 0.0f;
        float sort_ms       = 0.0f;
        float ranges_ms     = 0.0f;
        float rasterize_ms  = 0.0f;
        float total_ms      = 0.0f;
        int   num_rendered  = 0;
    };

    Renderer();
    ~Renderer();

    // Allocate image-sized buffers (framebuffer, tile ranges)
    void init(int width, int height);
    void resize(int width, int height);
    void setConfig(const RendererConfig& c) { config_ = c; }
    void setBackgroundColor(float3 c) { config_.bg_color = c; }

    // Render one frame. Returns device pointer to framebuffer (H x W float3).
    float3* render(const GaussianModel& model, const Camera& camera);

    int  getWidth()  const { return width_; }
    int  getHeight() const { return height_; }
    float3* getFramebufferDevice() const { return d_framebuffer_; }

    const Timings& getLastTimings() const { return timings_; }

    // Copy framebuffer to host and optionally save as PPM
    void downloadFramebuffer(std::vector<float3>& h_fb) const;
    void saveFramebufferPPM(const std::string& path) const;

private:
    // Buffer management
    void ensureGaussianBuffers(int N);
    void freeGaussianBuffers();
    void ensureRenderedBuffers(int N);
    void freeRenderedBuffers();
    void allocateImageBuffers(int width, int height);
    void freeImageBuffers();
    void ensureScanTempStorage(int N);
    void ensureSortTempStorage(int N);

    RendererConfig config_;

    // Image dims
    int width_ = 0, height_ = 0;
    int grid_w_ = 0, grid_h_ = 0, num_tiles_ = 0;

    // Capacities (buffers grow, never shrink)
    int capacity_gaussians_ = 0;
    int capacity_rendered_  = 0;

    // Per-Gaussian buffers
    float2*   d_means2D_        = nullptr;
    float4*   d_conic_opacity_  = nullptr;
    float3*   d_colors_         = nullptr;
    float*    d_depths_         = nullptr;
    int*      d_radii_          = nullptr;
    int*      d_tiles_touched_  = nullptr;
    uint32_t* d_offsets_        = nullptr;

    // (Gaussian, tile) pair buffers — size = num_rendered
    uint64_t* d_keys_unsorted_   = nullptr;
    uint32_t* d_values_unsorted_ = nullptr;
    uint64_t* d_keys_sorted_     = nullptr;
    uint32_t* d_values_sorted_   = nullptr;

    // Per-tile buffer
    uint2* d_tile_ranges_ = nullptr;

    // Framebuffer
    float3* d_framebuffer_ = nullptr;

    // View matrix on device (16 floats, column-major)
    float* d_viewmatrix_ = nullptr;

    // Pre-allocated CUB temp storage
    void*  d_scan_temp_     = nullptr;
    size_t scan_temp_bytes_ = 0;
    void*  d_sort_temp_     = nullptr;
    size_t sort_temp_bytes_ = 0;

    Timings timings_;
};

#endif // RENDERER_H
