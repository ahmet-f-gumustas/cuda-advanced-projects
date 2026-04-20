#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"
#include "renderer.h"

struct BenchmarkCase {
    int num_gaussians;
    int width;
    int height;
    const char* label;
};

void runCase(const BenchmarkCase& c) {
    GaussianModel model;
    float3 bmin = {-3, -3, -3}, bmax = {3, 3, 3};
    model.generateRandom(c.num_gaussians, 3, bmin, bmax, 42);

    Camera camera;
    CameraConfig cfg;
    cfg.width = c.width; cfg.height = c.height; cfg.fov_y = 60.0f;
    camera.setConfig(cfg);

    Renderer renderer;
    renderer.init(c.width, c.height);

    // Warm up
    for (int i = 0; i < 3; i++) renderer.render(model, camera);

    // Measure 10 frames
    const int FRAMES = 10;
    float total = 0.0f;
    Renderer::Timings first_t;
    for (int i = 0; i < FRAMES; i++) {
        renderer.render(model, camera);
        auto& t = renderer.getLastTimings();
        total += t.total_ms;
        if (i == 0) first_t = t;
    }
    float avg = total / FRAMES;

    printf("  %-24s %7d Gaussians @ %4dx%-4d : %7.3f ms  (%6.1f FPS)\n",
           c.label, c.num_gaussians, c.width, c.height, avg, 1000.0f / avg);
    printf("    breakdown: pp=%.2f scan=%.2f dup=%.2f sort=%.2f (%d pairs) ranges=%.2f rast=%.2f\n",
           first_t.preprocess_ms, first_t.scan_ms, first_t.duplicate_ms,
           first_t.sort_ms, first_t.num_rendered, first_t.ranges_ms, first_t.rasterize_ms);
}

int main() {
    printf("=== 3D Gaussian Splatting — Benchmark ===\n\n");
    printDeviceInfo();

    BenchmarkCase cases[] = {
        {  10000,  800,  600, "Small"      },
        {  50000,  800,  600, "Medium"     },
        { 100000, 1280,  720, "HD"         },
        { 500000, 1280,  720, "HD-dense"   },
        { 500000, 1920, 1080, "FHD"        },
        {1000000, 1920, 1080, "FHD-1M"     },
        {2000000, 1920, 1080, "FHD-2M"     },
    };

    for (auto& c : cases) runCase(c);

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
