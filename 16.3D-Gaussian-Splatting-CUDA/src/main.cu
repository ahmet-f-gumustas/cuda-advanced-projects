#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"
#include "renderer.h"

int main() {
    printf("=== 3D Gaussian Splatting — Phase 3 (Renderer class) ===\n\n");
    printDeviceInfo();

    // --- Camera ---
    Camera camera;
    CameraConfig cam_cfg;
    cam_cfg.width = 800;
    cam_cfg.height = 600;
    cam_cfg.fov_y = 60.0f;
    camera.setConfig(cam_cfg);
    camera.printInfo();

    // --- Scene ---
    GaussianModel model;
    int N = 50000;
    float3 bmin = {-3.0f, -3.0f, -3.0f};
    float3 bmax = { 3.0f,  3.0f,  3.0f};
    model.generateRandom(N, 3, bmin, bmax);

    // --- Renderer ---
    Renderer renderer;
    renderer.init(cam_cfg.width, cam_cfg.height);
    renderer.setBackgroundColor(make_float3(0.05f, 0.05f, 0.08f));

    // --- Render 3 times to warm up and measure ---
    printf("Rendering 5 frames...\n\n");
    float total_ms_accum = 0.0f;
    const int WARMUP = 2;
    const int FRAMES = 5;

    for (int i = 0; i < WARMUP + FRAMES; i++) {
        renderer.render(model, camera);
        auto& t = renderer.getLastTimings();

        if (i == WARMUP) {
            // Print breakdown on first timed frame
            printf("  Frame %d breakdown:\n", i);
            printf("    preprocess : %6.3f ms\n", t.preprocess_ms);
            printf("    scan       : %6.3f ms\n", t.scan_ms);
            printf("    duplicate  : %6.3f ms\n", t.duplicate_ms);
            printf("    sort       : %6.3f ms  (%d pairs)\n", t.sort_ms, t.num_rendered);
            printf("    ranges     : %6.3f ms\n", t.ranges_ms);
            printf("    rasterize  : %6.3f ms\n", t.rasterize_ms);
            printf("    total      : %6.3f ms  (%.1f FPS)\n\n", t.total_ms, 1000.0f / t.total_ms);
        }
        if (i >= WARMUP) total_ms_accum += t.total_ms;
    }
    float avg_ms = total_ms_accum / FRAMES;
    printf("Average over %d frames: %.3f ms  (%.1f FPS)\n\n", FRAMES, avg_ms, 1000.0f / avg_ms);

    // --- Save one static frame ---
    renderer.saveFramebufferPPM("render.ppm");

    // --- Render from a rotated camera ---
    camera.orbit(60.0f, 20.0f);
    renderer.render(model, camera);
    renderer.saveFramebufferPPM("render_rotated.ppm");

    // --- Render at higher resolution ---
    printf("\nRendering at 1280x720...\n");
    camera.reset();
    camera.setSize(1280, 720);
    CameraConfig hd_cfg = cam_cfg;
    hd_cfg.width = 1280; hd_cfg.height = 720;
    camera.setConfig(hd_cfg);
    renderer.resize(1280, 720);
    renderer.render(model, camera);
    auto& t = renderer.getLastTimings();
    printf("  1280x720 total: %.3f ms  (%.1f FPS)\n", t.total_ms, 1000.0f / t.total_ms);
    renderer.saveFramebufferPPM("render_hd.ppm");

    printf("\n=== Phase 3 Complete ===\n");
    return 0;
}
