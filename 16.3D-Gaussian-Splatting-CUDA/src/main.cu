#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"
#include "renderer.h"
#include "ply_reader.h"

static void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --ply <path>          Load gaussians from a 3DGS-format .ply file\n");
    printf("  --random <N>          Generate N random gaussians (default 50000)\n");
    printf("  --width <W>           Render width  (default 800)\n");
    printf("  --height <H>          Render height (default 600)\n");
    printf("  --sh <degree>         SH degree for random scenes (0..3, default 3)\n");
    printf("  --out <path>          Output PPM filename (default render.ppm)\n");
    printf("\n");
}

int main(int argc, char** argv) {
    printf("=== 3D Gaussian Splatting — Phase 4 (PLY support) ===\n\n");
    printDeviceInfo();

    // --- Defaults ---
    std::string ply_path;
    std::string out_path = "render.ppm";
    int random_count = 50000;
    int width = 800, height = 600;
    int sh_deg = 3;

    // --- Parse args ---
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--ply"    && i+1 < argc) ply_path = argv[++i];
        else if (a == "--random" && i+1 < argc) random_count = std::atoi(argv[++i]);
        else if (a == "--width"  && i+1 < argc) width = std::atoi(argv[++i]);
        else if (a == "--height" && i+1 < argc) height = std::atoi(argv[++i]);
        else if (a == "--sh"     && i+1 < argc) sh_deg = std::atoi(argv[++i]);
        else if (a == "--out"    && i+1 < argc) out_path = argv[++i];
        else if (a == "--help" || a == "-h")    { printUsage(argv[0]); return 0; }
    }

    // --- Camera ---
    Camera camera;
    CameraConfig cam_cfg;
    cam_cfg.width  = width;
    cam_cfg.height = height;
    cam_cfg.fov_y  = 60.0f;
    camera.setConfig(cam_cfg);
    camera.printInfo();

    // --- Scene: PLY or random ---
    GaussianModel model;
    if (!ply_path.empty()) {
        if (!model.loadFromPLY(ply_path)) {
            fprintf(stderr, "Failed to load PLY: %s\n", ply_path.c_str());
            return 1;
        }
    } else {
        float3 bmin = {-3.0f, -3.0f, -3.0f};
        float3 bmax = { 3.0f,  3.0f,  3.0f};
        model.generateRandom(random_count, sh_deg, bmin, bmax);
    }

    // --- Renderer ---
    Renderer renderer;
    renderer.init(width, height);
    renderer.setBackgroundColor(make_float3(0.05f, 0.05f, 0.08f));

    // --- Warm up + measure ---
    const int WARMUP = 2;
    const int FRAMES = 5;
    for (int i = 0; i < WARMUP; i++) renderer.render(model, camera);

    float total = 0.0f;
    Renderer::Timings t0;
    for (int i = 0; i < FRAMES; i++) {
        renderer.render(model, camera);
        auto& t = renderer.getLastTimings();
        total += t.total_ms;
        if (i == 0) t0 = t;
    }
    float avg = total / FRAMES;

    printf("\nFirst-frame breakdown:\n");
    printf("  preprocess : %6.3f ms\n", t0.preprocess_ms);
    printf("  scan       : %6.3f ms\n", t0.scan_ms);
    printf("  duplicate  : %6.3f ms\n", t0.duplicate_ms);
    printf("  sort       : %6.3f ms  (%d pairs)\n", t0.sort_ms, t0.num_rendered);
    printf("  ranges     : %6.3f ms\n", t0.ranges_ms);
    printf("  rasterize  : %6.3f ms\n", t0.rasterize_ms);
    printf("  total      : %6.3f ms\n", t0.total_ms);
    printf("\nAverage over %d frames: %.3f ms  (%.1f FPS)\n\n",
           FRAMES, avg, 1000.0f / avg);

    renderer.saveFramebufferPPM(out_path);

    printf("\n=== Phase 4 Complete ===\n");
    return 0;
}
