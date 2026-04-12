#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"

int main() {
    printf("=== 3D Gaussian Splatting — Phase 1 Checkpoint ===\n\n");

    // Print GPU info
    printDeviceInfo();

    // Create camera and print info
    Camera camera;
    CameraConfig cam_cfg;
    cam_cfg.width = 1280;
    cam_cfg.height = 720;
    cam_cfg.fov_y = 60.0f;
    camera.setConfig(cam_cfg);
    camera.printInfo();

    // Print view matrix
    Mat4 view = camera.getViewMatrix();
    view.print("View Matrix");
    printf("\n");

    // Print projection matrix
    Mat4 proj = camera.getProjectionMatrix();
    proj.print("Projection Matrix");
    printf("\n");

    // Generate random Gaussians
    GaussianModel model;
    float3 bbox_min = {-2.0f, -2.0f, -2.0f};
    float3 bbox_max = { 2.0f,  2.0f,  2.0f};
    model.generateRandom(10000, 3, bbox_min, bbox_max);

    printf("Model info:\n");
    printf("  Count: %d\n", model.getCount());
    printf("  SH degree: %d (%d coefficients)\n", model.getSHDegree(), model.getNumSHCoeffs());
    printf("  Allocated: %s\n", model.isAllocated() ? "yes" : "no");
    printf("\n");

    // Test camera controls
    printf("Testing camera orbit...\n");
    camera.orbit(45.0f, 15.0f);
    camera.printInfo();

    printf("Testing camera zoom...\n");
    camera.zoom(2.0f);
    camera.printInfo();

    printf("Testing camera reset...\n");
    camera.reset();
    camera.printInfo();

    printf("=== Phase 1 Complete ===\n");
    return 0;
}
