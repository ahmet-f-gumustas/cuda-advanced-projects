#include "cuda_utils.h"
#include "gaussian.h"
#include "camera.h"

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("TEST: %s ... ", name)
#define PASS() do { printf("PASSED\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAILED: %s\n", msg); tests_failed++; } while(0)

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

    // View matrix should have non-zero diagonal
    bool view_ok = (view.at(0,0) != 0 && view.at(1,1) != 0 && view.at(2,2) != 0);
    // Projection: aspect ratio check
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
    // Position should change after orbit
    bool changed = (fabsf(pos_before.x - pos_after.x) > 0.01f ||
                    fabsf(pos_before.z - pos_after.z) > 0.01f);
    if (changed) { PASS(); } else { FAIL("position did not change"); }
}

int main() {
    printf("=== 3D Gaussian Splatting — Phase 1 Tests ===\n\n");
    printDeviceInfo();

    test_gaussian_allocation();
    test_camera_matrices();
    test_camera_focal();
    test_camera_orbit();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
