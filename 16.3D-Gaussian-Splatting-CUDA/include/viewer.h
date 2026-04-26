#ifndef VIEWER_H
#define VIEWER_H

#include "cuda_utils.h"
#include "camera.h"
#include "gaussian.h"
#include "renderer.h"

struct GLFWwindow;

// ============================================================
// Viewer — interactive OpenGL display + camera controls
//
// Controls:
//   Left mouse drag  : orbit camera
//   Right mouse drag : pan camera
//   Scroll wheel     : zoom in/out
//   R                : reset camera
//   P                : print stats
//   ESC              : exit
// ============================================================

class Viewer {
public:
    Viewer();
    ~Viewer();

    bool init(int width, int height, const char* title);
    void run(const GaussianModel& model, Renderer& renderer, Camera& camera);

private:
    GLFWwindow* window_ = nullptr;
    int width_ = 0, height_ = 0;

    // OpenGL handles
    unsigned int program_ = 0;
    unsigned int vao_     = 0;
    unsigned int vbo_     = 0;
    unsigned int texture_ = 0;

    // GPU-side display buffer (uchar4 RGBA)
    void* d_display_buffer_   = nullptr;
    int   display_capacity_   = 0;
    std::vector<unsigned char> h_display_buffer_;

    // Input state
    double last_x_ = 0.0, last_y_ = 0.0;
    bool   dragging_left_  = false;
    bool   dragging_right_ = false;

    // Cached pointers for static callbacks
    Camera*   camera_ptr_   = nullptr;
    Renderer* renderer_ptr_ = nullptr;
    bool      print_stats_  = false;

    // FPS tracking
    double last_fps_time_ = 0.0;
    int    fps_frames_    = 0;

    // Internal
    bool initGL();
    void shutdownGL();
    void displayFrame(float3* d_fb, int W, int H);
    void ensureDisplayBuffer(int W, int H);
    void updateTitle(double frame_ms);

    // Callbacks
    static void onCursorPos(GLFWwindow* w, double x, double y);
    static void onMouseButton(GLFWwindow* w, int button, int action, int mods);
    static void onScroll(GLFWwindow* w, double xoff, double yoff);
    static void onKey(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void onResize(GLFWwindow* w, int width, int height);
};

#endif // VIEWER_H
