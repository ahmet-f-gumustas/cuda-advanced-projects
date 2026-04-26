#include "viewer.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstring>

// ============================================================
// Minimal modern-OpenGL declarations (no GLEW, no <GL/gl.h>)
// ============================================================

#ifndef APIENTRY
#define APIENTRY
#endif

typedef unsigned int  GLenum;
typedef unsigned int  GLuint;
typedef int           GLint;
typedef int           GLsizei;
typedef unsigned char GLboolean;
typedef unsigned int  GLbitfield;
typedef float         GLfloat;
typedef char          GLchar;
typedef ptrdiff_t     GLsizeiptr;
typedef ptrdiff_t     GLintptr;
typedef void          GLvoid;

#define GL_FALSE              0
#define GL_TRUE               1
#define GL_FLOAT              0x1406
#define GL_UNSIGNED_BYTE      0x1401
#define GL_TRIANGLE_STRIP     0x0005
#define GL_RGBA               0x1908
#define GL_TEXTURE_2D         0x0DE1
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_WRAP_S     0x2802
#define GL_TEXTURE_WRAP_T     0x2803
#define GL_LINEAR             0x2601
#define GL_COLOR_BUFFER_BIT   0x00004000
#define GL_VERTEX_SHADER      0x8B31
#define GL_FRAGMENT_SHADER    0x8B30
#define GL_COMPILE_STATUS     0x8B81
#define GL_LINK_STATUS        0x8B82
#define GL_ARRAY_BUFFER       0x8892
#define GL_STATIC_DRAW        0x88E4
#define GL_TEXTURE0           0x84C0
#define GL_CLAMP_TO_EDGE      0x812F

// Modern GL (load via glfwGetProcAddress)
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum);
typedef void   (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint, GLsizei, const GLchar* const*, const GLint*);
typedef void   (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint);
typedef void   (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void   (APIENTRY *PFNGLDELETESHADERPROC)(GLuint);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)();
typedef void   (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint, GLuint);
typedef void   (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint);
typedef void   (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void   (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint);
typedef void   (APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint);
typedef GLint  (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint, const GLchar*);
typedef void   (APIENTRY *PFNGLUNIFORM1IPROC)(GLint, GLint);
typedef void   (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei, GLuint*);
typedef void   (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint);
typedef void   (APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei, const GLuint*);
typedef void   (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei, GLuint*);
typedef void   (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void   (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum, GLsizeiptr, const void*, GLenum);
typedef void   (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint*);
typedef void   (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
typedef void   (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint);
typedef void   (APIENTRY *PFNGLACTIVETEXTUREPROC)(GLenum);

// Legacy GL 1.1 (also loadable via glfwGetProcAddress on most systems)
typedef void   (APIENTRY *PFNGLCLEARPROC)(GLbitfield);
typedef void   (APIENTRY *PFNGLCLEARCOLORPROC)(GLfloat, GLfloat, GLfloat, GLfloat);
typedef void   (APIENTRY *PFNGLVIEWPORTPROC)(GLint, GLint, GLsizei, GLsizei);
typedef void   (APIENTRY *PFNGLGENTEXTURESPROC)(GLsizei, GLuint*);
typedef void   (APIENTRY *PFNGLBINDTEXTUREPROC)(GLenum, GLuint);
typedef void   (APIENTRY *PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint*);
typedef void   (APIENTRY *PFNGLTEXIMAGE2DPROC)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
typedef void   (APIENTRY *PFNGLTEXPARAMETERIPROC)(GLenum, GLenum, GLint);
typedef void   (APIENTRY *PFNGLDRAWARRAYSPROC)(GLenum, GLint, GLsizei);

static PFNGLCREATESHADERPROC            glCreateShader            = nullptr;
static PFNGLSHADERSOURCEPROC            glShaderSource            = nullptr;
static PFNGLCOMPILESHADERPROC           glCompileShader           = nullptr;
static PFNGLGETSHADERIVPROC             glGetShaderiv             = nullptr;
static PFNGLGETSHADERINFOLOGPROC        glGetShaderInfoLog        = nullptr;
static PFNGLDELETESHADERPROC            glDeleteShader            = nullptr;
static PFNGLCREATEPROGRAMPROC           glCreateProgram           = nullptr;
static PFNGLATTACHSHADERPROC            glAttachShader            = nullptr;
static PFNGLLINKPROGRAMPROC             glLinkProgram             = nullptr;
static PFNGLGETPROGRAMIVPROC            glGetProgramiv            = nullptr;
static PFNGLGETPROGRAMINFOLOGPROC       glGetProgramInfoLog       = nullptr;
static PFNGLUSEPROGRAMPROC              glUseProgram              = nullptr;
static PFNGLDELETEPROGRAMPROC           glDeleteProgram           = nullptr;
static PFNGLGETUNIFORMLOCATIONPROC      glGetUniformLocation      = nullptr;
static PFNGLUNIFORM1IPROC               glUniform1i               = nullptr;
static PFNGLGENVERTEXARRAYSPROC         glGenVertexArrays         = nullptr;
static PFNGLBINDVERTEXARRAYPROC         glBindVertexArray         = nullptr;
static PFNGLDELETEVERTEXARRAYSPROC      glDeleteVertexArrays      = nullptr;
static PFNGLGENBUFFERSPROC              glGenBuffers              = nullptr;
static PFNGLBINDBUFFERPROC              glBindBuffer              = nullptr;
static PFNGLBUFFERDATAPROC              glBufferData              = nullptr;
static PFNGLDELETEBUFFERSPROC           glDeleteBuffers           = nullptr;
static PFNGLVERTEXATTRIBPOINTERPROC     glVertexAttribPointer     = nullptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
static PFNGLACTIVETEXTUREPROC           glActiveTexture           = nullptr;
static PFNGLCLEARPROC                   glClear                   = nullptr;
static PFNGLCLEARCOLORPROC              glClearColor              = nullptr;
static PFNGLVIEWPORTPROC                glViewport                = nullptr;
static PFNGLGENTEXTURESPROC             glGenTextures             = nullptr;
static PFNGLBINDTEXTUREPROC             glBindTexture             = nullptr;
static PFNGLDELETETEXTURESPROC          glDeleteTextures          = nullptr;
static PFNGLTEXIMAGE2DPROC              glTexImage2D              = nullptr;
static PFNGLTEXPARAMETERIPROC           glTexParameteri           = nullptr;
static PFNGLDRAWARRAYSPROC              glDrawArrays              = nullptr;

#define LOAD_GL(name) name = (decltype(name))glfwGetProcAddress(#name)

static bool loadGLFunctions() {
    LOAD_GL(glCreateShader);
    LOAD_GL(glShaderSource);
    LOAD_GL(glCompileShader);
    LOAD_GL(glGetShaderiv);
    LOAD_GL(glGetShaderInfoLog);
    LOAD_GL(glDeleteShader);
    LOAD_GL(glCreateProgram);
    LOAD_GL(glAttachShader);
    LOAD_GL(glLinkProgram);
    LOAD_GL(glGetProgramiv);
    LOAD_GL(glGetProgramInfoLog);
    LOAD_GL(glUseProgram);
    LOAD_GL(glDeleteProgram);
    LOAD_GL(glGetUniformLocation);
    LOAD_GL(glUniform1i);
    LOAD_GL(glGenVertexArrays);
    LOAD_GL(glBindVertexArray);
    LOAD_GL(glDeleteVertexArrays);
    LOAD_GL(glGenBuffers);
    LOAD_GL(glBindBuffer);
    LOAD_GL(glBufferData);
    LOAD_GL(glDeleteBuffers);
    LOAD_GL(glVertexAttribPointer);
    LOAD_GL(glEnableVertexAttribArray);
    LOAD_GL(glActiveTexture);
    LOAD_GL(glClear);
    LOAD_GL(glClearColor);
    LOAD_GL(glViewport);
    LOAD_GL(glGenTextures);
    LOAD_GL(glBindTexture);
    LOAD_GL(glDeleteTextures);
    LOAD_GL(glTexImage2D);
    LOAD_GL(glTexParameteri);
    LOAD_GL(glDrawArrays);

    return glCreateShader && glShaderSource && glCompileShader && glLinkProgram &&
           glGenVertexArrays && glGenBuffers && glActiveTexture &&
           glClear && glTexImage2D && glDrawArrays;
}

// ============================================================
// Inline shaders (fullscreen textured quad)
// ============================================================

static const char* kVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* kFragmentShader = R"(
#version 330 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vTex);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// ============================================================
// CUDA kernel: float3 framebuffer → uchar4 (RGBA8) for display
// ============================================================

__global__ void fb_to_rgba8_kernel(const float3* __restrict__ in,
                                    uchar4* __restrict__ out,
                                    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float3 c = in[i];
    out[i] = make_uchar4(
        (unsigned char)(fminf(1.0f, fmaxf(0.0f, c.x)) * 255.0f),
        (unsigned char)(fminf(1.0f, fmaxf(0.0f, c.y)) * 255.0f),
        (unsigned char)(fminf(1.0f, fmaxf(0.0f, c.z)) * 255.0f),
        255
    );
}

// ============================================================
// Viewer implementation
// ============================================================

Viewer::Viewer() = default;

Viewer::~Viewer() {
    if (d_display_buffer_) cudaFree(d_display_buffer_);
    shutdownGL();
    if (window_) {
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

bool Viewer::init(int width, int height, const char* title) {
    if (!glfwInit()) {
        fprintf(stderr, "GLFW init failed\n");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0); // disable vsync to measure raw FPS

    if (!loadGLFunctions()) {
        fprintf(stderr, "Failed to load required OpenGL functions\n");
        return false;
    }

    width_  = width;
    height_ = height;

    glfwSetWindowUserPointer(window_, this);
    glfwSetCursorPosCallback   (window_, onCursorPos);
    glfwSetMouseButtonCallback (window_, onMouseButton);
    glfwSetScrollCallback      (window_, onScroll);
    glfwSetKeyCallback         (window_, onKey);
    glfwSetFramebufferSizeCallback(window_, onResize);

    if (!initGL()) return false;

    last_fps_time_ = glfwGetTime();
    return true;
}

bool Viewer::initGL() {
    GLuint vs = compileShader(GL_VERTEX_SHADER,   kVertexShader);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs) return false;
    program_ = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    if (!program_) return false;

    // Fullscreen quad: pos.xy + tex.uv  (Y flipped — image is y-down)
    float verts[] = {
        // pos     tex
        -1.f, -1.f,  0.f, 1.f,
         1.f, -1.f,  1.f, 1.f,
        -1.f,  1.f,  0.f, 0.f,
         1.f,  1.f,  1.f, 0.f,
    };

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                           (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return true;
}

void Viewer::shutdownGL() {
    if (texture_) glDeleteTextures(1, &texture_);
    if (vbo_)     glDeleteBuffers(1, &vbo_);
    if (vao_)     glDeleteVertexArrays(1, &vao_);
    if (program_) glDeleteProgram(program_);
    program_ = vao_ = vbo_ = texture_ = 0;
}

void Viewer::ensureDisplayBuffer(int W, int H) {
    int needed = W * H;
    if (needed <= display_capacity_) return;
    if (d_display_buffer_) CUDA_CHECK(cudaFree(d_display_buffer_));
    CUDA_CHECK(cudaMalloc(&d_display_buffer_, (size_t)needed * sizeof(uchar4)));
    h_display_buffer_.resize((size_t)needed * 4);
    display_capacity_ = needed;
}

void Viewer::displayFrame(float3* d_fb, int W, int H) {
    int N = W * H;
    ensureDisplayBuffer(W, H);

    // Convert float3 → uchar4 on GPU
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fb_to_rgba8_kernel<<<blocks, threads>>>(d_fb, (uchar4*)d_display_buffer_, N);
    CUDA_CHECK_LAST_ERROR();

    // D2H copy
    CUDA_CHECK(cudaMemcpy(h_display_buffer_.data(), d_display_buffer_,
                          (size_t)N * sizeof(uchar4), cudaMemcpyDeviceToHost));

    // Upload to texture
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 h_display_buffer_.data());

    // Draw
    glViewport(0, 0, width_, height_);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(glGetUniformLocation(program_, "uTex"), 0);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void Viewer::updateTitle(double frame_ms) {
    fps_frames_++;
    double now = glfwGetTime();
    if (now - last_fps_time_ >= 0.5) {
        double fps = fps_frames_ / (now - last_fps_time_);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "3D Gaussian Splatting — %dx%d — %.1f FPS — %.2f ms/frame",
                 width_, height_, fps, frame_ms);
        glfwSetWindowTitle(window_, buf);
        last_fps_time_ = now;
        fps_frames_ = 0;
    }
}

void Viewer::run(const GaussianModel& model, Renderer& renderer, Camera& camera) {
    camera_ptr_   = &camera;
    renderer_ptr_ = &renderer;
    camera.setSize(width_, height_);
    CameraConfig cfg = camera.getConfig();
    cfg.width = width_;
    cfg.height = height_;
    camera.setConfig(cfg);
    renderer.resize(width_, height_);

    printf("\nViewer running. Controls:\n");
    printf("  LMB drag  : orbit\n");
    printf("  RMB drag  : pan\n");
    printf("  Scroll    : zoom\n");
    printf("  R         : reset camera\n");
    printf("  P         : print stats\n");
    printf("  ESC       : exit\n\n");

    while (!glfwWindowShouldClose(window_)) {
        // If resize happened, sync state
        if (renderer.getWidth() != width_ || renderer.getHeight() != height_) {
            renderer.resize(width_, height_);
            CameraConfig cf = camera.getConfig();
            cf.width = width_; cf.height = height_;
            camera.setConfig(cf);
        }

        float3* d_fb = renderer.render(model, camera);
        auto& t = renderer.getLastTimings();

        if (print_stats_) {
            printf("Frame: pp=%.2f scan=%.2f dup=%.2f sort=%.2f rast=%.2f total=%.2f ms (%d pairs)\n",
                   t.preprocess_ms, t.scan_ms, t.duplicate_ms, t.sort_ms,
                   t.rasterize_ms, t.total_ms, t.num_rendered);
            print_stats_ = false;
        }

        displayFrame(d_fb, width_, height_);
        updateTitle(t.total_ms);

        glfwSwapBuffers(window_);
        glfwPollEvents();
    }
}

// ============================================================
// Static GLFW callbacks
// ============================================================

void Viewer::onCursorPos(GLFWwindow* w, double x, double y) {
    Viewer* v = (Viewer*)glfwGetWindowUserPointer(w);
    if (!v || !v->camera_ptr_) return;

    double dx = x - v->last_x_;
    double dy = y - v->last_y_;
    v->last_x_ = x;
    v->last_y_ = y;

    if (v->dragging_left_) {
        v->camera_ptr_->orbit((float)dx, (float)dy);
    } else if (v->dragging_right_) {
        v->camera_ptr_->pan((float)dx, (float)dy);
    }
}

void Viewer::onMouseButton(GLFWwindow* w, int button, int action, int /*mods*/) {
    Viewer* v = (Viewer*)glfwGetWindowUserPointer(w);
    if (!v) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        v->dragging_left_ = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        v->dragging_right_ = (action == GLFW_PRESS);
    }
    if (action == GLFW_PRESS) {
        glfwGetCursorPos(w, &v->last_x_, &v->last_y_);
    }
}

void Viewer::onScroll(GLFWwindow* w, double /*xoff*/, double yoff) {
    Viewer* v = (Viewer*)glfwGetWindowUserPointer(w);
    if (!v || !v->camera_ptr_) return;
    v->camera_ptr_->zoom((float)yoff);
}

void Viewer::onKey(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/) {
    Viewer* v = (Viewer*)glfwGetWindowUserPointer(w);
    if (!v) return;
    if (action != GLFW_PRESS) return;

    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(w, GLFW_TRUE);
            break;
        case GLFW_KEY_R:
            if (v->camera_ptr_) v->camera_ptr_->reset();
            break;
        case GLFW_KEY_P:
            v->print_stats_ = true;
            break;
    }
}

void Viewer::onResize(GLFWwindow* w, int width, int height) {
    Viewer* v = (Viewer*)glfwGetWindowUserPointer(w);
    if (!v) return;
    v->width_  = width;
    v->height_ = height;
}
