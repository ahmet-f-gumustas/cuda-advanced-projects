#include "renderer.h"
#include <iostream>
#include <cmath>
#include <cstring>

// OpenGL function pointers (modern OpenGL 3.3+)
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

// Function pointers for modern OpenGL
PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLDELETESHADERPROC glDeleteShader = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLBUFFERSUBDATAPROC glBufferSubData = nullptr;
PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = nullptr;

// Load OpenGL functions
void loadOpenGLFunctions() {
    glCreateShader = (PFNGLCREATESHADERPROC)glfwGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)glfwGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)glfwGetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)glfwGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)glfwGetProcAddress("glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)glfwGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)glfwGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)glfwGetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)glfwGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)glfwGetProcAddress("glGetProgramInfoLog");
    glDeleteShader = (PFNGLDELETESHADERPROC)glfwGetProcAddress("glDeleteShader");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)glfwGetProcAddress("glDeleteProgram");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glfwGetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glfwGetProcAddress("glBindVertexArray");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glfwGetProcAddress("glDeleteVertexArrays");
    glGenBuffers = (PFNGLGENBUFFERSPROC)glfwGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)glfwGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)glfwGetProcAddress("glBufferData");
    glBufferSubData = (PFNGLBUFFERSUBDATAPROC)glfwGetProcAddress("glBufferSubData");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glfwGetProcAddress("glDeleteBuffers");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)glfwGetProcAddress("glVertexAttribPointer");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)glfwGetProcAddress("glEnableVertexAttribArray");
    glUseProgram = (PFNGLUSEPROGRAMPROC)glfwGetProcAddress("glUseProgram");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)glfwGetProcAddress("glGetUniformLocation");
    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)glfwGetProcAddress("glUniformMatrix4fv");
}

// Vertex shader source
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 particleColor;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = 3.0;
    particleColor = aColor;
}
)";

// Fragment shader source
const char* fragmentShaderSource = R"(
#version 330 core
in vec3 particleColor;
out vec4 FragColor;

void main() {
    // Make points circular
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    // Soft edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);

    FragColor = vec4(particleColor, alpha);
}
)";

// Global pointer for callbacks
static Renderer* g_renderer = nullptr;

// ============================================================================
// Callback Functions
// ============================================================================

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (!g_renderer) return;

    auto& mouse = g_renderer->getMouseState();

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouse.left_pressed = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse.right_pressed = (action == GLFW_PRESS);
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!g_renderer) return;

    auto& mouse = g_renderer->getMouseState();
    auto& camera = g_renderer->getCamera();

    if (mouse.first_mouse) {
        mouse.last_x = xpos;
        mouse.last_y = ypos;
        mouse.first_mouse = false;
    }

    double xoffset = xpos - mouse.last_x;
    double yoffset = mouse.last_y - ypos;
    mouse.last_x = xpos;
    mouse.last_y = ypos;

    const float sensitivity = 0.3f;

    if (mouse.left_pressed) {
        // Rotate camera
        camera.rotation_y += xoffset * sensitivity;
        camera.rotation_x += yoffset * sensitivity;

        // Clamp pitch
        if (camera.rotation_x > 89.0f) camera.rotation_x = 89.0f;
        if (camera.rotation_x < -89.0f) camera.rotation_x = -89.0f;
    } else if (mouse.right_pressed) {
        // Pan camera
        camera.pan_x += xoffset * 0.01f;
        camera.pan_y += yoffset * 0.01f;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!g_renderer) return;

    auto& camera = g_renderer->getCamera();
    camera.distance -= yoffset * 2.0f;

    if (camera.distance < 5.0f) camera.distance = 5.0f;
    if (camera.distance > 200.0f) camera.distance = 200.0f;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if (!g_renderer) return;
    auto& camera = g_renderer->getCamera();

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        const float speed = 2.0f;

        if (key == GLFW_KEY_W) camera.distance -= speed;
        if (key == GLFW_KEY_S) camera.distance += speed;
        if (key == GLFW_KEY_R) {
            // Reset camera
            camera = Camera();
        }
    }
}

// ============================================================================
// Matrix Math (Simple 4x4 matrix operations)
// ============================================================================

void matrixIdentity(float* m) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

void matrixPerspective(float* m, float fov, float aspect, float near, float far) {
    memset(m, 0, 16 * sizeof(float));

    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);

    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0f;
    m[14] = (2.0f * far * near) / (near - far);
}

void matrixLookAt(float* m, float eyeX, float eyeY, float eyeZ,
                  float centerX, float centerY, float centerZ,
                  float upX, float upY, float upZ) {
    // Forward vector
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    float flen = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;

    // Right vector (cross product of forward and up)
    float rx = fy * upZ - fz * upY;
    float ry = fz * upX - fx * upZ;
    float rz = fx * upY - fy * upX;
    float rlen = sqrtf(rx*rx + ry*ry + rz*rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    // Up vector (cross product of right and forward)
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    m[0] = rx;  m[1] = ux;  m[2] = -fx; m[3] = 0.0f;
    m[4] = ry;  m[5] = uy;  m[6] = -fy; m[7] = 0.0f;
    m[8] = rz;  m[9] = uz;  m[10] = -fz; m[11] = 0.0f;
    m[12] = -(rx*eyeX + ry*eyeY + rz*eyeZ);
    m[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
    m[14] = (fx*eyeX + fy*eyeY + fz*eyeZ);
    m[15] = 1.0f;
}

// ============================================================================
// Renderer Implementation
// ============================================================================

Renderer::Renderer(int width, int height, const std::string& title)
    : window_(nullptr), width_(width), height_(height), title_(title),
      shader_program_(0), vao_(0), vbo_position_(0), vbo_color_(0),
      colors_(nullptr), max_bodies_(100000),
      last_frame_time_(0.0), fps_(0.0), frame_count_(0) {

    g_renderer = this;
}

Renderer::~Renderer() {
    if (colors_) delete[] colors_;

    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_position_) glDeleteBuffers(1, &vbo_position_);
    if (vbo_color_) glDeleteBuffers(1, &vbo_color_);
    if (shader_program_) glDeleteProgram(shader_program_);

    if (window_) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();

    g_renderer = nullptr;
}

bool Renderer::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Set OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

    // Create window
    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);

    // Load OpenGL functions
    loadOpenGLFunctions();

    glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetCursorPosCallback(window_, cursor_position_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    glfwSetKeyCallback(window_, key_callback);

    // Enable V-Sync
    glfwSwapInterval(1);

    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);

    // Create shader program
    shader_program_ = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (!shader_program_) {
        std::cerr << "Failed to create shader program" << std::endl;
        return false;
    }

    // Setup buffers
    setupBuffers();

    // Allocate color array
    colors_ = new float[max_bodies_ * 3];

    last_frame_time_ = glfwGetTime();

    std::cout << "\n=== OpenGL Renderer Initialized ===" << std::endl;
    std::cout << "  OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "  GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "  Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Left Mouse: Rotate camera" << std::endl;
    std::cout << "  Right Mouse: Pan camera" << std::endl;
    std::cout << "  Scroll Wheel: Zoom in/out" << std::endl;
    std::cout << "  W/S: Zoom in/out" << std::endl;
    std::cout << "  R: Reset camera" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    std::cout << "====================================\n" << std::endl;

    return true;
}

unsigned int Renderer::compileShader(const char* source, unsigned int type) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }

    return shader;
}

unsigned int Renderer::createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    unsigned int vertexShader = compileShader(vertexSrc, GL_VERTEX_SHADER);
    unsigned int fragmentShader = compileShader(fragmentSrc, GL_FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) {
        return 0;
    }

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void Renderer::setupBuffers() {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_position_);
    glGenBuffers(1, &vbo_color_);

    glBindVertexArray(vao_);

    // Position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position_);
    glBufferData(GL_ARRAY_BUFFER, max_bodies_ * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_color_);
    glBufferData(GL_ARRAY_BUFFER, max_bodies_ * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void Renderer::render(const float* x, const float* y, const float* z,
                      const float* mass, int numBodies) {
    // Clear screen
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Prepare positions
    float* positions = new float[numBodies * 3];
    for (int i = 0; i < numBodies; ++i) {
        positions[i * 3 + 0] = x[i];
        positions[i * 3 + 1] = y[i];
        positions[i * 3 + 2] = z[i];
    }

    // Generate colors based on mass
    float min_mass = mass[0], max_mass = mass[0];
    for (int i = 1; i < numBodies; ++i) {
        if (mass[i] < min_mass) min_mass = mass[i];
        if (mass[i] > max_mass) max_mass = mass[i];
    }

    for (int i = 0; i < numBodies; ++i) {
        float normalized_mass = (mass[i] - min_mass) / (max_mass - min_mass + 0.001f);

        // Color gradient: blue (cold) -> yellow -> red (hot)
        if (normalized_mass < 0.5f) {
            float t = normalized_mass * 2.0f;
            colors_[i * 3 + 0] = t;           // R
            colors_[i * 3 + 1] = t * 0.8f;    // G
            colors_[i * 3 + 2] = 1.0f - t;    // B
        } else {
            float t = (normalized_mass - 0.5f) * 2.0f;
            colors_[i * 3 + 0] = 1.0f;        // R
            colors_[i * 3 + 1] = 1.0f - t;    // G
            colors_[i * 3 + 2] = 0.0f;        // B
        }
    }

    // Update buffers
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numBodies * 3 * sizeof(float), positions);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_color_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numBodies * 3 * sizeof(float), colors_);

    delete[] positions;

    // Setup matrices
    float projection[16], view[16];

    float aspect = (float)width_ / (float)height_;
    matrixPerspective(projection, camera_.fov, aspect, 0.1f, 500.0f);

    // Calculate camera position
    float radY = camera_.rotation_y * M_PI / 180.0f;
    float radX = camera_.rotation_x * M_PI / 180.0f;

    float eyeX = camera_.distance * cosf(radX) * sinf(radY) + camera_.pan_x;
    float eyeY = camera_.distance * sinf(radX) + camera_.pan_y;
    float eyeZ = camera_.distance * cosf(radX) * cosf(radY);

    matrixLookAt(view, eyeX, eyeY, eyeZ, camera_.pan_x, camera_.pan_y, 0.0f, 0.0f, 1.0f, 0.0f);

    // Use shader and set uniforms
    glUseProgram(shader_program_);

    int projLoc = glGetUniformLocation(shader_program_, "projection");
    int viewLoc = glGetUniformLocation(shader_program_, "view");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);

    // Draw particles
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, numBodies);
    glBindVertexArray(0);

    // Calculate FPS
    frame_count_++;
    double current_time = glfwGetTime();
    if (current_time - last_frame_time_ >= 1.0) {
        fps_ = frame_count_ / (current_time - last_frame_time_);

        char title[256];
        snprintf(title, sizeof(title), "%s | FPS: %.1f | Bodies: %d",
                title_.c_str(), fps_, numBodies);
        glfwSetWindowTitle(window_, title);

        frame_count_ = 0;
        last_frame_time_ = current_time;
    }
}

void Renderer::updateCamera(float deltaTime) {
    // Camera update logic (if needed for animation)
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void Renderer::swapBuffers() {
    glfwSwapBuffers(window_);
    glfwPollEvents();
}
