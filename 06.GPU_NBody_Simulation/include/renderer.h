#ifndef RENDERER_H
#define RENDERER_H

#include <GLFW/glfw3.h>
#include <string>
#include <memory>

// Camera for 3D view
struct Camera {
    float distance;      // Distance from origin
    float rotation_x;    // Rotation around X axis (up/down)
    float rotation_y;    // Rotation around Y axis (left/right)
    float pan_x;         // Pan offset X
    float pan_y;         // Pan offset Y
    float fov;           // Field of view

    Camera() : distance(50.0f), rotation_x(30.0f), rotation_y(45.0f),
               pan_x(0.0f), pan_y(0.0f), fov(45.0f) {}
};

// Mouse state for interaction
struct MouseState {
    double last_x;
    double last_y;
    bool left_pressed;
    bool right_pressed;
    bool first_mouse;

    MouseState() : last_x(0), last_y(0), left_pressed(false),
                   right_pressed(false), first_mouse(true) {}
};

// Renderer class for OpenGL visualization
class Renderer {
public:
    Renderer(int width, int height, const std::string& title);
    ~Renderer();

    // Initialize OpenGL context and resources
    bool initialize();

    // Render bodies
    void render(const float* x, const float* y, const float* z,
                const float* mass, int numBodies);

    // Update camera and handle input
    void updateCamera(float deltaTime);

    // Check if window should close
    bool shouldClose() const;

    // Swap buffers and poll events
    void swapBuffers();

    // Get window pointer
    GLFWwindow* getWindow() const { return window_; }

    // Camera access
    Camera& getCamera() { return camera_; }
    const Camera& getCamera() const { return camera_; }

    // Mouse state access
    MouseState& getMouseState() { return mouse_state_; }

private:
    // Compile shader
    unsigned int compileShader(const char* source, unsigned int type);

    // Create shader program
    unsigned int createShaderProgram(const char* vertexSrc, const char* fragmentSrc);

    // Setup vertex buffers
    void setupBuffers();

    // Window and rendering
    GLFWwindow* window_;
    int width_;
    int height_;
    std::string title_;

    // OpenGL objects
    unsigned int shader_program_;
    unsigned int vao_;
    unsigned int vbo_position_;
    unsigned int vbo_color_;

    // Camera and interaction
    Camera camera_;
    MouseState mouse_state_;

    // Colors for visualization
    float* colors_;
    int max_bodies_;

    // Performance
    double last_frame_time_;
    double fps_;
    int frame_count_;
};

// Global callback functions (need to be static or global for GLFW)
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

#endif // RENDERER_H
