#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "model_manager.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

class Visualizer {
public:
    Visualizer(ModelManager* manager, int width = 1600, int height = 800);
    ~Visualizer();

    bool initialize();
    void run();
    void cleanup();

private:
    ModelManager* modelManager;
    GLFWwindow* window;
    int windowWidth;
    int windowHeight;

    // OpenGL state
    bool isRunning;
    float time;

    // Rendering methods
    void render();
    void renderLinearRegression();
    void renderKMeans();
    void renderText(float x, float y, const std::string& text);

    // Helper methods
    void drawPoint(float x, float y, float r, float g, float b, float size = 3.0f);
    void drawLine(float x1, float y1, float x2, float y2, float r, float g, float b, float width = 2.0f);
    void drawCircle(float x, float y, float radius, float r, float g, float b);

    // Coordinate transformation (data space to screen space)
    void dataToScreen(float dataX, float dataY, float& screenX, float& screenY, bool leftPanel);

    // Callbacks
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
};

#endif // VISUALIZER_H
