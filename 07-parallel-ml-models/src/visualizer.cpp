#include "visualizer.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>

Visualizer::Visualizer(ModelManager* manager, int width, int height)
    : modelManager(manager), window(nullptr),
      windowWidth(width), windowHeight(height),
      isRunning(false), time(0.0f) {
}

Visualizer::~Visualizer() {
    cleanup();
}

bool Visualizer::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    // Create window - Use compatibility profile for legacy OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    window = glfwCreateWindow(windowWidth, windowHeight,
                              "CUDA Parallel ML Models - Linear Regression & K-Means",
                              nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, keyCallback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return false;
    }

    // OpenGL settings
    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(3.0f);
    glLineWidth(2.0f);

    // Setup projection matrix for 2D rendering
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    isRunning = true;
    std::cout << "Visualization initialized successfully!\n";
    std::cout << "Press ESC to exit\n\n";

    return true;
}

void Visualizer::run() {
    while (isRunning && !glfwWindowShouldClose(window)) {
        time += 0.016f;  // Approximately 60 FPS

        // Poll events
        glfwPollEvents();

        // Render
        render();

        // Swap buffers
        glfwSwapBuffers(window);

        // Wait for models to finish if they are training
        if (!modelManager->areModelsTraining()) {
            // Small delay after training completes
            static bool displayed = false;
            if (!displayed) {
                std::cout << "\n========================================\n";
                std::cout << "ALL MODELS TRAINED SUCCESSFULLY!\n";
                std::cout << "========================================\n";
                displayed = true;
            }
        }
    }
}

void Visualizer::render() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    // Draw separator line
    glColor3f(0.3f, 0.3f, 0.35f);
    glBegin(GL_LINES);
    glVertex2f(0.0f, -1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    // Render both panels
    renderLinearRegression();
    renderKMeans();
}

void Visualizer::renderLinearRegression() {
    LinearRegression* model = modelManager->getLinearRegression();
    if (!model) return;

    // Title
    glColor3f(1.0f, 1.0f, 1.0f);

    // Draw sample training data points (subsample for visualization)
    std::vector<float> sampleX, sampleY;
    for (int i = 0; i < 100; i++) {
        sampleX.push_back(i * 0.1f);
        sampleY.push_back(2.5f * i * 0.1f + 1.0f + (rand() % 100 - 50) / 100.0f);
    }

    // Draw data points
    glPointSize(4.0f);
    for (size_t i = 0; i < sampleX.size(); i++) {
        float sx, sy;
        dataToScreen(sampleX[i], sampleY[i], sx, sy, true);
        drawPoint(sx, sy, 0.3f, 0.7f, 1.0f, 4.0f);
    }

    // Draw regression line if model is trained
    if (model->isTrainingComplete()) {
        float slope = model->getSlope();
        float intercept = model->getIntercept();

        float x1 = 0.0f, y1 = intercept;
        float x2 = 10.0f, y2 = slope * x2 + intercept;

        float sx1, sy1, sx2, sy2;
        dataToScreen(x1, y1, sx1, sy1, true);
        dataToScreen(x2, y2, sx2, sy2, true);

        drawLine(sx1, sy1, sx2, sy2, 1.0f, 0.3f, 0.3f, 3.0f);

        // Draw legend
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "y = " << slope << "x + " << intercept;

        // Draw equation text (simplified - just showing as colored box)
        glColor3f(1.0f, 0.3f, 0.3f);
        glBegin(GL_QUADS);
        glVertex2f(-0.95f, 0.85f);
        glVertex2f(-0.55f, 0.85f);
        glVertex2f(-0.55f, 0.75f);
        glVertex2f(-0.95f, 0.75f);
        glEnd();
    }

    // Draw training status
    if (!model->isTrainingComplete()) {
        // Animated training indicator
        float pulse = 0.5f + 0.5f * sin(time * 3.0f);
        drawCircle(-0.75f, -0.8f, 0.05f, 1.0f, pulse, 0.0f);
    } else {
        // Completion indicator
        drawCircle(-0.75f, -0.8f, 0.05f, 0.0f, 1.0f, 0.0f);
    }

    // Draw axes
    glColor3f(0.4f, 0.4f, 0.45f);
    glBegin(GL_LINES);
    // X-axis
    glVertex2f(-0.95f, 0.0f);
    glVertex2f(-0.05f, 0.0f);
    // Y-axis
    glVertex2f(-0.5f, -0.9f);
    glVertex2f(-0.5f, 0.9f);
    glEnd();
}

void Visualizer::renderKMeans() {
    KMeans* model = modelManager->getKMeans();
    if (!model) return;

    // Generate visualization data
    std::vector<float> data;
    std::vector<int> labels;

    // Create sample data for visualization
    int samplesPerCluster = 100;
    float centers[3][2] = {
        {2.0f, 2.0f},
        {6.0f, 6.0f},
        {2.0f, 8.0f}
    };

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < samplesPerCluster; i++) {
            float x = centers[c][0] + (rand() % 100 - 50) / 62.5f;
            float y = centers[c][1] + (rand() % 100 - 50) / 62.5f;
            data.push_back(x);
            data.push_back(y);
        }
    }

    // Get predictions if model is trained
    if (model->isTrainingComplete()) {
        labels = model->predict(data, data.size() / 2);
    }

    // Draw data points
    glPointSize(5.0f);
    float colors[3][3] = {
        {1.0f, 0.3f, 0.3f},  // Red
        {0.3f, 1.0f, 0.3f},  // Green
        {0.3f, 0.3f, 1.0f}   // Blue
    };

    for (size_t i = 0; i < data.size() / 2; i++) {
        float sx, sy;
        dataToScreen(data[i * 2], data[i * 2 + 1], sx, sy, false);

        int cluster = model->isTrainingComplete() ? labels[i] : (i / samplesPerCluster);
        drawPoint(sx, sy, colors[cluster][0], colors[cluster][1], colors[cluster][2], 5.0f);
    }

    // Draw centroids if model is trained
    if (model->isTrainingComplete()) {
        std::vector<float> centroids = model->getCentroids();
        glPointSize(15.0f);

        for (int c = 0; c < 3; c++) {
            float sx, sy;
            dataToScreen(centroids[c * 2], centroids[c * 2 + 1], sx, sy, false);

            // Draw centroid as large point with border
            drawCircle(sx, sy, 0.04f, 1.0f, 1.0f, 1.0f);
            drawCircle(sx, sy, 0.03f, colors[c][0], colors[c][1], colors[c][2]);
        }
    }

    // Draw training status
    if (!model->isTrainingComplete()) {
        float pulse = 0.5f + 0.5f * sin(time * 3.0f);
        drawCircle(0.25f, -0.8f, 0.05f, 1.0f, pulse, 0.0f);
    } else {
        drawCircle(0.25f, -0.8f, 0.05f, 0.0f, 1.0f, 0.0f);
    }

    // Draw axes
    glColor3f(0.4f, 0.4f, 0.45f);
    glBegin(GL_LINES);
    // X-axis
    glVertex2f(0.05f, 0.0f);
    glVertex2f(0.95f, 0.0f);
    // Y-axis
    glVertex2f(0.5f, -0.9f);
    glVertex2f(0.5f, 0.9f);
    glEnd();
}

void Visualizer::drawPoint(float x, float y, float r, float g, float b, float size) {
    glPointSize(size);
    glColor3f(r, g, b);
    glBegin(GL_POINTS);
    glVertex2f(x, y);
    glEnd();
}

void Visualizer::drawLine(float x1, float y1, float x2, float y2,
                          float r, float g, float b, float width) {
    glLineWidth(width);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

void Visualizer::drawCircle(float x, float y, float radius, float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);
    int segments = 32;
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        glVertex2f(x + radius * cos(angle), y + radius * sin(angle));
    }
    glEnd();
}

void Visualizer::dataToScreen(float dataX, float dataY, float& screenX, float& screenY, bool leftPanel) {
    // Normalize data to [0, 1]
    float normX = dataX / 10.0f;
    float normY = dataY / 30.0f;

    // Map to screen coordinates [-1, 1]
    if (leftPanel) {
        screenX = -0.95f + normX * 0.9f;
        screenY = -0.85f + normY * 1.7f;
    } else {
        screenX = 0.05f + normX * 0.9f;
        screenY = -0.85f + normY * 1.7f;
    }
}

void Visualizer::cleanup() {
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

void Visualizer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        Visualizer* viz = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
        viz->isRunning = false;
    }
}
