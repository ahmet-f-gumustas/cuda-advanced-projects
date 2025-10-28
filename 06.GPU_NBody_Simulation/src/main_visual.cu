#include "nbody.h"
#include "cuda_utils.h"
#include "renderer.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <cstring>

// Host memory management (same as main.cu)
Bodies* createBodies(int numBodies);
void freeBodies(Bodies* bodies);
DeviceBodies* createDeviceBodies(int numBodies);
void freeDeviceBodies(DeviceBodies* bodies);
void copyBodiesToDevice(const Bodies* host, DeviceBodies* device);
void copyBodiesFromDevice(Bodies* host, const DeviceBodies* device);
void initializeRandomBodies(Bodies* bodies, float positionScale, float velocityScale, float massScale);
void initializeSphericalCluster(Bodies* bodies, float radius);
void initializeGalaxyCollision(Bodies* bodies);

// ============================================================================
// Host Memory Management (Copied from main.cu for standalone compilation)
// ============================================================================

Bodies* createBodies(int numBodies) {
    Bodies* bodies = new Bodies;
    bodies->count = numBodies;
    bodies->x = new float[numBodies];
    bodies->y = new float[numBodies];
    bodies->z = new float[numBodies];
    bodies->vx = new float[numBodies];
    bodies->vy = new float[numBodies];
    bodies->vz = new float[numBodies];
    bodies->mass = new float[numBodies];
    return bodies;
}

void freeBodies(Bodies* bodies) {
    if (bodies) {
        delete[] bodies->x;
        delete[] bodies->y;
        delete[] bodies->z;
        delete[] bodies->vx;
        delete[] bodies->vy;
        delete[] bodies->vz;
        delete[] bodies->mass;
        delete bodies;
    }
}

DeviceBodies* createDeviceBodies(int numBodies) {
    DeviceBodies* bodies = new DeviceBodies;
    bodies->d_x = cudaMallocDevice<float>(numBodies);
    bodies->d_y = cudaMallocDevice<float>(numBodies);
    bodies->d_z = cudaMallocDevice<float>(numBodies);
    bodies->d_vx = cudaMallocDevice<float>(numBodies);
    bodies->d_vy = cudaMallocDevice<float>(numBodies);
    bodies->d_vz = cudaMallocDevice<float>(numBodies);
    bodies->d_mass = cudaMallocDevice<float>(numBodies);
    bodies->d_fx = cudaMallocDevice<float>(numBodies);
    bodies->d_fy = cudaMallocDevice<float>(numBodies);
    bodies->d_fz = cudaMallocDevice<float>(numBodies);
    return bodies;
}

void freeDeviceBodies(DeviceBodies* bodies) {
    if (bodies) {
        cudaFreeWrapper(bodies->d_x);
        cudaFreeWrapper(bodies->d_y);
        cudaFreeWrapper(bodies->d_z);
        cudaFreeWrapper(bodies->d_vx);
        cudaFreeWrapper(bodies->d_vy);
        cudaFreeWrapper(bodies->d_vz);
        cudaFreeWrapper(bodies->d_mass);
        cudaFreeWrapper(bodies->d_fx);
        cudaFreeWrapper(bodies->d_fy);
        cudaFreeWrapper(bodies->d_fz);
        delete bodies;
    }
}

void copyBodiesToDevice(const Bodies* host, DeviceBodies* device) {
    cudaMemcpyH2D(device->d_x, host->x, host->count);
    cudaMemcpyH2D(device->d_y, host->y, host->count);
    cudaMemcpyH2D(device->d_z, host->z, host->count);
    cudaMemcpyH2D(device->d_vx, host->vx, host->count);
    cudaMemcpyH2D(device->d_vy, host->vy, host->count);
    cudaMemcpyH2D(device->d_vz, host->vz, host->count);
    cudaMemcpyH2D(device->d_mass, host->mass, host->count);
}

void copyBodiesFromDevice(Bodies* host, const DeviceBodies* device) {
    cudaMemcpyD2H(host->x, device->d_x, host->count);
    cudaMemcpyD2H(host->y, device->d_y, host->count);
    cudaMemcpyD2H(host->z, device->d_z, host->count);
    cudaMemcpyD2H(host->vx, device->d_vx, host->count);
    cudaMemcpyD2H(host->vy, device->d_vy, host->count);
    cudaMemcpyD2H(host->vz, device->d_vz, host->count);
}

void initializeRandomBodies(Bodies* bodies, float positionScale,
                           float velocityScale, float massScale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-positionScale, positionScale);
    std::uniform_real_distribution<float> vel_dist(-velocityScale, velocityScale);
    std::uniform_real_distribution<float> mass_dist(0.5f * massScale, 1.5f * massScale);

    for (int i = 0; i < bodies->count; ++i) {
        bodies->x[i] = pos_dist(gen);
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen);
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = mass_dist(gen);
    }
}

void initializeSphericalCluster(Bodies* bodies, float radius) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 0.3f);

    for (int i = 0; i < bodies->count; ++i) {
        float theta = 2.0f * M_PI * uniform(gen);
        float phi = std::acos(2.0f * uniform(gen) - 1.0f);
        float r = radius * std::cbrt(uniform(gen));

        bodies->x[i] = r * std::sin(phi) * std::cos(theta);
        bodies->y[i] = r * std::sin(phi) * std::sin(theta);
        bodies->z[i] = r * std::cos(phi);
        bodies->vx[i] = normal(gen);
        bodies->vy[i] = normal(gen);
        bodies->vz[i] = normal(gen);
        bodies->mass[i] = 1.0f;
    }
}

void initializeGalaxyCollision(Bodies* bodies) {
    int half = bodies->count / 2;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> pos_dist(0.0f, 1.5f);
    std::normal_distribution<float> vel_dist(0.0f, 0.3f);

    for (int i = 0; i < half; ++i) {
        bodies->x[i] = pos_dist(gen) - 10.0f;
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen) + 2.0f;
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = 1.0f;
    }

    for (int i = half; i < bodies->count; ++i) {
        bodies->x[i] = pos_dist(gen) + 10.0f;
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen) - 2.0f;
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = 1.0f;
    }
}

// ============================================================================
// Main Visual Application
// ============================================================================

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n <num>        Number of bodies (default: 2048)\n";
    std::cout << "  -dt <float>     Time step (default: 0.01)\n";
    std::cout << "  -s <float>      Softening factor (default: 0.1)\n";
    std::cout << "  -init <name>    Initialization: random, cluster, galaxy (default: galaxy)\n";
    std::cout << "  -width <num>    Window width (default: 1280)\n";
    std::cout << "  -height <num>   Window height (default: 720)\n";
    std::cout << "  -h, --help      Show this help message\n";
}

int main(int argc, char** argv) {
    // Default parameters
    int numBodies = 2048;
    float dt = 0.01f;
    float softening = 0.1f;
    std::string initMode = "galaxy";
    int windowWidth = 1280;
    int windowHeight = 720;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-dt") == 0 && i + 1 < argc) {
            dt = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            softening = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "-init") == 0 && i + 1 < argc) {
            initMode = argv[++i];
        } else if (strcmp(argv[i], "-width") == 0 && i + 1 < argc) {
            windowWidth = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-height") == 0 && i + 1 < argc) {
            windowHeight = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "GPU N-Body Simulation (Visual)\n";
    std::cout << "========================================\n\n";

    // Print device information
    printDeviceInfo();

    std::cout << "Simulation Parameters:\n";
    std::cout << "  Number of bodies: " << numBodies << "\n";
    std::cout << "  Time step (dt): " << dt << "\n";
    std::cout << "  Softening factor: " << softening << "\n";
    std::cout << "  Initialization: " << initMode << "\n";
    std::cout << "  Window size: " << windowWidth << "x" << windowHeight << "\n\n";

    // Create and initialize bodies
    Bodies* hostBodies = createBodies(numBodies);

    if (initMode == "cluster") {
        initializeSphericalCluster(hostBodies, 10.0f);
    } else if (initMode == "galaxy") {
        initializeGalaxyCollision(hostBodies);
    } else {
        initializeRandomBodies(hostBodies, 10.0f, 1.0f, 1.0f);
    }

    // Create device memory
    DeviceBodies* deviceBodies = createDeviceBodies(numBodies);
    copyBodiesToDevice(hostBodies, deviceBodies);

    // Create renderer
    Renderer renderer(windowWidth, windowHeight, "GPU N-Body Simulation");
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return 1;
    }

    std::cout << "Starting real-time simulation...\n";
    std::cout << "Press ESC to exit\n\n";

    // Main simulation loop
    int iteration = 0;
    double lastTime = glfwGetTime();
    double simTime = 0.0;

    while (!renderer.shouldClose()) {
        // Run simulation step (using optimized kernel)
        NBodyKernels::computeRegisterTiled(*deviceBodies, numBodies, dt, softening);

        // Copy data back to host for rendering (every frame)
        copyBodiesFromDevice(hostBodies, deviceBodies);

        // Render
        renderer.render(hostBodies->x, hostBodies->y, hostBodies->z,
                       hostBodies->mass, numBodies);

        renderer.swapBuffers();

        // Update statistics
        iteration++;
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastTime;
        simTime += deltaTime;
        lastTime = currentTime;

        // Print stats every 5 seconds
        if (simTime >= 5.0) {
            double avgFrameTime = simTime / iteration * 1000.0;
            std::cout << "Iteration: " << iteration
                     << " | Avg frame time: " << std::fixed << std::setprecision(2)
                     << avgFrameTime << " ms"
                     << " | FPS: " << (1000.0 / avgFrameTime) << "\n";

            simTime = 0.0;
            iteration = 0;
        }
    }

    std::cout << "\nSimulation stopped.\n";

    // Cleanup
    freeBodies(hostBodies);
    freeDeviceBodies(deviceBodies);

    return 0;
}
