#include "nbody.h"
#include "cuda_utils.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================================
// Host Memory Management
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

// ============================================================================
// Body Initialization Functions
// ============================================================================

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
        // Random position within sphere
        float theta = 2.0f * M_PI * uniform(gen);
        float phi = std::acos(2.0f * uniform(gen) - 1.0f);
        float r = radius * std::cbrt(uniform(gen));

        bodies->x[i] = r * std::sin(phi) * std::cos(theta);
        bodies->y[i] = r * std::sin(phi) * std::sin(theta);
        bodies->z[i] = r * std::cos(phi);

        // Small random velocity
        bodies->vx[i] = normal(gen);
        bodies->vy[i] = normal(gen);
        bodies->vz[i] = normal(gen);

        // Uniform mass
        bodies->mass[i] = 1.0f;
    }
}

void initializeGalaxyCollision(Bodies* bodies) {
    int half = bodies->count / 2;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> pos_dist(0.0f, 1.0f);
    std::normal_distribution<float> vel_dist(0.0f, 0.5f);

    // Galaxy 1 (left)
    for (int i = 0; i < half; ++i) {
        bodies->x[i] = pos_dist(gen) - 5.0f;
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen) + 1.0f;  // Moving right
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = 1.0f;
    }

    // Galaxy 2 (right)
    for (int i = half; i < bodies->count; ++i) {
        bodies->x[i] = pos_dist(gen) + 5.0f;
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen) - 1.0f;  // Moving left
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = 1.0f;
    }
}

// ============================================================================
// Statistics and Validation
// ============================================================================

float computeTotalEnergy(const Bodies* bodies) {
    float kineticEnergy = 0.0f;
    float potentialEnergy = 0.0f;

    // Kinetic energy: KE = 0.5 * m * v²
    for (int i = 0; i < bodies->count; ++i) {
        float vSqr = bodies->vx[i] * bodies->vx[i] +
                    bodies->vy[i] * bodies->vy[i] +
                    bodies->vz[i] * bodies->vz[i];
        kineticEnergy += 0.5f * bodies->mass[i] * vSqr;
    }

    // Potential energy: PE = -G * m1 * m2 / r
    for (int i = 0; i < bodies->count; ++i) {
        for (int j = i + 1; j < bodies->count; ++j) {
            float dx = bodies->x[j] - bodies->x[i];
            float dy = bodies->y[j] - bodies->y[i];
            float dz = bodies->z[j] - bodies->z[i];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0.0f) {
                potentialEnergy -= bodies->mass[i] * bodies->mass[j] / dist;
            }
        }
    }

    return kineticEnergy + potentialEnergy;
}

void printStatistics(const Bodies* bodies) {
    // Compute center of mass
    float cmx = 0.0f, cmy = 0.0f, cmz = 0.0f;
    float totalMass = 0.0f;

    for (int i = 0; i < bodies->count; ++i) {
        cmx += bodies->mass[i] * bodies->x[i];
        cmy += bodies->mass[i] * bodies->y[i];
        cmz += bodies->mass[i] * bodies->z[i];
        totalMass += bodies->mass[i];
    }

    cmx /= totalMass;
    cmy /= totalMass;
    cmz /= totalMass;

    // Compute average velocity
    float avgVx = 0.0f, avgVy = 0.0f, avgVz = 0.0f;
    for (int i = 0; i < bodies->count; ++i) {
        avgVx += bodies->vx[i];
        avgVy += bodies->vy[i];
        avgVz += bodies->vz[i];
    }
    avgVx /= bodies->count;
    avgVy /= bodies->count;
    avgVz /= bodies->count;

    float energy = computeTotalEnergy(bodies);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "System Statistics:\n";
    std::cout << "  Total Mass: " << totalMass << "\n";
    std::cout << "  Center of Mass: (" << cmx << ", " << cmy << ", " << cmz << ")\n";
    std::cout << "  Average Velocity: (" << avgVx << ", " << avgVy << ", " << avgVz << ")\n";
    std::cout << "  Total Energy: " << energy << "\n";
}

// ============================================================================
// Benchmark Function
// ============================================================================

double benchmarkKernel(void (*kernel)(DeviceBodies&, int, float, float),
                      DeviceBodies& bodies, int numBodies,
                      float dt, float softening, int iterations) {
    CudaTimer timer;

    // Warmup
    for (int i = 0; i < 3; ++i) {
        kernel(bodies, numBodies, dt, softening);
    }

    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        kernel(bodies, numBodies, dt, softening);
    }
    timer.stop();

    float totalMs = timer.elapsed();
    double avgMs = totalMs / iterations;

    // Compute interactions per second
    double interactions = static_cast<double>(numBodies) * numBodies;
    double interactionsPerSec = (interactions * iterations) / (totalMs / 1000.0);

    return avgMs;
}

// ============================================================================
// Main Function
// ============================================================================

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n <num>        Number of bodies (default: 4096)\n";
    std::cout << "  -i <num>        Number of iterations (default: 100)\n";
    std::cout << "  -dt <float>     Time step (default: 0.01)\n";
    std::cout << "  -s <float>      Softening factor (default: 0.1)\n";
    std::cout << "  -mode <name>    Kernel mode: naive, shared, tiled, all (default: all)\n";
    std::cout << "  -init <name>    Initialization: random, cluster, galaxy (default: random)\n";
    std::cout << "  -h, --help      Show this help message\n";
}

int main(int argc, char** argv) {
    // Default parameters
    int numBodies = 4096;
    int numIterations = 100;
    float dt = 0.01f;
    float softening = 0.1f;
    std::string mode = "all";
    std::string initMode = "random";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            numIterations = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-dt") == 0 && i + 1 < argc) {
            dt = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            softening = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "-mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "-init") == 0 && i + 1 < argc) {
            initMode = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "GPU N-Body Simulation\n";
    std::cout << "========================================\n\n";

    // Print device information
    printDeviceInfo();

    std::cout << "Simulation Parameters:\n";
    std::cout << "  Number of bodies: " << numBodies << "\n";
    std::cout << "  Time step (dt): " << dt << "\n";
    std::cout << "  Softening factor: " << softening << "\n";
    std::cout << "  Iterations: " << numIterations << "\n";
    std::cout << "  Initialization: " << initMode << "\n\n";

    // Create and initialize bodies
    Bodies* hostBodies = createBodies(numBodies);

    if (initMode == "cluster") {
        initializeSphericalCluster(hostBodies, 10.0f);
    } else if (initMode == "galaxy") {
        initializeGalaxyCollision(hostBodies);
    } else {
        initializeRandomBodies(hostBodies, 10.0f, 1.0f, 1.0f);
    }

    std::cout << "Initial State:\n";
    printStatistics(hostBodies);
    std::cout << "\n";

    // Create device memory
    DeviceBodies* deviceBodies = createDeviceBodies(numBodies);

    // Benchmark results
    std::cout << "========================================\n";
    std::cout << "Performance Benchmarks\n";
    std::cout << "========================================\n\n";

    if (mode == "naive" || mode == "all") {
        copyBodiesToDevice(hostBodies, deviceBodies);
        double avgMs = benchmarkKernel(NBodyKernels::computeNaive,
                                      *deviceBodies, numBodies,
                                      dt, softening, numIterations);

        std::cout << "Naive Implementation:\n";
        std::cout << "  Average time per iteration: " << avgMs << " ms\n";
        std::cout << "  Performance: " << (1000.0 / avgMs) << " iterations/sec\n\n";
    }

    if (mode == "shared" || mode == "all") {
        copyBodiesToDevice(hostBodies, deviceBodies);
        double avgMs = benchmarkKernel(NBodyKernels::computeSharedMemory,
                                      *deviceBodies, numBodies,
                                      dt, softening, numIterations);

        std::cout << "Shared Memory Implementation:\n";
        std::cout << "  Average time per iteration: " << avgMs << " ms\n";
        std::cout << "  Performance: " << (1000.0 / avgMs) << " iterations/sec\n\n";
    }

    if (mode == "tiled" || mode == "all") {
        copyBodiesToDevice(hostBodies, deviceBodies);
        double avgMs = benchmarkKernel(NBodyKernels::computeRegisterTiled,
                                      *deviceBodies, numBodies,
                                      dt, softening, numIterations);

        std::cout << "Register-Tiled Implementation:\n";
        std::cout << "  Average time per iteration: " << avgMs << " ms\n";
        std::cout << "  Performance: " << (1000.0 / avgMs) << " iterations/sec\n\n";
    }

    // Copy final state back and print
    copyBodiesFromDevice(hostBodies, deviceBodies);

    std::cout << "========================================\n";
    std::cout << "Final State (after " << numIterations << " iterations):\n";
    std::cout << "========================================\n";
    printStatistics(hostBodies);
    std::cout << "\n";

    // Cleanup
    freeBodies(hostBodies);
    freeDeviceBodies(deviceBodies);

    std::cout << "Simulation completed successfully!\n\n";

    return 0;
}
