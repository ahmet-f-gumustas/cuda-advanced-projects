#include "nbody.h"
#include "cuda_utils.h"
#include <random>
#include <cmath>

// ============================================================================
// Helper functions implementation for Python bindings
// These are copies of the functions from main.cu for standalone compilation
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
    std::normal_distribution<float> pos_dist(0.0f, 1.0f);
    std::normal_distribution<float> vel_dist(0.0f, 0.5f);

    for (int i = 0; i < half; ++i) {
        bodies->x[i] = pos_dist(gen) - 5.0f;
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen) + 1.0f;
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = 1.0f;
    }

    for (int i = half; i < bodies->count; ++i) {
        bodies->x[i] = pos_dist(gen) + 5.0f;
        bodies->y[i] = pos_dist(gen);
        bodies->z[i] = pos_dist(gen);
        bodies->vx[i] = vel_dist(gen) - 1.0f;
        bodies->vy[i] = vel_dist(gen);
        bodies->vz[i] = vel_dist(gen);
        bodies->mass[i] = 1.0f;
    }
}

float computeTotalEnergy(const Bodies* bodies) {
    float kineticEnergy = 0.0f;
    float potentialEnergy = 0.0f;

    // Kinetic energy
    for (int i = 0; i < bodies->count; ++i) {
        float vSqr = bodies->vx[i] * bodies->vx[i] +
                    bodies->vy[i] * bodies->vy[i] +
                    bodies->vz[i] * bodies->vz[i];
        kineticEnergy += 0.5f * bodies->mass[i] * vSqr;
    }

    // Potential energy
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

// Device memory management
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

    return avgMs;
}
