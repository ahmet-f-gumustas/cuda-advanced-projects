#ifndef NBODY_H
#define NBODY_H

#include <cuda_runtime.h>
#include <cstddef>

// N-body simulation parameters
struct SimulationParams {
    int numBodies;          // Number of bodies in simulation
    float deltaTime;        // Time step for integration
    float softeningFactor;  // Softening to prevent singularities (ε²)
    int numIterations;      // Number of simulation steps
};

// Body representation: position (x,y,z) and velocity (vx,vy,vz)
// Stored as Structure of Arrays (SoA) for better memory coalescing
struct Bodies {
    float* x;      // Position X coordinates
    float* y;      // Position Y coordinates
    float* z;      // Position Z coordinates
    float* vx;     // Velocity X components
    float* vy;     // Velocity Y components
    float* vz;     // Velocity Z components
    float* mass;   // Mass of each body
    int count;     // Total number of bodies
};

// Device memory bodies
struct DeviceBodies {
    float* d_x;
    float* d_y;
    float* d_z;
    float* d_vx;
    float* d_vy;
    float* d_vz;
    float* d_mass;
    float* d_fx;    // Force accumulation buffers
    float* d_fy;
    float* d_fz;
};

// Kernel implementations
namespace NBodyKernels {
    // Naive O(N²) implementation - straightforward but slow
    void computeNaive(DeviceBodies& bodies, int numBodies,
                     float dt, float softening);

    // Shared memory tile-based implementation - optimized
    void computeSharedMemory(DeviceBodies& bodies, int numBodies,
                            float dt, float softening);

    // Register-tiled implementation - most optimized
    void computeRegisterTiled(DeviceBodies& bodies, int numBodies,
                             float dt, float softening);
}

// Host functions
Bodies* createBodies(int numBodies);
void freeBodies(Bodies* bodies);
DeviceBodies* createDeviceBodies(int numBodies);
void freeDeviceBodies(DeviceBodies* bodies);

void copyBodiesToDevice(const Bodies* host, DeviceBodies* device);
void copyBodiesFromDevice(Bodies* host, const DeviceBodies* device);

void initializeRandomBodies(Bodies* bodies, float positionScale,
                           float velocityScale, float massScale);
void initializeGalaxyCollision(Bodies* bodies);
void initializeSphericalCluster(Bodies* bodies, float radius);

// Benchmark utilities
double benchmarkKernel(void (*kernel)(DeviceBodies&, int, float, float),
                      DeviceBodies& bodies, int numBodies,
                      float dt, float softening, int iterations);

// Energy computation for validation
float computeTotalEnergy(const Bodies* bodies);
void printStatistics(const Bodies* bodies);

#endif // NBODY_H
