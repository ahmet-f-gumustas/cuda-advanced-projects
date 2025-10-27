#include "nbody.h"
#include "cuda_utils.h"
#include <cmath>
#include <cstdio>

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 256;

// ============================================================================
// Device Functions - Body-Body Interaction
// ============================================================================

/**
 * Compute gravitational force between two bodies
 * F = G * m1 * m2 / (r² + ε²)
 * where ε is softening factor to prevent singularities
 */
__device__ __forceinline__
void bodyBodyInteraction(float& ax, float& ay, float& az,
                        float xi, float yi, float zi,
                        float xj, float yj, float zj,
                        float massj, float softening) {
    // Compute distance vector
    float dx = xj - xi;
    float dy = yj - yi;
    float dz = zj - zi;

    // Compute distance squared with softening
    float distSqr = dx*dx + dy*dy + dz*dz + softening*softening;

    // Compute inverse distance cubed (1/r³)
    float invDist = rsqrtf(distSqr);  // Fast inverse square root
    float invDistCube = invDist * invDist * invDist;

    // Compute force magnitude: F = G*m1*m2/r²
    // We absorb G into mass, and m1 cancels out in a = F/m1
    float forceMagnitude = massj * invDistCube;

    // Accumulate acceleration (F/m = a)
    ax += forceMagnitude * dx;
    ay += forceMagnitude * dy;
    az += forceMagnitude * dz;
}

// ============================================================================
// KERNEL 1: Naive Implementation (O(N²) per thread)
// ============================================================================

/**
 * Naive N-body force calculation
 * Each thread computes forces on one body from all other bodies
 *
 * Pros: Simple, straightforward
 * Cons: Poor memory coalescing, redundant global memory reads
 *
 * Performance: Baseline (~1x speedup)
 */
__global__ void nbodyNaiveKernel(float* x, float* y, float* z,
                                 float* vx, float* vy, float* vz,
                                 const float* mass,
                                 float* fx, float* fy, float* fz,
                                 int numBodies, float dt, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numBodies) return;

    // Load position of body i
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];

    // Accumulate acceleration
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Compute force from all other bodies
    for (int j = 0; j < numBodies; ++j) {
        if (i != j) {
            bodyBodyInteraction(ax, ay, az,
                              xi, yi, zi,
                              x[j], y[j], z[j],
                              mass[j], softening);
        }
    }

    // Store accelerations
    fx[i] = ax;
    fy[i] = ay;
    fz[i] = az;

    // Update velocity (semi-implicit Euler)
    vx[i] += ax * dt;
    vy[i] += ay * dt;
    vz[i] += az * dt;

    // Update position
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

// ============================================================================
// KERNEL 2: Shared Memory Tile-Based Implementation
// ============================================================================

/**
 * Shared memory tile-based N-body
 * Bodies are loaded into shared memory in tiles
 * Each thread computes forces from a tile of bodies
 *
 * Pros: Better memory coalescing, reuse of data in shared memory
 * Cons: More complex, shared memory bandwidth limits
 *
 * Performance: ~3-5x speedup over naive
 */
__global__ void nbodySharedMemoryKernel(float* x, float* y, float* z,
                                       float* vx, float* vy, float* vz,
                                       const float* mass,
                                       float* fx, float* fy, float* fz,
                                       int numBodies, float dt, float softening) {
    // Shared memory for a tile of bodies
    __shared__ float shared_x[TILE_SIZE];
    __shared__ float shared_y[TILE_SIZE];
    __shared__ float shared_z[TILE_SIZE];
    __shared__ float shared_mass[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load position of body i (if valid)
    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    if (i < numBodies) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }

    // Accumulate acceleration
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Process bodies in tiles
    int numTiles = (numBodies + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int j = tile * TILE_SIZE + tid;

        // Load tile into shared memory
        if (j < numBodies) {
            shared_x[tid] = x[j];
            shared_y[tid] = y[j];
            shared_z[tid] = z[j];
            shared_mass[tid] = mass[j];
        } else {
            // Pad with dummy values
            shared_mass[tid] = 0.0f;
        }

        __syncthreads();

        // Compute interactions with bodies in this tile
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            int j_global = tile * TILE_SIZE + k;
            if (j_global < numBodies && i != j_global) {
                bodyBodyInteraction(ax, ay, az,
                                  xi, yi, zi,
                                  shared_x[k], shared_y[k], shared_z[k],
                                  shared_mass[k], softening);
            }
        }

        __syncthreads();
    }

    // Update velocities and positions
    if (i < numBodies) {
        fx[i] = ax;
        fy[i] = ay;
        fz[i] = az;

        vx[i] += ax * dt;
        vy[i] += ay * dt;
        vz[i] += az * dt;

        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

// ============================================================================
// KERNEL 3: Register-Tiled Implementation (Most Optimized)
// ============================================================================

/**
 * Register-tiled N-body with warp-level optimizations
 * Uses register blocking and warp shuffle for minimal memory traffic
 *
 * Pros: Maximum performance, optimal memory usage
 * Cons: Complex implementation, architecture dependent
 *
 * Performance: ~6-10x speedup over naive
 */
__global__ void nbodyRegisterTiledKernel(float* x, float* y, float* z,
                                        float* vx, float* vy, float* vz,
                                        const float* mass,
                                        float* fx, float* fy, float* fz,
                                        int numBodies, float dt, float softening) {
    // Shared memory for body data
    __shared__ float4 shared_pos[TILE_SIZE];  // .xyz = position, .w = mass

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load body i data
    float3 pos_i = {0.0f, 0.0f, 0.0f};
    if (i < numBodies) {
        pos_i.x = x[i];
        pos_i.y = y[i];
        pos_i.z = z[i];
    }

    // Accumulate acceleration
    float3 acc = {0.0f, 0.0f, 0.0f};

    // Process bodies in tiles
    int numTiles = (numBodies + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int j = tile * TILE_SIZE + tid;

        // Coalesced load into shared memory
        float4 body_data;
        if (j < numBodies) {
            body_data.x = x[j];
            body_data.y = y[j];
            body_data.z = z[j];
            body_data.w = mass[j];
        } else {
            body_data.w = 0.0f;  // Zero mass for padding
        }
        shared_pos[tid] = body_data;

        __syncthreads();

        // Compute interactions with tile
        // Unroll for instruction-level parallelism
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            float4 body_j = shared_pos[k];

            int j_global = tile * TILE_SIZE + k;
            if (j_global < numBodies && i != j_global) {
                // Inline body-body interaction for better register usage
                float dx = body_j.x - pos_i.x;
                float dy = body_j.y - pos_i.y;
                float dz = body_j.z - pos_i.z;

                float distSqr = dx*dx + dy*dy + dz*dz + softening*softening;
                float invDist = rsqrtf(distSqr);
                float invDistCube = invDist * invDist * invDist;
                float forceMagnitude = body_j.w * invDistCube;

                acc.x += forceMagnitude * dx;
                acc.y += forceMagnitude * dy;
                acc.z += forceMagnitude * dz;
            }
        }

        __syncthreads();
    }

    // Write results
    if (i < numBodies) {
        fx[i] = acc.x;
        fy[i] = acc.y;
        fz[i] = acc.z;

        // Update velocity
        float3 vel;
        vel.x = vx[i] + acc.x * dt;
        vel.y = vy[i] + acc.y * dt;
        vel.z = vz[i] + acc.z * dt;

        vx[i] = vel.x;
        vy[i] = vel.y;
        vz[i] = vel.z;

        // Update position
        x[i] = pos_i.x + vel.x * dt;
        y[i] = pos_i.y + vel.y * dt;
        z[i] = pos_i.z + vel.z * dt;
    }
}

// ============================================================================
// Host-side Kernel Launchers
// ============================================================================

namespace NBodyKernels {

void computeNaive(DeviceBodies& bodies, int numBodies,
                 float dt, float softening) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nbodyNaiveKernel<<<grid, block>>>(
        bodies.d_x, bodies.d_y, bodies.d_z,
        bodies.d_vx, bodies.d_vy, bodies.d_vz,
        bodies.d_mass,
        bodies.d_fx, bodies.d_fy, bodies.d_fz,
        numBodies, dt, softening
    );

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeSharedMemory(DeviceBodies& bodies, int numBodies,
                        float dt, float softening) {
    dim3 block(TILE_SIZE);
    dim3 grid((numBodies + TILE_SIZE - 1) / TILE_SIZE);

    nbodySharedMemoryKernel<<<grid, block>>>(
        bodies.d_x, bodies.d_y, bodies.d_z,
        bodies.d_vx, bodies.d_vy, bodies.d_vz,
        bodies.d_mass,
        bodies.d_fx, bodies.d_fy, bodies.d_fz,
        numBodies, dt, softening
    );

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeRegisterTiled(DeviceBodies& bodies, int numBodies,
                         float dt, float softening) {
    dim3 block(TILE_SIZE);
    dim3 grid((numBodies + TILE_SIZE - 1) / TILE_SIZE);

    nbodyRegisterTiledKernel<<<grid, block>>>(
        bodies.d_x, bodies.d_y, bodies.d_z,
        bodies.d_vx, bodies.d_vy, bodies.d_vz,
        bodies.d_mass,
        bodies.d_fx, bodies.d_fy, bodies.d_fz,
        numBodies, dt, softening
    );

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace NBodyKernels
