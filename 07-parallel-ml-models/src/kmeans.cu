#include "kmeans.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <limits>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for computing distances and assigning clusters
__global__ void assignClustersKernel(const float* X, const float* centroids,
                                      int* labels, float* distances,
                                      int numSamples, int numFeatures, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        float minDist = INFINITY;
        int bestCluster = 0;

        // Find nearest centroid
        for (int c = 0; c < numClusters; c++) {
            float dist = 0.0f;
            for (int f = 0; f < numFeatures; f++) {
                float diff = X[idx * numFeatures + f] - centroids[c * numFeatures + f];
                dist += diff * diff;
            }

            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }

        labels[idx] = bestCluster;
        distances[idx] = minDist;
    }
}

// CUDA kernel for computing new centroids
__global__ void updateCentroidsKernel(const float* X, const int* labels,
                                       float* newCentroids, int* clusterCounts,
                                       int numSamples, int numFeatures, int numClusters) {
    int featureIdx = blockIdx.x;  // Each block handles one feature
    int clusterIdx = threadIdx.x; // Each thread handles one cluster

    if (featureIdx < numFeatures && clusterIdx < numClusters) {
        float sum = 0.0f;
        int count = 0;

        // Sum all points belonging to this cluster for this feature
        for (int i = 0; i < numSamples; i++) {
            if (labels[i] == clusterIdx) {
                sum += X[i * numFeatures + featureIdx];
                if (featureIdx == 0) {  // Count only once per cluster
                    count++;
                }
            }
        }

        // Store count (only for first feature to avoid redundancy)
        if (featureIdx == 0) {
            clusterCounts[clusterIdx] = count;
        }

        // Compute mean (avoid division by zero)
        if (count > 0) {
            newCentroids[clusterIdx * numFeatures + featureIdx] = sum / count;
        }
    }
}

// CUDA kernel for computing inertia
__global__ void computeInertiaKernel(const float* distances, float* inertia, int numSamples) {
    extern __shared__ float sharedInertia[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes partial inertia
    float localInertia = 0.0f;
    if (idx < numSamples) {
        localInertia = distances[idx];
    }
    sharedInertia[tid] = localInertia;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedInertia[tid] += sharedInertia[tid + s];
        }
        __syncthreads();
    }

    // First thread writes result
    if (tid == 0) {
        atomicAdd(inertia, sharedInertia[0]);
    }
}

KMeans::KMeans(int numClusters, int numFeatures)
    : numClusters(numClusters), numFeatures(numFeatures),
      inertia(0.0f), trainingComplete(false) {

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_centroids, numClusters * numFeatures * sizeof(float)));

    // Allocate host memory
    h_centroids = new float[numClusters * numFeatures];

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
}

KMeans::~KMeans() {
    CUDA_CHECK(cudaFree(d_centroids));
    delete[] h_centroids;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void KMeans::train(const std::vector<float>& X, int numSamples, int maxIterations) {
    // Allocate device memory
    float *d_X, *d_distances;
    int *d_labels, *d_clusterCounts;
    float *d_newCentroids, *d_inertia;

    CUDA_CHECK(cudaMalloc(&d_X, numSamples * numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, numSamples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clusterCounts, numClusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_newCentroids, numClusters * numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inertia, sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_X, X.data(), numSamples * numFeatures * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // Initialize centroids with first k samples (K-means++ would be better but keeping it simple)
    for (int c = 0; c < numClusters; c++) {
        int sampleIdx = (c * numSamples) / numClusters;
        for (int f = 0; f < numFeatures; f++) {
            h_centroids[c * numFeatures + f] = X[sampleIdx * numFeatures + f];
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_centroids, h_centroids,
                               numClusters * numFeatures * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    inertiaHistory.clear();

    // K-Means iterations
    for (int iter = 0; iter < maxIterations; iter++) {
        // Step 1: Assign clusters
        assignClustersKernel<<<gridSize, blockSize, 0, stream>>>(
            d_X, d_centroids, d_labels, d_distances, numSamples, numFeatures, numClusters);

        // Step 2: Update centroids
        dim3 updateGrid(numFeatures);
        dim3 updateBlock(numClusters);
        updateCentroidsKernel<<<updateGrid, updateBlock, 0, stream>>>(
            d_X, d_labels, d_newCentroids, d_clusterCounts,
            numSamples, numFeatures, numClusters);

        // Copy new centroids
        CUDA_CHECK(cudaMemcpyAsync(d_centroids, d_newCentroids,
                                   numClusters * numFeatures * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        // Compute inertia every 5 iterations
        if (iter % 5 == 0) {
            float h_inertia = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(d_inertia, &h_inertia, sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

            computeInertiaKernel<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
                d_distances, d_inertia, numSamples);

            CUDA_CHECK(cudaMemcpyAsync(&h_inertia, d_inertia, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            inertiaHistory.push_back(h_inertia);
        }
    }

    // Copy final centroids back to host
    CUDA_CHECK(cudaMemcpy(h_centroids, d_centroids,
                          numClusters * numFeatures * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Compute final inertia
    float h_inertia = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_inertia, &h_inertia, sizeof(float), cudaMemcpyHostToDevice));
    assignClustersKernel<<<gridSize, blockSize, 0, stream>>>(
        d_X, d_centroids, d_labels, d_distances, numSamples, numFeatures, numClusters);
    computeInertiaKernel<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
        d_distances, d_inertia, numSamples);
    CUDA_CHECK(cudaMemcpy(&inertia, d_inertia, sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_clusterCounts));
    CUDA_CHECK(cudaFree(d_newCentroids));
    CUDA_CHECK(cudaFree(d_inertia));

    trainingComplete = true;
}

std::vector<int> KMeans::predict(const std::vector<float>& X, int numSamples) {
    // Allocate device memory
    float *d_X, *d_distances;
    int *d_labels;

    CUDA_CHECK(cudaMalloc(&d_X, numSamples * numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, numSamples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, numSamples * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_X, X.data(), numSamples * numFeatures * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;
    assignClustersKernel<<<gridSize, blockSize, 0, stream>>>(
        d_X, d_centroids, d_labels, d_distances, numSamples, numFeatures, numClusters);

    // Copy results back
    std::vector<int> labels(numSamples);
    CUDA_CHECK(cudaMemcpyAsync(labels.data(), d_labels, numSamples * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Clean up
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_distances));

    return labels;
}

std::vector<float> KMeans::getCentroids() const {
    return std::vector<float>(h_centroids, h_centroids + numClusters * numFeatures);
}
