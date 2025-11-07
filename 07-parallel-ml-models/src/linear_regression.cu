#include "linear_regression.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for computing predictions: y_pred = w * x + b
__global__ void predictKernel(const float* X, float* predictions,
                               const float* weights, const float* bias,
                               int numSamples, int numFeatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        float pred = *bias;
        for (int f = 0; f < numFeatures; f++) {
            pred += weights[f] * X[idx * numFeatures + f];
        }
        predictions[idx] = pred;
    }
}

// CUDA kernel for computing gradients
__global__ void computeGradientsKernel(const float* X, const float* y,
                                        const float* predictions,
                                        float* gradWeights, float* gradBias,
                                        int numSamples, int numFeatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numFeatures) {
        float grad = 0.0f;
        for (int i = 0; i < numSamples; i++) {
            float error = predictions[i] - y[i];
            grad += error * X[i * numFeatures + idx];
        }
        gradWeights[idx] = grad / numSamples;
    }

    // First thread computes bias gradient
    if (idx == 0) {
        float grad = 0.0f;
        for (int i = 0; i < numSamples; i++) {
            grad += predictions[i] - y[i];
        }
        *gradBias = grad / numSamples;
    }
}

// CUDA kernel for updating weights
__global__ void updateWeightsKernel(float* weights, float* bias,
                                     const float* gradWeights, const float* gradBias,
                                     float learningRate, int numFeatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numFeatures) {
        weights[idx] -= learningRate * gradWeights[idx];
    }

    if (idx == 0) {
        *bias -= learningRate * (*gradBias);
    }
}

// CUDA kernel for computing MSE loss
__global__ void computeLossKernel(const float* predictions, const float* y,
                                   float* loss, int numSamples) {
    extern __shared__ float sharedLoss[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes partial loss
    float localLoss = 0.0f;
    if (idx < numSamples) {
        float error = predictions[idx] - y[idx];
        localLoss = error * error;
    }
    sharedLoss[tid] = localLoss;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedLoss[tid] += sharedLoss[tid + s];
        }
        __syncthreads();
    }

    // First thread writes result
    if (tid == 0) {
        atomicAdd(loss, sharedLoss[0]);
    }
}

LinearRegression::LinearRegression(int numFeatures)
    : numFeatures(numFeatures), trainingComplete(false) {

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights, numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, sizeof(float)));

    // Allocate host memory
    h_weights = new float[numFeatures];
    h_bias = 0.0f;

    // Initialize weights to small random values
    for (int i = 0; i < numFeatures; i++) {
        h_weights[i] = 0.01f * (rand() % 100 - 50) / 100.0f;
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, numFeatures * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, &h_bias, sizeof(float), cudaMemcpyHostToDevice));

    // Create stream for asynchronous operations
    CUDA_CHECK(cudaStreamCreate(&stream));
}

LinearRegression::~LinearRegression() {
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    delete[] h_weights;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void LinearRegression::train(const std::vector<float>& X, const std::vector<float>& y,
                              int numSamples, int numIterations, float learningRate) {

    // Allocate device memory for data
    float *d_X, *d_y, *d_predictions, *d_gradWeights, *d_gradBias, *d_loss;
    CUDA_CHECK(cudaMalloc(&d_X, numSamples * numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predictions, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradWeights, numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradBias, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_X, X.data(), numSamples * numFeatures * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, y.data(), numSamples * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    lossHistory.clear();

    // Training loop
    for (int iter = 0; iter < numIterations; iter++) {
        // Forward pass: compute predictions
        predictKernel<<<gridSize, blockSize, 0, stream>>>(
            d_X, d_predictions, d_weights, d_bias, numSamples, numFeatures);

        // Compute gradients
        int gradGridSize = (numFeatures + blockSize - 1) / blockSize;
        computeGradientsKernel<<<gradGridSize, blockSize, 0, stream>>>(
            d_X, d_y, d_predictions, d_gradWeights, d_gradBias, numSamples, numFeatures);

        // Update weights
        updateWeightsKernel<<<gradGridSize, blockSize, 0, stream>>>(
            d_weights, d_bias, d_gradWeights, d_gradBias, learningRate, numFeatures);

        // Compute loss every 10 iterations
        if (iter % 10 == 0) {
            float h_loss = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(d_loss, &h_loss, sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

            computeLossKernel<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
                d_predictions, d_y, d_loss, numSamples);

            CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_loss, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            lossHistory.push_back(h_loss / numSamples);
        }
    }

    // Copy final weights back to host
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, numFeatures * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_gradWeights));
    CUDA_CHECK(cudaFree(d_gradBias));
    CUDA_CHECK(cudaFree(d_loss));

    trainingComplete = true;
}

std::vector<float> LinearRegression::predict(const std::vector<float>& X, int numSamples) {
    // Allocate device memory
    float *d_X, *d_predictions;
    CUDA_CHECK(cudaMalloc(&d_X, numSamples * numFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predictions, numSamples * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_X, X.data(), numSamples * numFeatures * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;
    predictKernel<<<gridSize, blockSize, 0, stream>>>(
        d_X, d_predictions, d_weights, d_bias, numSamples, numFeatures);

    // Copy results back
    std::vector<float> predictions(numSamples);
    CUDA_CHECK(cudaMemcpyAsync(predictions.data(), d_predictions,
                               numSamples * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Clean up
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_predictions));

    return predictions;
}
