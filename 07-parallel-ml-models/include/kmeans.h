#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cuda_runtime.h>

class KMeans {
public:
    KMeans(int numClusters, int numFeatures);
    ~KMeans();

    // Train the model
    void train(const std::vector<float>& X, int numSamples, int maxIterations);

    // Predict cluster labels for new data
    std::vector<int> predict(const std::vector<float>& X, int numSamples);

    // Get cluster centers
    std::vector<float> getCentroids() const;

    // Get inertia (within-cluster sum of squares)
    float getInertia() const { return inertia; }

    // Check if training is complete
    bool isTrainingComplete() const { return trainingComplete; }

    // Get iteration history
    std::vector<float> getInertiaHistory() const { return inertiaHistory; }

private:
    int numClusters;
    int numFeatures;
    float* d_centroids;
    float* h_centroids;
    float inertia;
    bool trainingComplete;

    std::vector<float> inertiaHistory;
    cudaStream_t stream;
};

#endif // KMEANS_H
