#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <cuda_runtime.h>

class LinearRegression {
public:
    LinearRegression(int numFeatures);
    ~LinearRegression();

    // Train the model using gradient descent
    void train(const std::vector<float>& X, const std::vector<float>& y,
               int numSamples, int numIterations, float learningRate);

    // Make predictions
    std::vector<float> predict(const std::vector<float>& X, int numSamples);

    // Get model parameters
    float getSlope() const { return h_weights[0]; }
    float getIntercept() const { return h_bias; }

    // Get training loss history
    std::vector<float> getLossHistory() const { return lossHistory; }

    // Check if training is complete
    bool isTrainingComplete() const { return trainingComplete; }

private:
    int numFeatures;
    float* d_weights;
    float* d_bias;
    float* h_weights;
    float h_bias;

    std::vector<float> lossHistory;
    bool trainingComplete;

    cudaStream_t stream;
};

#endif // LINEAR_REGRESSION_H
