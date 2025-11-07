#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include "linear_regression.h"
#include "kmeans.h"
#include <thread>
#include <memory>
#include <atomic>

class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    // Start parallel training of both models
    void startParallelTraining();

    // Check if both models are done training
    bool areModelsTraining() const;

    // Get models
    LinearRegression* getLinearRegression() { return linearModel.get(); }
    KMeans* getKMeans() { return kmeansModel.get(); }

    // Get training progress
    float getLinearRegressionProgress() const;
    float getKMeansProgress() const;

private:
    std::unique_ptr<LinearRegression> linearModel;
    std::unique_ptr<KMeans> kmeansModel;

    std::unique_ptr<std::thread> linearThread;
    std::unique_ptr<std::thread> kmeansThread;

    std::atomic<bool> linearTraining;
    std::atomic<bool> kmeansTraining;

    // Training data
    std::vector<float> linearX;
    std::vector<float> linearY;
    std::vector<float> kmeansData;

    void trainLinearModel();
    void trainKMeansModel();
    void generateData();
};

#endif // MODEL_MANAGER_H
