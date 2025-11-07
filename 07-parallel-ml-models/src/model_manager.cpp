#include "model_manager.h"
#include <iostream>
#include <random>
#include <cmath>

ModelManager::ModelManager()
    : linearTraining(false), kmeansTraining(false) {

    // Initialize models
    linearModel = std::make_unique<LinearRegression>(1);  // 1 feature (simple y = mx + b)
    kmeansModel = std::make_unique<KMeans>(3, 2);         // 3 clusters, 2D data

    // Generate training data
    generateData();
}

ModelManager::~ModelManager() {
    // Wait for threads to finish
    if (linearThread && linearThread->joinable()) {
        linearThread->join();
    }
    if (kmeansThread && kmeansThread->joinable()) {
        kmeansThread->join();
    }
}

void ModelManager::generateData() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.0f, 10.0f);

    // Generate linear regression data: y = 2.5x + 1.0 + noise
    int numSamples = 1000;
    linearX.resize(numSamples);
    linearY.resize(numSamples);

    for (int i = 0; i < numSamples; i++) {
        float x = uniform(gen);
        linearX[i] = x;
        linearY[i] = 2.5f * x + 1.0f + noise(gen);
    }

    // Generate K-Means data: 3 clusters in 2D space
    int samplesPerCluster = 300;
    kmeansData.resize(samplesPerCluster * 3 * 2);  // 3 clusters, 2 features each

    // Cluster centers
    float centers[3][2] = {
        {2.0f, 2.0f},
        {6.0f, 6.0f},
        {2.0f, 8.0f}
    };

    for (int c = 0; c < 3; c++) {
        std::normal_distribution<float> distX(centers[c][0], 0.8f);
        std::normal_distribution<float> distY(centers[c][1], 0.8f);

        for (int i = 0; i < samplesPerCluster; i++) {
            int idx = (c * samplesPerCluster + i) * 2;
            kmeansData[idx] = distX(gen);
            kmeansData[idx + 1] = distY(gen);
        }
    }

    std::cout << "Generated training data:\n";
    std::cout << "  Linear Regression: " << numSamples << " samples\n";
    std::cout << "  K-Means: " << (samplesPerCluster * 3) << " samples in 3 clusters\n";
}

void ModelManager::trainLinearModel() {
    std::cout << "[Linear Regression] Training started on GPU...\n";
    linearTraining = true;

    int numSamples = linearX.size();
    int numIterations = 500;
    float learningRate = 0.01f;

    linearModel->train(linearX, linearY, numSamples, numIterations, learningRate);

    std::cout << "[Linear Regression] Training completed!\n";
    std::cout << "  Final parameters: slope = " << linearModel->getSlope()
              << ", intercept = " << linearModel->getIntercept() << "\n";

    linearTraining = false;
}

void ModelManager::trainKMeansModel() {
    std::cout << "[K-Means] Training started on GPU...\n";
    kmeansTraining = true;

    int numSamples = kmeansData.size() / 2;
    int maxIterations = 100;

    kmeansModel->train(kmeansData, numSamples, maxIterations);

    std::cout << "[K-Means] Training completed!\n";
    std::cout << "  Final inertia: " << kmeansModel->getInertia() << "\n";

    kmeansTraining = false;
}

void ModelManager::startParallelTraining() {
    std::cout << "\n========================================\n";
    std::cout << "STARTING PARALLEL MODEL TRAINING\n";
    std::cout << "========================================\n\n";

    // Launch both training processes in parallel threads
    linearThread = std::make_unique<std::thread>(&ModelManager::trainLinearModel, this);
    kmeansThread = std::make_unique<std::thread>(&ModelManager::trainKMeansModel, this);

    std::cout << "Both models are now training in parallel on GPU!\n";
    std::cout << "Using separate CUDA streams for concurrent execution.\n\n";
}

bool ModelManager::areModelsTraining() const {
    return linearTraining || kmeansTraining;
}

float ModelManager::getLinearRegressionProgress() const {
    if (!linearModel->isTrainingComplete() && linearTraining) {
        return 0.5f;  // Training in progress
    }
    return linearModel->isTrainingComplete() ? 1.0f : 0.0f;
}

float ModelManager::getKMeansProgress() const {
    if (!kmeansModel->isTrainingComplete() && kmeansTraining) {
        return 0.5f;  // Training in progress
    }
    return kmeansModel->isTrainingComplete() ? 1.0f : 0.0f;
}
