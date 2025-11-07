#include "model_manager.h"
#include "visualizer.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "======================================================\n";
    std::cout << "  CUDA Parallel ML Models Demonstration\n";
    std::cout << "======================================================\n";
    std::cout << "\nThis demo showcases parallel training of two ML models:\n";
    std::cout << "  1. Linear Regression - Finding best fit line\n";
    std::cout << "  2. K-Means Clustering - Grouping data into clusters\n";
    std::cout << "\nBoth models train simultaneously on GPU using CUDA!\n";
    std::cout << "======================================================\n\n";

    try {
        // Create model manager
        std::cout << "Initializing models...\n";
        ModelManager manager;

        // Create visualizer
        std::cout << "Initializing OpenGL visualization...\n";
        Visualizer visualizer(&manager, 1600, 800);

        if (!visualizer.initialize()) {
            std::cerr << "Failed to initialize visualizer\n";
            return -1;
        }

        // Start parallel training (in background threads)
        manager.startParallelTraining();

        // Run visualization loop
        std::cout << "Starting visualization...\n";
        std::cout << "======================================================\n\n";
        visualizer.run();

        // Cleanup
        std::cout << "\n\nCleaning up...\n";
        visualizer.cleanup();

        // Print final results
        std::cout << "\n======================================================\n";
        std::cout << "FINAL RESULTS\n";
        std::cout << "======================================================\n\n";

        std::cout << "Linear Regression:\n";
        std::cout << "  Slope: " << manager.getLinearRegression()->getSlope() << "\n";
        std::cout << "  Intercept: " << manager.getLinearRegression()->getIntercept() << "\n";
        std::cout << "  Expected: y = 2.5x + 1.0\n\n";

        std::cout << "K-Means:\n";
        std::cout << "  Final Inertia: " << manager.getKMeans()->getInertia() << "\n";
        std::cout << "  Clusters: 3\n\n";

        std::cout << "======================================================\n";
        std::cout << "Demo completed successfully!\n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
