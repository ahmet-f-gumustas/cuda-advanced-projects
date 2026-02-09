#pragma once

#include "common.hpp"
#include <deque>

namespace cuda_stabilizer {

// Smoothing method types
enum class SmoothingMethod {
    MOVING_AVERAGE,
    GAUSSIAN,
    KALMAN
};

class TrajectorySmoother {
public:
    TrajectorySmoother(int smoothing_radius = 30, SmoothingMethod method = SmoothingMethod::GAUSSIAN);
    ~TrajectorySmoother();

    // Add a new trajectory point
    void addPoint(const Trajectory& point);

    // Get smoothed trajectory at specific index
    Trajectory getSmoothedPoint(int index) const;

    // Get current smoothed trajectory (for real-time)
    Trajectory getCurrentSmoothed() const;

    // Smooth entire trajectory (for offline processing)
    std::vector<Trajectory> smoothTrajectory(const std::vector<Trajectory>& trajectory);

    // Reset smoother state
    void reset();

    // Configuration
    void setSmoothingRadius(int radius) { smoothing_radius_ = radius; }
    int getSmoothingRadius() const { return smoothing_radius_; }

    void setSmoothingMethod(SmoothingMethod method) { method_ = method; }
    SmoothingMethod getSmoothingMethod() const { return method_; }

private:
    // Moving average smoothing
    Trajectory movingAverageSmooth(const std::vector<Trajectory>& trajectory, int index) const;

    // Gaussian weighted smoothing
    Trajectory gaussianSmooth(const std::vector<Trajectory>& trajectory, int index) const;

    // Kalman filter smoothing
    void kalmanPredict();
    void kalmanUpdate(const Trajectory& measurement);
    Trajectory kalmanGetState() const;

private:
    int smoothing_radius_;
    SmoothingMethod method_;

    // Trajectory buffer for real-time processing
    std::deque<Trajectory> trajectory_buffer_;

    // Kalman filter state
    struct KalmanState {
        // State: [x, y, angle, scale, vx, vy, va, vs]
        float x[8];
        float P[8][8];  // Covariance matrix
        float Q[8][8];  // Process noise
        float R[4][4];  // Measurement noise

        KalmanState();
        void reset();
    } kalman_state_;

    // Gaussian kernel cache
    std::vector<float> gaussian_kernel_;
    void computeGaussianKernel();
};

} // namespace cuda_stabilizer
