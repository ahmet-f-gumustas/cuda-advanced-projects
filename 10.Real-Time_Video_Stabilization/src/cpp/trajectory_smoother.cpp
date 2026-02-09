#include "trajectory_smoother.hpp"
#include <cmath>
#include <algorithm>

namespace cuda_stabilizer {

TrajectorySmoother::KalmanState::KalmanState() {
    reset();
}

void TrajectorySmoother::KalmanState::reset() {
    // Initialize state to zero
    for (int i = 0; i < 8; i++) {
        x[i] = (i == 3) ? 1.0f : 0.0f;  // Scale starts at 1
    }

    // Initialize covariance matrix (high uncertainty initially)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            P[i][j] = (i == j) ? 100.0f : 0.0f;
        }
    }

    // Process noise
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Q[i][j] = (i == j) ? 0.001f : 0.0f;
        }
    }

    // Measurement noise
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            R[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

TrajectorySmoother::TrajectorySmoother(int smoothing_radius, SmoothingMethod method)
    : smoothing_radius_(smoothing_radius)
    , method_(method)
{
    computeGaussianKernel();
}

TrajectorySmoother::~TrajectorySmoother() {
}

void TrajectorySmoother::computeGaussianKernel() {
    int kernel_size = 2 * smoothing_radius_ + 1;
    gaussian_kernel_.resize(kernel_size);

    float sigma = smoothing_radius_ / 3.0f;
    float sum = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        int x = i - smoothing_radius_;
        gaussian_kernel_[i] = std::exp(-0.5f * (x * x) / (sigma * sigma));
        sum += gaussian_kernel_[i];
    }

    // Normalize
    for (int i = 0; i < kernel_size; i++) {
        gaussian_kernel_[i] /= sum;
    }
}

void TrajectorySmoother::addPoint(const Trajectory& point) {
    trajectory_buffer_.push_back(point);

    // Keep buffer size manageable
    if (trajectory_buffer_.size() > static_cast<size_t>(smoothing_radius_ * 4)) {
        trajectory_buffer_.pop_front();
    }

    if (method_ == SmoothingMethod::KALMAN) {
        kalmanUpdate(point);
    }
}

Trajectory TrajectorySmoother::getCurrentSmoothed() const {
    if (trajectory_buffer_.empty()) {
        return Trajectory();
    }

    switch (method_) {
        case SmoothingMethod::MOVING_AVERAGE: {
            Trajectory avg;
            int count = 0;
            int start = std::max(0, static_cast<int>(trajectory_buffer_.size()) - smoothing_radius_);

            for (int i = start; i < static_cast<int>(trajectory_buffer_.size()); i++) {
                avg.x += trajectory_buffer_[i].x;
                avg.y += trajectory_buffer_[i].y;
                avg.a += trajectory_buffer_[i].a;
                count++;
            }

            if (count > 0) {
                avg.x /= count;
                avg.y /= count;
                avg.a /= count;
            }
            return avg;
        }

        case SmoothingMethod::GAUSSIAN: {
            Trajectory result;
            float total_weight = 0.0f;

            int buf_size = static_cast<int>(trajectory_buffer_.size());
            int center = buf_size - 1;

            for (int i = -smoothing_radius_; i <= 0; i++) {
                int idx = center + i;
                if (idx >= 0 && idx < buf_size) {
                    float weight = gaussian_kernel_[i + smoothing_radius_];
                    result.x += trajectory_buffer_[idx].x * weight;
                    result.y += trajectory_buffer_[idx].y * weight;
                    result.a += trajectory_buffer_[idx].a * weight;
                    total_weight += weight;
                }
            }

            if (total_weight > 0.0f) {
                result.x /= total_weight;
                result.y /= total_weight;
                result.a /= total_weight;
            }
            return result;
        }

        case SmoothingMethod::KALMAN:
            return kalmanGetState();
    }

    return trajectory_buffer_.back();
}

Trajectory TrajectorySmoother::getSmoothedPoint(int index) const {
    if (trajectory_buffer_.empty() || index < 0) {
        return Trajectory();
    }

    std::vector<Trajectory> temp(trajectory_buffer_.begin(), trajectory_buffer_.end());
    return gaussianSmooth(temp, std::min(index, static_cast<int>(temp.size()) - 1));
}

std::vector<Trajectory> TrajectorySmoother::smoothTrajectory(const std::vector<Trajectory>& trajectory) {
    std::vector<Trajectory> smoothed(trajectory.size());

    switch (method_) {
        case SmoothingMethod::MOVING_AVERAGE:
            for (size_t i = 0; i < trajectory.size(); i++) {
                smoothed[i] = movingAverageSmooth(trajectory, i);
            }
            break;

        case SmoothingMethod::GAUSSIAN:
            for (size_t i = 0; i < trajectory.size(); i++) {
                smoothed[i] = gaussianSmooth(trajectory, i);
            }
            break;

        case SmoothingMethod::KALMAN: {
            // Forward pass
            KalmanState state;
            state.reset();
            std::vector<Trajectory> forward(trajectory.size());

            for (size_t i = 0; i < trajectory.size(); i++) {
                kalman_state_ = state;
                kalmanPredict();
                kalmanUpdate(trajectory[i]);
                state = kalman_state_;
                forward[i] = kalmanGetState();
            }

            // Backward pass
            state.reset();
            std::vector<Trajectory> backward(trajectory.size());

            for (int i = trajectory.size() - 1; i >= 0; i--) {
                kalman_state_ = state;
                kalmanPredict();
                kalmanUpdate(trajectory[i]);
                state = kalman_state_;
                backward[i] = kalmanGetState();
            }

            // Combine forward and backward
            for (size_t i = 0; i < trajectory.size(); i++) {
                smoothed[i].x = (forward[i].x + backward[i].x) / 2.0f;
                smoothed[i].y = (forward[i].y + backward[i].y) / 2.0f;
                smoothed[i].a = (forward[i].a + backward[i].a) / 2.0f;
                smoothed[i].s = 1.0f;
            }
            break;
        }
    }

    return smoothed;
}

void TrajectorySmoother::reset() {
    trajectory_buffer_.clear();
    kalman_state_.reset();
}

Trajectory TrajectorySmoother::movingAverageSmooth(const std::vector<Trajectory>& trajectory, int index) const {
    Trajectory result;
    int count = 0;

    int start = std::max(0, index - smoothing_radius_);
    int end = std::min(static_cast<int>(trajectory.size()) - 1, index + smoothing_radius_);

    for (int i = start; i <= end; i++) {
        result.x += trajectory[i].x;
        result.y += trajectory[i].y;
        result.a += trajectory[i].a;
        count++;
    }

    if (count > 0) {
        result.x /= count;
        result.y /= count;
        result.a /= count;
    }

    return result;
}

Trajectory TrajectorySmoother::gaussianSmooth(const std::vector<Trajectory>& trajectory, int index) const {
    Trajectory result;
    float total_weight = 0.0f;

    for (int i = -smoothing_radius_; i <= smoothing_radius_; i++) {
        int idx = index + i;
        if (idx >= 0 && idx < static_cast<int>(trajectory.size())) {
            float weight = gaussian_kernel_[i + smoothing_radius_];
            result.x += trajectory[idx].x * weight;
            result.y += trajectory[idx].y * weight;
            result.a += trajectory[idx].a * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0f) {
        result.x /= total_weight;
        result.y /= total_weight;
        result.a /= total_weight;
    }

    return result;
}

void TrajectorySmoother::kalmanPredict() {
    // State transition: position += velocity
    kalman_state_.x[0] += kalman_state_.x[4];  // x += vx
    kalman_state_.x[1] += kalman_state_.x[5];  // y += vy
    kalman_state_.x[2] += kalman_state_.x[6];  // a += va
    kalman_state_.x[3] *= kalman_state_.x[7];  // s *= vs

    // Update covariance P = F*P*F' + Q (simplified)
    for (int i = 0; i < 8; i++) {
        kalman_state_.P[i][i] += kalman_state_.Q[i][i];
    }
}

void TrajectorySmoother::kalmanUpdate(const Trajectory& measurement) {
    // Measurement: z = [x, y, a, s]
    float z[4] = {measurement.x, measurement.y, measurement.a, measurement.s};

    // Innovation: y = z - H*x (H is identity for positions)
    float y[4];
    for (int i = 0; i < 4; i++) {
        y[i] = z[i] - kalman_state_.x[i];
    }

    // Innovation covariance: S = H*P*H' + R
    float S[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            S[i][j] = kalman_state_.P[i][j] + kalman_state_.R[i][j];
        }
    }

    // Kalman gain: K = P*H'*S^-1 (simplified - assume S is diagonal)
    float K[8][4];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            K[i][j] = kalman_state_.P[i][j] / (S[j][j] + 1e-6f);
        }
    }

    // Update state: x = x + K*y
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            kalman_state_.x[i] += K[i][j] * y[j];
        }
    }

    // Update covariance: P = (I - K*H)*P (simplified)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            kalman_state_.P[i][j] *= (1.0f - K[i][i]);
        }
    }
}

Trajectory TrajectorySmoother::kalmanGetState() const {
    return Trajectory(
        kalman_state_.x[0],
        kalman_state_.x[1],
        kalman_state_.x[2],
        kalman_state_.x[3]
    );
}

} // namespace cuda_stabilizer
