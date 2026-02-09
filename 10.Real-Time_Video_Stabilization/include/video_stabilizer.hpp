#pragma once

#include "common.hpp"
#include "optical_flow.cuh"
#include "motion_estimation.cuh"
#include "frame_warping.cuh"
#include "gaussian_pyramid.cuh"
#include "trajectory_smoother.hpp"
#include "memory_manager.hpp"

namespace cuda_stabilizer {

class VideoStabilizer {
public:
    VideoStabilizer(const StabilizerConfig& config = StabilizerConfig());
    ~VideoStabilizer();

    // Initialize with video dimensions
    bool initialize(int width, int height, int channels = 3);

    // Process a single frame
    bool processFrame(const cv::Mat& input, cv::Mat& output);

    // Process entire video file
    bool processVideo(const std::string& input_path, const std::string& output_path);

    // Real-time processing mode
    bool startRealTimeProcessing(int camera_id = 0);
    void stopRealTimeProcessing();

    // Get processing statistics
    float getAverageProcessingTime() const { return avg_processing_time_; }
    float getFPS() const { return fps_; }
    int getFrameCount() const { return frame_count_; }

    // Configuration
    void setConfig(const StabilizerConfig& config) { config_ = config; }
    const StabilizerConfig& getConfig() const { return config_; }

    // Get trajectory data for visualization
    const std::vector<Trajectory>& getOriginalTrajectory() const { return original_trajectory_; }
    const std::vector<Trajectory>& getSmoothedTrajectory() const { return smoothed_trajectory_; }

private:
    // Estimate motion between previous and current frame
    TransformParams estimateMotion(const cv::Mat& prev_gray, const cv::Mat& curr_gray);

    // Apply stabilization transform to frame
    void applyStabilization(const cv::Mat& input, cv::Mat& output, const TransformParams& transform);

    // Compute cumulative trajectory
    void updateTrajectory(const TransformParams& motion);

    // Get stabilization transform for current frame
    TransformParams getStabilizationTransform();

    // Fix border artifacts after warping
    void fixBorders(cv::Mat& frame);

private:
    StabilizerConfig config_;
    MemoryManager memory_manager_;

    // Frame dimensions
    int width_;
    int height_;
    int channels_;
    bool initialized_;

    // Previous frame storage
    cv::Mat prev_frame_;
    cv::Mat prev_gray_;

    // Trajectory tracking
    std::vector<TransformParams> transforms_;
    std::vector<Trajectory> original_trajectory_;
    std::vector<Trajectory> smoothed_trajectory_;
    Trajectory cumulative_trajectory_;

    // Processing statistics
    int frame_count_;
    float avg_processing_time_;
    float fps_;

    // Real-time processing flag
    bool is_running_;

    // GPU resources
    float* d_prev_frame_;
    float* d_curr_frame_;
    float* d_flow_x_;
    float* d_flow_y_;
    unsigned char* d_input_frame_;
    unsigned char* d_output_frame_;

    // Feature tracking
    float* d_prev_points_;
    float* d_curr_points_;
    unsigned char* d_status_;
    int max_features_;

    // Trajectory smoother (must be after max_features_ for init order)
    TrajectorySmoother trajectory_smoother_;
};

} // namespace cuda_stabilizer
