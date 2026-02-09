#include "video_stabilizer.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>

namespace cuda_stabilizer {

VideoStabilizer::VideoStabilizer(const StabilizerConfig& config)
    : config_(config)
    , width_(0)
    , height_(0)
    , channels_(3)
    , initialized_(false)
    , frame_count_(0)
    , avg_processing_time_(0.0f)
    , fps_(0.0f)
    , is_running_(false)
    , d_prev_frame_(nullptr)
    , d_curr_frame_(nullptr)
    , d_flow_x_(nullptr)
    , d_flow_y_(nullptr)
    , d_input_frame_(nullptr)
    , d_output_frame_(nullptr)
    , d_prev_points_(nullptr)
    , d_curr_points_(nullptr)
    , d_status_(nullptr)
    , max_features_(1000)
    , trajectory_smoother_(config.smoothing_radius)
{
}

VideoStabilizer::~VideoStabilizer() {
    if (initialized_) {
        // Free GPU resources
        if (d_prev_frame_) cudaFree(d_prev_frame_);
        if (d_curr_frame_) cudaFree(d_curr_frame_);
        if (d_flow_x_) cudaFree(d_flow_x_);
        if (d_flow_y_) cudaFree(d_flow_y_);
        if (d_input_frame_) cudaFree(d_input_frame_);
        if (d_output_frame_) cudaFree(d_output_frame_);
        if (d_prev_points_) cudaFree(d_prev_points_);
        if (d_curr_points_) cudaFree(d_curr_points_);
        if (d_status_) cudaFree(d_status_);

        releaseOpticalFlow();
        releaseMotionEstimation();
        releaseFrameWarping();
    }
}

bool VideoStabilizer::initialize(int width, int height, int channels) {
    width_ = width;
    height_ = height;
    channels_ = channels;

    size_t frame_size = width * height * channels;
    size_t gray_size = width * height;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_prev_frame_, gray_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curr_frame_, gray_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flow_x_, gray_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flow_y_, gray_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_frame_, frame_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_output_frame_, frame_size * sizeof(unsigned char)));

    // Feature tracking buffers
    CUDA_CHECK(cudaMalloc(&d_prev_points_, max_features_ * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curr_points_, max_features_ * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_status_, max_features_ * sizeof(unsigned char)));

    // Initialize CUDA modules
    initOpticalFlow(width, height, config_.pyramid_levels);
    initMotionEstimation(width, height);
    initFrameWarping(width, height);

    // Initialize trajectory
    cumulative_trajectory_ = Trajectory(0, 0, 0, 1.0f);

    initialized_ = true;
    return true;
}

TransformParams VideoStabilizer::estimateMotion(const cv::Mat& prev_gray, const cv::Mat& curr_gray) {
    // Use OpenCV's goodFeaturesToTrack and calcOpticalFlowPyrLK for robust feature tracking
    std::vector<cv::Point2f> prev_pts, curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Detect features in previous frame
    cv::goodFeaturesToTrack(prev_gray, prev_pts, max_features_, 0.01, 10);

    if (prev_pts.empty()) {
        return TransformParams();
    }

    // Track features to current frame
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

    // Filter good points
    std::vector<cv::Point2f> good_prev, good_curr;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            good_prev.push_back(prev_pts[i]);
            good_curr.push_back(curr_pts[i]);
        }
    }

    if (good_prev.size() < 4) {
        return TransformParams();
    }

    // Estimate affine transformation using RANSAC
    cv::Mat transform_matrix = cv::estimateAffinePartial2D(good_prev, good_curr);

    if (transform_matrix.empty()) {
        return TransformParams();
    }

    // Extract transform parameters
    double a = transform_matrix.at<double>(0, 0);
    double b = transform_matrix.at<double>(0, 1);
    double c = transform_matrix.at<double>(1, 0);
    double d = transform_matrix.at<double>(1, 1);
    double dx = transform_matrix.at<double>(0, 2);
    double dy = transform_matrix.at<double>(1, 2);

    // Decompose affine matrix
    double scale_x = sqrt(a * a + c * c);
    double scale_y = sqrt(b * b + d * d);
    double scale = (scale_x + scale_y) / 2.0;
    double angle = atan2(c, a);

    return TransformParams(static_cast<float>(dx), static_cast<float>(dy),
                           static_cast<float>(angle), static_cast<float>(scale));
}

void VideoStabilizer::updateTrajectory(const TransformParams& motion) {
    cumulative_trajectory_.x += motion.dx;
    cumulative_trajectory_.y += motion.dy;
    cumulative_trajectory_.a += motion.da;

    original_trajectory_.push_back(cumulative_trajectory_);
    transforms_.push_back(motion);
}

TransformParams VideoStabilizer::getStabilizationTransform() {
    // Add current trajectory to smoother
    trajectory_smoother_.addPoint(cumulative_trajectory_);

    // Get smoothed trajectory
    Trajectory smoothed = trajectory_smoother_.getCurrentSmoothed();
    smoothed_trajectory_.push_back(smoothed);

    // Compute difference between original and smoothed trajectory
    Trajectory diff;
    diff.x = smoothed.x - cumulative_trajectory_.x;
    diff.y = smoothed.y - cumulative_trajectory_.y;
    diff.a = smoothed.a - cumulative_trajectory_.a;

    return TransformParams(diff.x, diff.y, diff.a, 1.0f);
}

void VideoStabilizer::applyStabilization(const cv::Mat& input, cv::Mat& output, const TransformParams& transform) {
    if (config_.use_gpu) {
        // Upload frame to GPU
        CUDA_CHECK(cudaMemcpy(d_input_frame_, input.data,
                              width_ * height_ * channels_ * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        // Apply transformation with crop
        warpFrameAffineCrop(d_input_frame_, d_output_frame_,
                           width_, height_, channels_,
                           transform, config_.crop_ratio);

        // Download result
        output.create(height_, width_, input.type());
        CUDA_CHECK(cudaMemcpy(output.data, d_output_frame_,
                              width_ * height_ * channels_ * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
    } else {
        // CPU fallback using OpenCV
        float cx = width_ / 2.0f;
        float cy = height_ / 2.0f;

        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(cx, cy), transform.da * 180.0 / M_PI, transform.ds);
        rot_mat.at<double>(0, 2) += transform.dx;
        rot_mat.at<double>(1, 2) += transform.dy;

        cv::warpAffine(input, output, rot_mat, input.size());

        // Apply crop
        float crop_scale = config_.crop_ratio;
        int new_w = static_cast<int>(width_ * crop_scale);
        int new_h = static_cast<int>(height_ * crop_scale);
        int x = (width_ - new_w) / 2;
        int y = (height_ - new_h) / 2;

        cv::Mat cropped = output(cv::Rect(x, y, new_w, new_h));
        cv::resize(cropped, output, cv::Size(width_, height_));
    }
}

void VideoStabilizer::fixBorders(cv::Mat& frame) {
    // Simple border fix - replicate edge pixels
    int border = static_cast<int>((1.0f - config_.crop_ratio) * std::min(width_, height_) / 2);
    cv::copyMakeBorder(frame, frame, border, border, border, border, cv::BORDER_REPLICATE);
    cv::resize(frame, frame, cv::Size(width_, height_));
}

bool VideoStabilizer::processFrame(const cv::Mat& input, cv::Mat& output) {
    if (!initialized_) {
        if (!initialize(input.cols, input.rows, input.channels())) {
            return false;
        }
    }

    CudaTimer timer;
    timer.start();

    cv::Mat curr_gray;
    cv::cvtColor(input, curr_gray, cv::COLOR_BGR2GRAY);

    if (frame_count_ == 0) {
        // First frame - just store it
        prev_frame_ = input.clone();
        prev_gray_ = curr_gray.clone();
        output = input.clone();
        frame_count_++;
        return true;
    }

    // Estimate motion between frames
    TransformParams motion = estimateMotion(prev_gray_, curr_gray);

    // Update cumulative trajectory
    updateTrajectory(motion);

    // Get stabilization transform
    TransformParams stabilization = getStabilizationTransform();

    // Apply stabilization
    applyStabilization(input, output, stabilization);

    // Update previous frame
    prev_frame_ = input.clone();
    prev_gray_ = curr_gray.clone();

    // Update statistics
    float ms = timer.stop();
    avg_processing_time_ = (avg_processing_time_ * frame_count_ + ms) / (frame_count_ + 1);
    fps_ = 1000.0f / avg_processing_time_;
    frame_count_++;

    return true;
}

bool VideoStabilizer::processVideo(const std::string& input_path, const std::string& output_path) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open input video: " << input_path << std::endl;
        return false;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Input video: " << width << "x" << height << " @ " << fps << " fps, "
              << total_frames << " frames" << std::endl;

    // Initialize
    if (!initialize(width, height, 3)) {
        std::cerr << "Error: Failed to initialize stabilizer" << std::endl;
        return false;
    }

    // Create video writer
    cv::VideoWriter writer(output_path,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not create output video: " << output_path << std::endl;
        return false;
    }

    // Two-pass stabilization for better results
    std::cout << "Pass 1: Analyzing motion..." << std::endl;

    cv::Mat frame, prev_gray_pass1;
    std::vector<TransformParams> all_transforms;

    int frame_idx = 0;
    while (cap.read(frame)) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (frame_idx > 0) {
            TransformParams motion = estimateMotion(prev_gray_pass1, gray);
            all_transforms.push_back(motion);
        }

        prev_gray_pass1 = gray.clone();
        frame_idx++;

        if (frame_idx % 100 == 0) {
            std::cout << "  Processed " << frame_idx << "/" << total_frames << " frames" << std::endl;
        }
    }

    // Compute trajectory and smooth
    std::cout << "Pass 1: Smoothing trajectory..." << std::endl;

    std::vector<Trajectory> trajectory(all_transforms.size() + 1);
    trajectory[0] = Trajectory(0, 0, 0, 1.0f);

    for (size_t i = 0; i < all_transforms.size(); i++) {
        trajectory[i + 1].x = trajectory[i].x + all_transforms[i].dx;
        trajectory[i + 1].y = trajectory[i].y + all_transforms[i].dy;
        trajectory[i + 1].a = trajectory[i].a + all_transforms[i].da;
    }

    // Smooth trajectory
    std::vector<Trajectory> smoothed = trajectory_smoother_.smoothTrajectory(trajectory);

    // Compute new transforms
    std::vector<TransformParams> new_transforms(all_transforms.size());
    for (size_t i = 0; i < all_transforms.size(); i++) {
        new_transforms[i].dx = all_transforms[i].dx + smoothed[i + 1].x - trajectory[i + 1].x;
        new_transforms[i].dy = all_transforms[i].dy + smoothed[i + 1].y - trajectory[i + 1].y;
        new_transforms[i].da = all_transforms[i].da + smoothed[i + 1].a - trajectory[i + 1].a;
        new_transforms[i].ds = 1.0f;
    }

    // Pass 2: Apply stabilization
    std::cout << "Pass 2: Applying stabilization..." << std::endl;

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    frame_idx = 0;

    while (cap.read(frame)) {
        cv::Mat stabilized;

        if (frame_idx == 0) {
            stabilized = frame.clone();
        } else {
            TransformParams transform;
            transform.dx = new_transforms[frame_idx - 1].dx;
            transform.dy = new_transforms[frame_idx - 1].dy;
            transform.da = new_transforms[frame_idx - 1].da;
            transform.ds = 1.0f;

            applyStabilization(frame, stabilized, transform);
        }

        if (config_.show_comparison) {
            // Create side-by-side comparison
            cv::Mat comparison;
            cv::hconcat(frame, stabilized, comparison);
            cv::putText(comparison, "Original", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::putText(comparison, "Stabilized", cv::Point(width + 10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Comparison", comparison);
            cv::waitKey(1);
        }

        writer.write(stabilized);
        frame_idx++;

        if (frame_idx % 100 == 0) {
            std::cout << "  Processed " << frame_idx << "/" << total_frames << " frames" << std::endl;
        }
    }

    std::cout << "Video stabilization complete!" << std::endl;
    std::cout << "Output saved to: " << output_path << std::endl;

    cap.release();
    writer.release();

    return true;
}

bool VideoStabilizer::startRealTimeProcessing(int camera_id) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << camera_id << std::endl;
        return false;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    if (!initialize(width, height, 3)) {
        return false;
    }

    is_running_ = true;
    cv::Mat frame, stabilized;

    std::cout << "Real-time stabilization started. Press 'q' to quit." << std::endl;

    while (is_running_) {
        if (!cap.read(frame)) {
            break;
        }

        processFrame(frame, stabilized);

        // Display
        cv::Mat display;
        if (config_.show_comparison) {
            cv::hconcat(frame, stabilized, display);
            cv::putText(display, "Original", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(display, "Stabilized", cv::Point(width + 10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
            display = stabilized;
        }

        // Show FPS
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps_));
        cv::putText(display, fps_text, cv::Point(display.cols - 120, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Video Stabilization", display);

        if (cv::waitKey(1) == 'q') {
            is_running_ = false;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return true;
}

void VideoStabilizer::stopRealTimeProcessing() {
    is_running_ = false;
}

} // namespace cuda_stabilizer
