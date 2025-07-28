#include "cuda_denoiser.h"
#include "image_processor.h"
#include "memory_manager.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <chrono>

using namespace rtid;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -i, --input <path>     Input image path\n"
              << "  -o, --output <path>    Output image path\n"
              << "  -a, --algorithm <name> Denoising algorithm (bilateral, nlm, gaussian, adaptive)\n"
              << "  -c, --camera           Use camera input (real-time processing)\n"
              << "  -b, --benchmark        Run benchmark on test images\n"
              << "  -h, --help             Show this help message\n"
              << "\nAlgorithm Parameters:\n"
              << "  --sigma-color <value>  Bilateral filter color sigma (default: 50.0)\n"
              << "  --sigma-space <value>  Bilateral filter space sigma (default: 50.0)\n"
              << "  --h-param <value>      NLM filtering strength (default: 10.0)\n"
              << "  --template-size <size> NLM template window size (default: 7)\n"
              << "  --search-size <size>   NLM search window size (default: 21)\n"
              << "  --gaussian-sigma <val> Gaussian sigma (default: 1.0)\n"
              << "  --kernel-size <size>   Kernel size (default: 5)\n"
              << std::endl;
}

DenoiseAlgorithm parseAlgorithm(const std::string& algo_name) {
    if (algo_name == "bilateral") return DenoiseAlgorithm::BILATERAL;
    if (algo_name == "nlm") return DenoiseAlgorithm::NON_LOCAL_MEANS;
    if (algo_name == "gaussian") return DenoiseAlgorithm::GAUSSIAN;
    if (algo_name == "adaptive") return DenoiseAlgorithm::ADAPTIVE_BILATERAL;
    
    std::cerr << "Unknown algorithm: " << algo_name << std::endl;
    return DenoiseAlgorithm::BILATERAL;
}

void processSingleImage(const std::string& input_path, const std::string& output_path,
                       DenoiseAlgorithm algorithm, const DenoiseParams& params) {
    // Load input image
    cv::Mat input_image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return;
    }
    
    std::cout << "Processing image: " << input_path << std::endl;
    std::cout << "Image size: " << input_image.cols << "x" << input_image.rows << std::endl;
    
    // Create denoiser
    CudaDenoiser denoiser(input_image.cols, input_image.rows);
    
    // Process image
    cv::Mat output_image;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = denoiser.denoise(input_image, output_image, algorithm, params);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (success) {
        // Save output image
        cv::imwrite(output_path, output_image);
        std::cout << "Processing completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Output saved to: " << output_path << std::endl;
        
        // Print performance stats
        denoiser.printPerformanceStats();
    } else {
        std::cerr << "Error: Image processing failed" << std::endl;
    }
}

void runRealTimeProcessing(DenoiseAlgorithm algorithm, const DenoiseParams& params) {
    std::cout << "Starting real-time camera processing..." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save current frame" << std::endl;
    
    // Create image processor
    ImageProcessor processor;
    
    if (!processor.initialize(1920, 1080)) {
        std::cerr << "Error: Failed to initialize image processor" << std::endl;
        return;
    }
    
    // Set algorithm and parameters
    processor.setDenoiseAlgorithm(algorithm);
    processor.setDenoiseParams(params);
    
    // Enable profiling
    processor.enableProfiling(true);
    
    // Start camera capture
    if (!processor.startCameraCapture(0)) {
        std::cerr << "Error: Failed to start camera capture" << std::endl;
        return;
    }
    
    // Start real-time processing
    if (!processor.startRealTimeProcessing()) {
        std::cerr << "Error: Failed to start real-time processing" << std::endl;
        return;
    }
    
    // Display loop
    cv::Mat frame, processed_frame;
    int frame_count = 0;
    auto last_stats_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // Get processed frame
        if (processor.dequeueResult(processed_frame, 30)) {
            // Display the processed frame
            cv::imshow("Real-time Denoising", processed_frame);
            frame_count++;
            
            // Print statistics every 5 seconds
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
            
            if (elapsed.count() >= 5) {
                auto stats = processor.getStats();
                std::cout << "\nPerformance Stats:" << std::endl;
                std::cout << "Average FPS: " << stats.avg_fps << std::endl;
                std::cout << "Average Latency: " << stats.avg_latency_ms << " ms" << std::endl;
                std::cout << "Frames Processed: " << stats.frames_processed << std::endl;
                std::cout << "Frames Dropped: " << stats.frames_dropped << std::endl;
                std::cout << "Memory Usage: " << stats.memory_usage_mb << " MB" << std::endl;
                
                last_stats_time = now;
            }
        }
        
        // Handle keyboard input
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 's') {
            // Save current frame
            if (!processed_frame.empty()) {
                std::string filename = "denoised_frame_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, processed_frame);
                std::cout << "Frame saved as: " << filename << std::endl;
            }
        }
    }
    
    // Cleanup
    processor.stopRealTimeProcessing();
    processor.stopCameraCapture();
    cv::destroyAllWindows();
    
    std::cout << "Real-time processing stopped." << std::endl;
    
    // Final statistics
    auto final_stats = processor.getStats();
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "Total frames processed: " << final_stats.frames_processed << std::endl;
    std::cout << "Average FPS: " << final_stats.avg_fps << std::endl;
    std::cout << "Peak FPS: " << final_stats.max_fps << std::endl;
}

void runBenchmark() {
    std::cout << "Running denoising benchmark..." << std::endl;
    
    // Create test image with noise
    cv::Mat clean_image = cv::Mat::zeros(512, 512, CV_8UC3);
    
    // Create a pattern for testing
    for (int y = 0; y < clean_image.rows; y++) {
        for (int x = 0; x < clean_image.cols; x++) {
            cv::Vec3b& pixel = clean_image.at<cv::Vec3b>(y, x);
            pixel[0] = static_cast<uchar>((x + y) % 256);
            pixel[1] = static_cast<uchar>((x * 2) % 256);
            pixel[2] = static_cast<uchar>((y * 2) % 256);
        }
    }
    
    // Add noise
    cv::Mat noisy_image;
    clean_image.copyTo(noisy_image);
    cv::Mat noise = cv::Mat::zeros(clean_image.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(50));
    noisy_image += noise;
    
    // Save test images
    cv::imwrite("test_clean.jpg", clean_image);
    cv::imwrite("test_noisy.jpg", noisy_image);
    
    // Initialize benchmark
    DenoiseBenchmark benchmark;
    
    // Run comprehensive benchmark
    auto results = benchmark.benchmarkAlgorithms(noisy_image, clean_image, 50);
    
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << std::left << std::setw(15) << "Algorithm" 
              << std::setw(12) << "Avg Time" 
              << std::setw(12) << "Min Time" 
              << std::setw(12) << "Max Time" 
              << std::setw(10) << "PSNR" 
              << std::setw(10) << "SSIM" 
              << std::setw(12) << "Memory" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& result : results) {
        std::string algo_name;
        switch (result.algorithm) {
            case DenoiseAlgorithm::BILATERAL: algo_name = "Bilateral"; break;
            case DenoiseAlgorithm::NON_LOCAL_MEANS: algo_name = "NLM"; break;
            case DenoiseAlgorithm::GAUSSIAN: algo_name = "Gaussian"; break;
            case DenoiseAlgorithm::ADAPTIVE_BILATERAL: algo_name = "Adaptive"; break;
        }
        
        std::cout << std::left << std::setw(15) << algo_name
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.avg_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.min_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.max_time_ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.psnr
                  << std::setw(10) << std::fixed << std::setprecision(4) << result.ssim
                  << std::setw(12) << result.memory_usage_mb << " MB" << std::endl;
    }
    
    std::cout << "\nBenchmark completed. Test images saved as test_clean.jpg and test_noisy.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Real-Time Image Denoising Engine" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Initialize CUDA
    if (!initializeCuda()) {
        std::cerr << "Error: Failed to initialize CUDA" << std::endl;
        return -1;
    }
    
    // Parse command line arguments
    std::string input_path, output_path;
    std::string algorithm_name = "bilateral";
    bool use_camera = false;
    bool run_benchmark_mode = false;
    
    // Default parameters
    DenoiseParams params;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                input_path = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_path = argv[++i];
            }
        } else if (arg == "-a" || arg == "--algorithm") {
            if (i + 1 < argc) {
                algorithm_name = argv[++i];
            }
        } else if (arg == "-c" || arg == "--camera") {
            use_camera = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            run_benchmark_mode = true;
        } else if (arg == "--sigma-color") {
            if (i + 1 < argc) {
                params.sigma_color = std::stof(argv[++i]);
            }
        } else if (arg == "--sigma-space") {
            if (i + 1 < argc) {
                params.sigma_space = std::stof(argv[++i]);
            }
        } else if (arg == "--h-param") {
            if (i + 1 < argc) {
                params.h = std::stof(argv[++i]);
            }
        } else if (arg == "--template-size") {
            if (i + 1 < argc) {
                params.template_window_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--search-size") {
            if (i + 1 < argc) {
                params.search_window_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--gaussian-sigma") {
            if (i + 1 < argc) {
                params.gaussian_sigma = std::stof(argv[++i]);
            }
        } else if (arg == "--kernel-size") {
            if (i + 1 < argc) {
                params.kernel_size = std::stoi(argv[++i]);
            }
        }
    }
    
    DenoiseAlgorithm algorithm = parseAlgorithm(algorithm_name);
    
    // Print configuration
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "Algorithm: " << algorithm_name << std::endl;
    std::cout << "Sigma Color: " << params.sigma_color << std::endl;
    std::cout << "Sigma Space: " << params.sigma_space << std::endl;
    std::cout << "H Parameter: " << params.h << std::endl;
    std::cout << "Template Size: " << params.template_window_size << std::endl;
    std::cout << "Search Size: " << params.search_window_size << std::endl;
    std::cout << "Gaussian Sigma: " << params.gaussian_sigma << std::endl;
    std::cout << "Kernel Size: " << params.kernel_size << std::endl;
    std::cout << std::endl;
    
    try {
        if (run_benchmark_mode) {
            // Run benchmark
            runBenchmark();
        } else if (use_camera) {
            // Real-time camera processing
            runRealTimeProcessing(algorithm, params);
        } else if (!input_path.empty()) {
            // Single image processing
            if (output_path.empty()) {
                // Generate output filename
                size_t dot_pos = input_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    output_path = input_path.substr(0, dot_pos) + "_denoised" + input_path.substr(dot_pos);
                } else {
                    output_path = input_path + "_denoised.jpg";
                }
            }
            
            processSingleImage(input_path, output_path, algorithm, params);
        } else {
            // No valid mode specified
            std::cout << "No input specified. Use -h for help." << std::endl;
            std::cout << "Quick start examples:" << std::endl;
            std::cout << "  " << argv[0] << " -i image.jpg                    # Process single image" << std::endl;
            std::cout << "  " << argv[0] << " -c -a bilateral                 # Real-time with bilateral filter" << std::endl;
            std::cout << "  " << argv[0] << " -b                              # Run benchmark" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    // Print memory statistics
    auto& memory_pool = GpuMemoryPool::getInstance();
    memory_pool.printMemoryStats();
    
    // Cleanup
    cleanupCuda();
    
    std::cout << "Application completed successfully." << std::endl;
    return 0;
}