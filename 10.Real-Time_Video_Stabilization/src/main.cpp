#include "video_stabilizer.hpp"
#include <iostream>
#include <string>
#include <getopt.h>

void printUsage(const char* program_name) {
    std::cout << "CUDA Real-Time Video Stabilization" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input <file>      Input video file" << std::endl;
    std::cout << "  -o, --output <file>     Output video file (default: output.mp4)" << std::endl;
    std::cout << "  -c, --camera <id>       Use camera for real-time stabilization (default: 0)" << std::endl;
    std::cout << "  -s, --smoothing <n>     Smoothing radius in frames (default: 30)" << std::endl;
    std::cout << "  -r, --crop <ratio>      Crop ratio 0.0-1.0 (default: 0.9)" << std::endl;
    std::cout << "  -g, --gpu               Force GPU processing (default: auto)" << std::endl;
    std::cout << "  -p, --compare           Show original vs stabilized comparison" << std::endl;
    std::cout << "  -m, --method <type>     Smoothing method: average, gaussian, kalman (default: gaussian)" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " -i shaky_video.mp4 -o stable_video.mp4" << std::endl;
    std::cout << "  " << program_name << " -c 0 -p                    # Real-time from webcam" << std::endl;
    std::cout << "  " << program_name << " -i input.mp4 -s 50 -r 0.85 # Custom smoothing" << std::endl;
}

void printSystemInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "====================================" << std::endl;
    std::cout << "CUDA Video Stabilization System Info" << std::endl;
    std::cout << "====================================" << std::endl;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    }

    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Default configuration
    std::string input_file;
    std::string output_file = "output.mp4";
    int camera_id = -1;
    bool use_camera = false;

    cuda_stabilizer::StabilizerConfig config;
    cuda_stabilizer::SmoothingMethod smoothing_method = cuda_stabilizer::SmoothingMethod::GAUSSIAN;

    // Parse command line arguments
    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"camera", required_argument, 0, 'c'},
        {"smoothing", required_argument, 0, 's'},
        {"crop", required_argument, 0, 'r'},
        {"gpu", no_argument, 0, 'g'},
        {"compare", no_argument, 0, 'p'},
        {"method", required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "i:o:c:s:r:gpm:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'c':
                camera_id = std::stoi(optarg);
                use_camera = true;
                break;
            case 's':
                config.smoothing_radius = std::stoi(optarg);
                break;
            case 'r':
                config.crop_ratio = std::stof(optarg);
                break;
            case 'g':
                config.use_gpu = true;
                break;
            case 'p':
                config.show_comparison = true;
                break;
            case 'm': {
                std::string method = optarg;
                if (method == "average") {
                    smoothing_method = cuda_stabilizer::SmoothingMethod::MOVING_AVERAGE;
                } else if (method == "gaussian") {
                    smoothing_method = cuda_stabilizer::SmoothingMethod::GAUSSIAN;
                } else if (method == "kalman") {
                    smoothing_method = cuda_stabilizer::SmoothingMethod::KALMAN;
                } else {
                    std::cerr << "Unknown smoothing method: " << method << std::endl;
                    return 1;
                }
                break;
            }
            case 'h':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }

    // Validate arguments
    if (!use_camera && input_file.empty()) {
        std::cerr << "Error: No input specified. Use -i for video file or -c for camera." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Print system info
    printSystemInfo();

    // Create stabilizer
    cuda_stabilizer::VideoStabilizer stabilizer(config);

    // Process
    if (use_camera) {
        std::cout << "Starting real-time stabilization from camera " << camera_id << "..." << std::endl;
        std::cout << "Press 'q' to quit" << std::endl;
        std::cout << std::endl;

        if (!stabilizer.startRealTimeProcessing(camera_id)) {
            std::cerr << "Error: Failed to start real-time processing" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Processing video file: " << input_file << std::endl;
        std::cout << "Output file: " << output_file << std::endl;
        std::cout << "Smoothing radius: " << config.smoothing_radius << " frames" << std::endl;
        std::cout << "Crop ratio: " << config.crop_ratio << std::endl;
        std::cout << std::endl;

        if (!stabilizer.processVideo(input_file, output_file)) {
            std::cerr << "Error: Failed to process video" << std::endl;
            return 1;
        }
    }

    // Print statistics
    std::cout << std::endl;
    std::cout << "Processing Statistics:" << std::endl;
    std::cout << "  Total frames processed: " << stabilizer.getFrameCount() << std::endl;
    std::cout << "  Average processing time: " << stabilizer.getAverageProcessingTime() << " ms/frame" << std::endl;
    std::cout << "  Average FPS: " << stabilizer.getFPS() << std::endl;

    return 0;
}
