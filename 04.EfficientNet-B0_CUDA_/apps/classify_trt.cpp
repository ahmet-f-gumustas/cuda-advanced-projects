#include "backends/tensorrt_backend.h"
#include "io/image_decoder.h"
#include "preprocess/preprocess.h"
#include "postprocess/postprocess.h"
#include "common/cuda_utils.h"
#include "common/timer.h"
#include "common/logger.h"
#include <iostream>
#include <iomanip>
#include <argparse/argparse.hpp>

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("classify_trt");
    
    program.add_argument("--model")
        .required()
        .help("Path to ONNX model file");
    
    program.add_argument("--image")
        .required()
        .help("Path to input image");
    
    program.add_argument("--fp16")
        .default_value(true)
        .implicit_value(true)
        .help("Use FP16 precision");
    
    program.add_argument("--int8")
        .default_value(false)
        .implicit_value(true)
        .help("Use INT8 precision");
    
    program.add_argument("--calib")
        .default_value(std::string(""))
        .help("Path to calibration file list for INT8");
    
    program.add_argument("--warmup")
        .default_value(5)
        .scan<'i', int>()
        .help("Number of warmup iterations");
    
    program.add_argument("--repeat")
        .default_value(50)
        .scan<'i', int>()
        .help("Number of timing iterations");
    
    program.add_argument("--class-names")
        .default_value(std::string(""))
        .help("Path to class names file");
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    
    try {
        // Initialize
        LOG_INFO("Initializing TensorRT backend...");
        auto backend = std::make_unique<TensorRTBackend>();
        
        backend->enable_fp16(program.get<bool>("--fp16"));
        
        if (program.get<bool>("--int8")) {
            backend->enable_int8(true, program.get<std::string>("--calib"));
        }
        
        // Load model
        std::string model_path = program.get<std::string>("--model");
        LOG_INFO("Loading model:", model_path);
        backend->load_model(model_path);
        
        // Create components
        auto decoder = create_image_decoder(true);
        auto preprocessor = std::make_unique<Preprocessor>(224, 224, false);  // TRT handles FP16 internally
        auto postprocessor = std::make_unique<Postprocessor>(program.get<std::string>("--class-names"));
        
        CudaStream stream;
        
        // Decode image
        std::string image_path = program.get<std::string>("--image");
        LOG_INFO("Processing image:", image_path);
        
        int width, height;
        DeviceMemory<uint8_t> d_image(224 * 224 * 3);
        
        if (!decoder->decode_to_device(image_path, d_image.get(), width, height, stream.get())) {
            auto image = decoder->decode(image_path);
            if (!image) {
                LOG_ERROR("Failed to decode image");
                return 1;
            }
            
            CUDA_CHECK(cudaMemcpyAsync(d_image.get(), image->data.data(), image->size(),
                                      cudaMemcpyHostToDevice, stream.get()));
            width = image->width;
            height = image->height;
        }
        
        // Allocate buffers
        size_t input_size = backend->get_input_size();
        size_t output_size = backend->get_output_size();
        
        DeviceMemory<char> d_input(input_size);
        DeviceMemory<float> d_output(output_size / sizeof(float));
        
        // Preprocess
        preprocessor->process(d_image.get(), height, width, d_input.get(), stream.get());
        
        // Warmup
        LOG_INFO("Running warmup iterations...");
        for (int i = 0; i < program.get<int>("--warmup"); ++i) {
            backend->infer(d_input.get(), d_output.get(), stream.get());
        }
        stream.synchronize();
        
        // Timing
        LOG_INFO("Running timed iterations...");
        CudaTimer timer;
        
        for (int i = 0; i < program.get<int>("--repeat"); ++i) {
            timer.start(stream.get());
            backend->infer(d_input.get(), d_output.get(), stream.get());
            timer.stop(stream.get());
        }
        
        // Get results
        auto results = postprocessor->process(d_output.get(), 1000, 5, stream.get());
        
        // Print results
        std::cout << "\n=== Classification Results ===" << std::endl;
        for (int i = 0; i < results.size(); ++i) {
            std::cout << std::setw(2) << i+1 << ". "
                      << std::left << std::setw(20) << results[i].class_name
                      << " (ID: " << std::setw(4) << results[i].class_id << ")"
                      << " - " << std::fixed << std::setprecision(2)
                      << results[i].probability * 100 << "%" << std::endl;
        }
        
        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Precision: ";
        if (program.get<bool>("--int8")) {
            std::cout << "INT8";
        } else if (program.get<bool>("--fp16")) {
            std::cout << "FP16";
        } else {
            std::cout << "FP32";
        }
        std::cout << std::endl;
        
        std::cout << "Mean latency: " << std::fixed << std::setprecision(3)
                  << timer.get_mean() << " ms" << std::endl;
        std::cout << "P50 latency:  " << timer.get_percentile(0.5) << " ms" << std::endl;
        std::cout << "P95 latency:  " << timer.get_percentile(0.95) << " ms" << std::endl;
        std::cout << "P99 latency:  " << timer.get_percentile(0.99) << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error:", e.what());
        return 1;
    }
    
    return 0;
}