#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace efficientnet {

// TensorRT Logger
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// Precision modes
enum class Precision {
    FP32,
    FP16,
    INT8
};

// Engine configuration
struct EngineConfig {
    std::string onnx_path;
    std::string engine_path;
    Precision precision = Precision::FP16;
    int max_batch_size = 1;
    size_t workspace_size = 1ULL << 30;  // 1GB
    bool enable_dla = false;
    int dla_core = 0;
    std::string calibration_data_path;
    int calibration_batch_size = 8;
};

// Inference result
struct InferenceResult {
    int class_id;
    float confidence;
    std::string class_name;
    float inference_time_ms;
};

// TensorRT Engine wrapper
class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    // Build engine from ONNX
    bool buildEngine(const EngineConfig& config);

    // Load serialized engine
    bool loadEngine(const std::string& engine_path);

    // Save engine to file
    bool saveEngine(const std::string& engine_path);

    // Run inference
    bool infer(const float* input_data, float* output_data, int batch_size = 1);

    // Get input/output dimensions
    nvinfer1::Dims getInputDims() const { return input_dims_; }
    nvinfer1::Dims getOutputDims() const { return output_dims_; }

    // Get timing info
    float getLastInferenceTime() const { return last_inference_time_ms_; }

    // Check if engine is ready
    bool isReady() const { return engine_ != nullptr && context_ != nullptr; }

    // Get engine info
    std::string getEngineInfo() const;

private:
    // Build engine from ONNX file
    bool buildFromOnnx(const EngineConfig& config);

    // Allocate device buffers
    bool allocateBuffers(int batch_size);

    // Free device buffers
    void freeBuffers();

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Buffer management
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;

    // Dimensions
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;

    // CUDA stream
    cudaStream_t stream_ = nullptr;

    // Timing
    float last_inference_time_ms_ = 0.0f;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    // Engine config cache
    EngineConfig config_;
};

// Utility functions
std::vector<std::string> loadClassNames(const std::string& filepath);
InferenceResult getTopPrediction(const float* output, int num_classes,
                                  const std::vector<std::string>& class_names);
std::vector<InferenceResult> getTopKPredictions(const float* output, int num_classes,
                                                 const std::vector<std::string>& class_names, int k);

}  // namespace efficientnet

#endif  // TRT_ENGINE_H
