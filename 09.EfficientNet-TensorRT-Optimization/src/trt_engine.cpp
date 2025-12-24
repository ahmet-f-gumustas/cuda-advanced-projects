#include "trt_engine.h"
#include "calibrator.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace efficientnet {

TrtEngine::TrtEngine() {
    cudaStreamCreate(&stream_);
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

TrtEngine::~TrtEngine() {
    freeBuffers();

    if (stream_) cudaStreamDestroy(stream_);
    if (start_event_) cudaEventDestroy(start_event_);
    if (stop_event_) cudaEventDestroy(stop_event_);
}

bool TrtEngine::buildEngine(const EngineConfig& config) {
    config_ = config;

    // Try to load cached engine first
    if (!config.engine_path.empty()) {
        std::ifstream file(config.engine_path, std::ios::binary);
        if (file.good()) {
            std::cout << "Loading cached engine from: " << config.engine_path << std::endl;
            if (loadEngine(config.engine_path)) {
                return true;
            }
            std::cout << "Failed to load cached engine, rebuilding..." << std::endl;
        }
    }

    // Build from ONNX
    if (!buildFromOnnx(config)) {
        return false;
    }

    // Save engine for next time
    if (!config.engine_path.empty()) {
        saveEngine(config.engine_path);
    }

    return true;
}

bool TrtEngine::buildFromOnnx(const EngineConfig& config) {
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return false;
    }

    // Create network with explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        return false;
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return false;
    }

    // Parse ONNX model
    if (!parser->parseFromFile(config.onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file: " << config.onnx_path << std::endl;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    std::cout << "Successfully parsed ONNX model" << std::endl;

    // Create builder config
    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!builder_config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }

    // Set workspace size
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                        config.workspace_size);

    // Set precision
    std::unique_ptr<Int8EntropyCalibrator> calibrator;

    switch (config.precision) {
        case Precision::FP16:
            if (builder->platformHasFastFp16()) {
                builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
                std::cout << "Using FP16 precision" << std::endl;
            } else {
                std::cout << "FP16 not supported, using FP32" << std::endl;
            }
            break;

        case Precision::INT8:
            if (builder->platformHasFastInt8()) {
                builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
                // Also enable FP16 for layers that don't support INT8
                if (builder->platformHasFastFp16()) {
                    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
                }

                // Create calibrator
                if (!config.calibration_data_path.empty()) {
                    calibrator = std::make_unique<Int8EntropyCalibrator>(
                        config.calibration_data_path,
                        config.engine_path + ".calibration",
                        config.calibration_batch_size,
                        224, 224, 3
                    );
                    builder_config->setInt8Calibrator(calibrator.get());
                    std::cout << "Using INT8 precision with calibration" << std::endl;
                } else {
                    std::cerr << "INT8 requires calibration data path" << std::endl;
                    return false;
                }
            } else {
                std::cout << "INT8 not supported, using FP32" << std::endl;
            }
            break;

        case Precision::FP32:
        default:
            std::cout << "Using FP32 precision" << std::endl;
            break;
    }

    // Enable DLA if requested
    if (config.enable_dla) {
        if (builder->getNbDLACores() > 0) {
            builder_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            builder_config->setDLACore(config.dla_core);
            builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            std::cout << "Using DLA core " << config.dla_core << std::endl;
        } else {
            std::cout << "DLA not available" << std::endl;
        }
    }

    // Build serialized network
    std::cout << "Building TensorRT engine (this may take a while)..." << std::endl;
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builder_config));
    if (!serialized_engine) {
        std::cerr << "Failed to build serialized engine" << std::endl;
        return false;
    }

    // Create runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "Failed to create runtime" << std::endl;
        return false;
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(
        serialized_engine->data(), serialized_engine->size()));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Get input/output dimensions
    input_dims_ = engine_->getTensorShape(engine_->getIOTensorName(0));
    output_dims_ = engine_->getTensorShape(engine_->getIOTensorName(1));

    std::cout << "Engine built successfully!" << std::endl;
    std::cout << getEngineInfo() << std::endl;

    return allocateBuffers(config.max_batch_size);
}

bool TrtEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read engine file" << std::endl;
        return false;
    }

    // Create runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "Failed to create runtime" << std::endl;
        return false;
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Get dimensions
    input_dims_ = engine_->getTensorShape(engine_->getIOTensorName(0));
    output_dims_ = engine_->getTensorShape(engine_->getIOTensorName(1));

    std::cout << "Engine loaded successfully!" << std::endl;
    std::cout << getEngineInfo() << std::endl;

    return allocateBuffers(config_.max_batch_size);
}

bool TrtEngine::saveEngine(const std::string& engine_path) {
    if (!engine_) {
        std::cerr << "No engine to save" << std::endl;
        return false;
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    if (!serialized) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << engine_path << std::endl;
        return false;
    }

    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cout << "Engine saved to: " << engine_path << std::endl;

    return true;
}

bool TrtEngine::allocateBuffers(int batch_size) {
    freeBuffers();

    // Calculate sizes
    input_size_ = batch_size;
    for (int i = 0; i < input_dims_.nbDims; i++) {
        if (input_dims_.d[i] > 0) {
            input_size_ *= input_dims_.d[i];
        }
    }
    input_size_ *= sizeof(float);

    output_size_ = batch_size;
    for (int i = 0; i < output_dims_.nbDims; i++) {
        if (output_dims_.d[i] > 0) {
            output_size_ *= output_dims_.d[i];
        }
    }
    output_size_ *= sizeof(float);

    // Allocate device memory
    if (cudaMalloc(&device_input_, input_size_) != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return false;
    }

    if (cudaMalloc(&device_output_, output_size_) != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer" << std::endl;
        cudaFree(device_input_);
        device_input_ = nullptr;
        return false;
    }

    return true;
}

void TrtEngine::freeBuffers() {
    if (device_input_) {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }
    if (device_output_) {
        cudaFree(device_output_);
        device_output_ = nullptr;
    }
}

bool TrtEngine::infer(const float* input_data, float* output_data, int batch_size) {
    if (!isReady()) {
        std::cerr << "Engine not ready" << std::endl;
        return false;
    }

    // Start timing
    cudaEventRecord(start_event_, stream_);

    // Copy input to device
    size_t input_bytes = batch_size;
    for (int i = 1; i < input_dims_.nbDims; i++) {
        input_bytes *= input_dims_.d[i];
    }
    input_bytes *= sizeof(float);

    cudaMemcpyAsync(device_input_, input_data, input_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // Set tensor addresses
    context_->setTensorAddress(engine_->getIOTensorName(0), device_input_);
    context_->setTensorAddress(engine_->getIOTensorName(1), device_output_);

    // Run inference
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "Inference failed" << std::endl;
        return false;
    }

    // Copy output to host
    size_t output_bytes = batch_size;
    for (int i = 1; i < output_dims_.nbDims; i++) {
        output_bytes *= output_dims_.d[i];
    }
    output_bytes *= sizeof(float);

    cudaMemcpyAsync(output_data, device_output_, output_bytes,
                    cudaMemcpyDeviceToHost, stream_);

    // Stop timing
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);

    cudaEventElapsedTime(&last_inference_time_ms_, start_event_, stop_event_);

    return true;
}

std::string TrtEngine::getEngineInfo() const {
    std::stringstream ss;
    ss << "=== TensorRT Engine Info ===" << std::endl;

    if (!engine_) {
        ss << "No engine loaded" << std::endl;
        return ss.str();
    }

    ss << "Input: " << engine_->getIOTensorName(0) << " [";
    for (int i = 0; i < input_dims_.nbDims; i++) {
        ss << input_dims_.d[i];
        if (i < input_dims_.nbDims - 1) ss << ", ";
    }
    ss << "]" << std::endl;

    ss << "Output: " << engine_->getIOTensorName(1) << " [";
    for (int i = 0; i < output_dims_.nbDims; i++) {
        ss << output_dims_.d[i];
        if (i < output_dims_.nbDims - 1) ss << ", ";
    }
    ss << "]" << std::endl;

    return ss.str();
}

// Utility functions
std::vector<std::string> loadClassNames(const std::string& filepath) {
    std::vector<std::string> class_names;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open class names file: " << filepath << std::endl;
        return class_names;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_names.push_back(line);
        }
    }

    return class_names;
}

InferenceResult getTopPrediction(const float* output, int num_classes,
                                  const std::vector<std::string>& class_names) {
    InferenceResult result;
    result.class_id = 0;
    result.confidence = output[0];

    for (int i = 1; i < num_classes; i++) {
        if (output[i] > result.confidence) {
            result.confidence = output[i];
            result.class_id = i;
        }
    }

    if (result.class_id < static_cast<int>(class_names.size())) {
        result.class_name = class_names[result.class_id];
    } else {
        result.class_name = "class_" + std::to_string(result.class_id);
    }

    return result;
}

std::vector<InferenceResult> getTopKPredictions(const float* output, int num_classes,
                                                 const std::vector<std::string>& class_names, int k) {
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < num_classes; i++) {
        scores.emplace_back(output[i], i);
    }

    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      std::greater<std::pair<float, int>>());

    std::vector<InferenceResult> results;
    for (int i = 0; i < k && i < static_cast<int>(scores.size()); i++) {
        InferenceResult result;
        result.class_id = scores[i].second;
        result.confidence = scores[i].first;

        if (result.class_id < static_cast<int>(class_names.size())) {
            result.class_name = class_names[result.class_id];
        } else {
            result.class_name = "class_" + std::to_string(result.class_id);
        }

        results.push_back(result);
    }

    return results;
}

}  // namespace efficientnet
