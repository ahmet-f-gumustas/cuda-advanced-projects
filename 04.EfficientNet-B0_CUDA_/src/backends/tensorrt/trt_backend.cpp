#include "backends/tensorrt_backend.h"
#include "backends/tensorrt/int8_calibrator.h"
#include "common/cuda_utils.h"
#include "common/logger.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <memory>

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kERROR: LOG_ERROR("[TRT]", msg); break;
            case Severity::kWARNING: LOG_WARNING("[TRT]", msg); break;
            case Severity::kINFO: LOG_INFO("[TRT]", msg); break;
            default: LOG_DEBUG("[TRT]", msg); break;
        }
    }
};

struct TensorRTBackend::Impl {
    TRTLogger logger;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // Binding bilgileri
    int input_binding_idx;
    int output_binding_idx;
    size_t input_size;
    size_t output_size;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    
    bool use_fp16 = true;
    bool use_int8 = false;
};

TensorRTBackend::TensorRTBackend() : impl_(std::make_unique<Impl>()) {}
TensorRTBackend::~TensorRTBackend() = default;

void TensorRTBackend::load_model(const std::string& model_path) {
    // Önce engine dosyası var mı kontrol et
    std::string engine_path = model_path + ".engine";
    std::ifstream engine_file(engine_path, std::ios::binary);
    
    if (engine_file.good()) {
        // Engine'i yükle
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        engine_file.read(engine_data.data(), size);
        
        auto runtime = nvinfer1::createInferRuntime(impl_->logger);
        impl_->engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
        runtime->destroy();
        
        LOG_INFO("Loaded TensorRT engine from:", engine_path);
    } else {
        // ONNX'ten engine oluştur
        build_engine_from_onnx(model_path);
        
        // Engine'i kaydet
        save_engine(engine_path);
    }
    
    // Context oluştur
    impl_->context.reset(impl_->engine->createExecutionContext());
    
    // Binding bilgilerini al
    for (int i = 0; i < impl_->engine->getNbBindings(); ++i) {
        if (impl_->engine->bindingIsInput(i)) {
            impl_->input_binding_idx = i;
            auto dims = impl_->engine->getBindingDimensions(i);
            impl_->input_shape.clear();
            for (int j = 0; j < dims.nbDims; ++j) {
                impl_->input_shape.push_back(dims.d[j]);
            }
        } else {
            impl_->output_binding_idx = i;
            auto dims = impl_->engine->getBindingDimensions(i);
            impl_->output_shape.clear();
            for (int j = 0; j < dims.nbDims; ++j) {
                impl_->output_shape.push_back(dims.d[j]);
            }
        }
    }
    
    // Boyutları hesapla
    impl_->input_size = impl_->engine->getBindingComponentsPerElement(impl_->input_binding_idx);
    impl_->output_size = impl_->engine->getBindingComponentsPerElement(impl_->output_binding_idx);
    
    for (auto dim : impl_->input_shape) {
        impl_->input_size *= dim;
    }
    for (auto dim : impl_->output_shape) {
        impl_->output_size *= dim;
    }
    
    // Element boyutu ekle
    auto input_type = impl_->engine->getBindingDataType(impl_->input_binding_idx);
    impl_->input_size *= (input_type == nvinfer1::DataType::kHALF ? 2 : 4);
    impl_->output_size *= 4;  // Output FP32
}

void TensorRTBackend::build_engine_from_onnx(const std::string& onnx_path) {
    auto builder = nvinfer1::createInferBuilder(impl_->logger);
    auto network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto parser = nvonnxparser::createParser(*network, impl_->logger);
    
    // ONNX'i parse et
    if (!parser->parseFromFile(onnx_path.c_str(), 
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        LOG_ERROR("Failed to parse ONNX file:", onnx_path);
        throw std::runtime_error("ONNX parse error");
    }
    
    // Builder config
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);  // 1GB workspace
    
    if (impl_->use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    if (impl_->use_int8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // INT8 calibrator ekle
        auto calibrator = std::make_unique<Int8EntropyCalibrator>(
            1, 224, 224, "tools/calib_list.txt", "calibration.cache");
        config->setInt8Calibrator(calibrator.get());
    }
    
    // Engine oluştur
    impl_->engine.reset(builder->buildEngineWithConfig(*network, *config));
    
    // Temizlik
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    
    LOG_INFO("Built TensorRT engine from ONNX");
}

void TensorRTBackend::save_engine(const std::string& path) {
    auto serialized = impl_->engine->serialize();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    serialized->destroy();
    LOG_INFO("Saved TensorRT engine to:", path);
}

void TensorRTBackend::infer(const void* input, void* output, cudaStream_t stream) {
    void* bindings[] = {const_cast<void*>(input), output};
    
    if (!impl_->context->enqueueV2(bindings, stream, nullptr)) {
        LOG_ERROR("TensorRT inference failed");
        throw std::runtime_error("TensorRT inference error");
    }
}

void TensorRTBackend::enable_fp16(bool enable) {
    impl_->use_fp16 = enable;
}

void TensorRTBackend::enable_int8(bool enable, const std::string& calib_data) {
    impl_->use_int8 = enable;
    // Kalibrasyon verisi path'i saklanabilir
}

size_t TensorRTBackend::get_input_size() const { return impl_->input_size; }
size_t TensorRTBackend::get_output_size() const { return impl_->output_size; }
std::vector<int64_t> TensorRTBackend::get_input_shape() const { return impl_->input_shape; }
std::vector<int64_t> TensorRTBackend::get_output_shape() const { return impl_->output_shape; }