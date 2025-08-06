#include "backends/ort_backend.h"
#include "common/cuda_utils.h"
#include "common/logger.h"
#include <onnxruntime_cxx_api.h>
#include <vector>

struct OrtBackend::Impl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info_cuda;
    
    // Model bilgileri
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    size_t input_size;
    size_t output_size;
    
    // IOBinding için
    std::unique_ptr<Ort::IoBinding> io_binding;
    bool use_cuda_graphs = false;
    bool use_fp16 = false;
    
    // Input/output isimleri
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    
    Impl(int device_id) 
        : env(ORT_LOGGING_LEVEL_WARNING, "efficientnet"),
          memory_info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, device_id, OrtMemType::OrtMemTypeDefault) {
        
        // Session options
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // CUDA EP ekle
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = device_id;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.do_copy_in_default_stream = 1;  // Güvenli default
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::DEFAULT;
        
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        
        LOG_INFO("ORT Backend initialized with CUDA device", device_id);
    }
    
    void set_cudnn_conv_algo_search(const std::string& mode) {
        // Session oluşturulmadan önce çağrılmalı
        if (session) {
            LOG_WARNING("Cannot change cuDNN conv algo search after session creation");
            return;
        }
        
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.do_copy_in_default_stream = 1;
        
        if (mode == "EXHAUSTIVE") {
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
        } else if (mode == "HEURISTIC") {
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::HEURISTIC;
        } else {
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::DEFAULT;
        }
        
        session_options = Ort::SessionOptions();
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }
};

OrtBackend::OrtBackend(int device_id) : impl_(std::make_unique<Impl>(device_id)) {}
OrtBackend::~OrtBackend() = default;

void OrtBackend::load_model(const std::string& model_path) {
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), 
                                                     impl_->session_options);
    
    // Input/output bilgilerini al
    size_t num_inputs = impl_->session->GetInputCount();
    size_t num_outputs = impl_->session->GetOutputCount();
    
    LOG_INFO("Model loaded:", model_path, "- Inputs:", num_inputs, "Outputs:", num_outputs);
    
    // Input bilgileri
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
        impl_->input_names_str.push_back(input_name.get());
        
        Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        impl_->input_shape = tensor_info.GetShape();
        
        LOG_INFO("Input", i, ":", impl_->input_names_str.back(), 
                 "shape:", impl_->input_shape);
    }
    
    // Output bilgileri
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
        impl_->output_names_str.push_back(output_name.get());
        
        Ort::TypeInfo type_info = impl_->session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        impl_->output_shape = tensor_info.GetShape();
        
        LOG_INFO("Output", i, ":", impl_->output_names_str.back(),
                 "shape:", impl_->output_shape);
    }
    
    // İsimleri char* dizisine dönüştür
    for (const auto& name : impl_->input_names_str) {
        impl_->input_names.push_back(name.c_str());
    }
    for (const auto& name : impl_->output_names_str) {
        impl_->output_names.push_back(name.c_str());
    }
    
    // Boyutları hesapla
    impl_->input_size = 1;
    for (auto dim : impl_->input_shape) {
        if (dim > 0) impl_->input_size *= dim;
    }
    impl_->input_size *= (impl_->use_fp16 ? 2 : 4);  // FP16 veya FP32
    
    impl_->output_size = 1;
    for (auto dim : impl_->output_shape) {
        if (dim > 0) impl_->output_size *= dim;
    }
    impl_->output_size *= 4;  // Output her zaman FP32
    
    // IOBinding oluştur
    impl_->io_binding = std::make_unique<Ort::IoBinding>(*impl_->session);
}

void OrtBackend::infer(const void* input, void* output, cudaStream_t stream) {
    if (impl_->use_cuda_graphs && impl_->io_binding) {
        // IOBinding ile çalıştır (adresler sabit kalmalı)
        impl_->io_binding->ClearBoundInputs();
        impl_->io_binding->ClearBoundOutputs();
        
        // Input bind
        Ort::Value input_tensor = Ort::Value::CreateTensor(
            impl_->memory_info_cuda, const_cast<void*>(input), 
            impl_->input_size, impl_->input_shape.data(), 
            impl_->input_shape.size());
        
        impl_->io_binding->BindInput(impl_->input_names[0], input_tensor);
        
        // Output bind
        Ort::Value output_tensor = Ort::Value::CreateTensor(
            impl_->memory_info_cuda, output, impl_->output_size,
            impl_->output_shape.data(), impl_->output_shape.size());
        
        impl_->io_binding->BindOutput(impl_->output_names[0], output_tensor);
        
        // Run with IOBinding
        impl_->session->Run(Ort::RunOptions{nullptr}, *impl_->io_binding);
    } else {
        // Normal run
        Ort::Value input_tensor = Ort::Value::CreateTensor(
            impl_->memory_info_cuda, const_cast<void*>(input),
            impl_->input_size, impl_->input_shape.data(),
            impl_->input_shape.size());
        
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(), &input_tensor, 1,
            impl_->output_names.data(), impl_->output_names.size());
        
        // Çıktıyı kopyala
        const float* output_data = output_tensors[0].GetTensorData<float>();
        CUDA_CHECK(cudaMemcpyAsync(output, output_data, impl_->output_size,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void OrtBackend::infer_batch(const void* input, void* output, 
                            int batch_size, cudaStream_t stream) {
    // Batch boyutunu güncelle
    auto batch_shape = impl_->input_shape;
    batch_shape[0] = batch_size;
    
    size_t batch_input_size = impl_->input_size * batch_size;
    size_t batch_output_size = impl_->output_size * batch_size;
    
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        impl_->memory_info_cuda, const_cast<void*>(input),
        batch_input_size, batch_shape.data(), batch_shape.size());
    
    auto output_tensors = impl_->session->Run(
        Ort::RunOptions{nullptr},
        impl_->input_names.data(), &input_tensor, 1,
        impl_->output_names.data(), impl_->output_names.size());
    
    const float* output_data = output_tensors[0].GetTensorData<float>();
    CUDA_CHECK(cudaMemcpyAsync(output, output_data, batch_output_size,
                               cudaMemcpyDeviceToDevice, stream));
}

size_t OrtBackend::get_input_size() const { return impl_->input_size; }
size_t OrtBackend::get_output_size() const { return impl_->output_size; }
std::vector<int64_t> OrtBackend::get_input_shape() const { return impl_->input_shape; }
std::vector<int64_t> OrtBackend::get_output_shape() const { return impl_->output_shape; }

void OrtBackend::enable_fp16(bool enable) {
    impl_->use_fp16 = enable;
    // Input size'ı güncelle
    if (impl_->input_shape.size() > 0) {
        impl_->input_size = 1;
        for (auto dim : impl_->input_shape) {
            if (dim > 0) impl_->input_size *= dim;
        }
        impl_->input_size *= (enable ? 2 : 4);
    }
}

void OrtBackend::enable_cuda_graphs(bool enable) {
    impl_->use_cuda_graphs = enable;
    LOG_INFO("CUDA Graphs", enable ? "enabled" : "disabled");
}

void OrtBackend::set_cudnn_conv_algo_search(const std::string& mode) {
    impl_->set_cudnn_conv_algo_search(mode);
}