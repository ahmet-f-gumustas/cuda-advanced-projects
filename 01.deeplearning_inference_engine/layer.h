#pragma once

#include "tensor.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <chrono>

namespace deep_engine {

// Layer configuration için modern C++ kullanımı
struct LayerConfig {
    std::unordered_map<std::string, std::any> params;
    
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const {
        auto it = params.find(key);
        if (it != params.end()) {
            try {
                return std::any_cast<T>(it->second);
            } catch (const std::bad_any_cast&) {
                return default_value;
            }
        }
        return default_value;
    }
    
    template<typename T>
    void set(const std::string& key, const T& value) {
        params[key] = value;
    }
};

// Execution context - stream, cublas handle vs. gibi şeyleri tutar
class ExecutionContext {
private:
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
    bool enable_profiling_;
    std::unordered_map<std::string, double> timings_;
    
public:
    ExecutionContext();
    ~ExecutionContext();
    
    cudaStream_t stream() const { return stream_; }
    cublasHandle_t cublas() const { return cublas_handle_; }
    cudnnHandle_t cudnn() const { return cudnn_handle_; }
    
    void synchronize() { cudaStreamSynchronize(stream_); }
    
    // Profiling utilities
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    std::unordered_map<std::string, double> get_timings() const { return timings_; }
    void enable_profiling(bool enable) { enable_profiling_ = enable; }
};

// Forward pass modları
enum class InferenceMode {
    NORMAL,      // Standard inference
    INT8,        // INT8 quantized inference
    MIXED,       // Mixed precision (FP16 compute, FP32 accumulate)
    TENSOR_RT    // TensorRT compatible mode
};

// Layer interface - PyTorch'tan ilham alındı ama daha optimize
class Layer {
protected:
    std::string name_;
    LayerConfig config_;
    bool trainable_;
    InferenceMode mode_;
    
    // Weights and bias
    std::unordered_map<std::string, Tensor> parameters_;
    
    // Cached computations for optimization
    mutable std::unordered_map<std::string, Tensor> cache_;
    
    // Layer-specific workspace for cuDNN operations
    mutable void* workspace_;
    mutable size_t workspace_size_;
    
public:
    Layer(const std::string& name = "unnamed_layer") 
        : name_(name), trainable_(false), mode_(InferenceMode::NORMAL),
          workspace_(nullptr), workspace_size_(0) {}
    
    virtual ~Layer() {
        if (workspace_) {
            cudaFree(workspace_);
        }
    }
    
    // Ana forward fonksiyonu - pure virtual
    virtual Tensor forward(const Tensor& input, ExecutionContext& ctx) = 0;
    
    // Batch forward - bazı layer'lar için optimize edilebilir
    virtual std::vector<Tensor> forward_batch(const std::vector<Tensor>& inputs, ExecutionContext& ctx) {
        std::vector<Tensor> outputs;
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            outputs.push_back(forward(input, ctx));
        }
        return outputs;
    }
    
    // Layer fusion için gerekli
    virtual bool can_fuse_with(const Layer* next) const { return false; }
    virtual std::unique_ptr<Layer> fuse_with(const Layer* next) const { return nullptr; }
    
    // Quantization support
    virtual void quantize_weights(int bits = 8);
    virtual bool supports_int8() const { return false; }
    
    // Shape inference - onnx gibi frameworklerle uyumluluk için
    virtual std::vector<int> infer_output_shape(const std::vector<int>& input_shape) const = 0;
    
    // Memory requirements
    virtual size_t get_workspace_size() const { return workspace_size_; }
    virtual size_t get_parameter_bytes() const;
    
    // Parameter access
    void set_parameter(const std::string& name, const Tensor& param) {
        parameters_[name] = param;
    }
    
    Tensor& get_parameter(const std::string& name) {
        return parameters_[name];
    }
    
    const std::unordered_map<std::string, Tensor>& parameters() const {
        return parameters_;
    }
    
    // Configuration
    void set_config(const LayerConfig& config) { config_ = config; }
    const LayerConfig& config() const { return config_; }
    
    // Mode settings
    void set_inference_mode(InferenceMode mode) { mode_ = mode; }
    InferenceMode inference_mode() const { return mode_; }
    
    // Utility
    const std::string& name() const { return name_; }
    virtual std::string type() const = 0;
    virtual void print_info() const;
    
    // Serialization
    virtual void save_weights(const std::string& path) const;
    virtual void load_weights(const std::string& path);
    
protected:
    // Helper method for workspace allocation
    void ensure_workspace(size_t required_size) const {
        if (required_size > workspace_size_) {
            if (workspace_) {
                cudaFree(workspace_);
            }
            cudaMalloc(&workspace_, required_size);
            workspace_size_ = required_size;
        }
    }
};

// Layer factory pattern
class LayerFactory {
private:
    using CreatorFunc = std::function<std::unique_ptr<Layer>(const LayerConfig&)>;
    std::unordered_map<std::string, CreatorFunc> creators_;
    
    LayerFactory() = default;  // Singleton
    
public:
    static LayerFactory& instance() {
        static LayerFactory factory;
        return factory;
    }
    
    void register_layer(const std::string& type, CreatorFunc creator) {
        creators_[type] = creator;
    }
    
    std::unique_ptr<Layer> create(const std::string& type, const LayerConfig& config) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second(config);
        }
        throw std::runtime_error("Unknown layer type: " + type);
    }
};

// Macro for easy layer registration
#define REGISTER_LAYER(TYPE, CLASS) \
    static bool _registered_##CLASS = []() { \
        LayerFactory::instance().register_layer(TYPE, \
            [](const LayerConfig& config) { \
                return std::make_unique<CLASS>(config); \
            }); \
        return true; \
    }();

} // namespace deep_engine