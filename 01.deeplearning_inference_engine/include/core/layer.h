#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include "tensor.h"
#include "types.h"

namespace deep_engine {

// Forward declarations
class ExecutionContext;

class Layer {
public:
    Layer(const std::string& name = "") : name_(name), trainable_(false) {}
    virtual ~Layer() = default;
    
    // Core interface
    virtual Tensor forward(const Tensor& input, ExecutionContext& ctx) = 0;
    
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs, ExecutionContext& ctx) {
        assert(inputs.size() == 1);
        return {forward(inputs[0], ctx)};
    }
    
    // Layer properties
    virtual std::string type() const = 0;
    virtual size_t num_params() const { return 0; }
    virtual size_t flops(const std::vector<int>& input_shape) const { return 0; }
    
    // Serialization
    virtual void save_params(std::ostream& os) const {}
    virtual void load_params(std::istream& is) {}
    
    // Quantization support
    virtual bool supports_quantization() const { return false; }
    virtual void quantize(int bits = 8) {}
    
    // Fusion support
    virtual bool can_fuse_with(const Layer& next) const { return false; }
    virtual std::unique_ptr<Layer> fuse_with(const Layer& next) const { return nullptr; }
    
    // Getters/Setters
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }
    
    bool is_trainable() const { return trainable_; }
    void set_trainable(bool trainable) { trainable_ = trainable; }
    
protected:
    std::string name_;
    bool trainable_;
    std::unordered_map<std::string, Tensor> params_;
};

// Layer factory
class LayerFactory {
public:
    using Creator = std::function<std::unique_ptr<Layer>()>;
    
    static LayerFactory& instance() {
        static LayerFactory factory;
        return factory;
    }
    
    void register_layer(const std::string& type, Creator creator) {
        creators_[type] = creator;
    }
    
    std::unique_ptr<Layer> create(const std::string& type) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second();
        }
        throw std::runtime_error("Unknown layer type: " + type);
    }
    
private:
    std::unordered_map<std::string, Creator> creators_;
};

// Registration helper
template<typename T>
class LayerRegistrar {
public:
    explicit LayerRegistrar(const std::string& type) {
        LayerFactory::instance().register_layer(type, []() {
            return std::make_unique<T>();
        });
    }
};

#define REGISTER_LAYER(type, class_name) \
    static LayerRegistrar<class_name> registrar_##class_name(type);

// Execution context for layer execution
class ExecutionContext {
public:
    ExecutionContext() : stream_(nullptr), workspace_(nullptr), workspace_size_(0) {
        cudaStreamCreate(&stream_);
    }
    
    ~ExecutionContext() {
        if (stream_) cudaStreamDestroy(stream_);
        if (workspace_) cudaFree(workspace_);
    }
    
    cudaStream_t stream() const { return stream_; }
    
    void* workspace(size_t size) {
        if (size > workspace_size_) {
            if (workspace_) cudaFree(workspace_);
            cudaMalloc(&workspace_, size);
            workspace_size_ = size;
        }
        return workspace_;
    }
    
    void synchronize() {
        cudaStreamSynchronize(stream_);
    }
    
    // Profiling support
    void enable_profiling(bool enable) { profiling_enabled_ = enable; }
    bool is_profiling_enabled() const { return profiling_enabled_; }
    
    void record_time(const std::string& layer_name, float time_ms) {
        if (profiling_enabled_) {
            timings_[layer_name] = time_ms;
        }
    }
    
    const std::unordered_map<std::string, float>& get_timings() const {
        return timings_;
    }
    
private:
    cudaStream_t stream_;
    void* workspace_;
    size_t workspace_size_;
    bool profiling_enabled_ = false;
    std::unordered_map<std::string, float> timings_;
};

} // namespace deep_engine