#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include "../core/graph.h"
#include "../core/tensor.h"

namespace deep_engine {

// Model format types
enum class ModelFormat {
    ONNX,
    TENSORFLOW,
    PYTORCH,
    KERAS,
    CUSTOM
};

// Base model loader interface
class ModelLoader {
public:
    virtual ~ModelLoader() = default;
    
    virtual std::unique_ptr<ComputationGraph> load(const std::string& path) = 0;
    virtual void save(const ComputationGraph& graph, const std::string& path) = 0;
    virtual ModelFormat format() const = 0;
};

// ONNX model loader
class ONNXModelLoader : public ModelLoader {
public:
    std::unique_ptr<ComputationGraph> load(const std::string& path) override;
    void save(const ComputationGraph& graph, const std::string& path) override;
    ModelFormat format() const override { return ModelFormat::ONNX; }
    
private:
    std::shared_ptr<Layer> convert_onnx_node(const void* node);
    Tensor load_onnx_tensor(const void* tensor);
};

// TensorFlow model loader
class TensorFlowModelLoader : public ModelLoader {
public:
    std::unique_ptr<ComputationGraph> load(const std::string& path) override;
    void save(const ComputationGraph& graph, const std::string& path) override;
    ModelFormat format() const override { return ModelFormat::TENSORFLOW; }
    
    // Load from frozen graph (.pb)
    std::unique_ptr<ComputationGraph> load_frozen_graph(const std::string& path);
    
    // Load from saved model directory
    std::unique_ptr<ComputationGraph> load_saved_model(const std::string& path);
};

// PyTorch model loader
class PyTorchModelLoader : public ModelLoader {
public:
    std::unique_ptr<ComputationGraph> load(const std::string& path) override;
    void save(const ComputationGraph& graph, const std::string& path) override;
    ModelFormat format() const override { return ModelFormat::PYTORCH; }
    
    // Load TorchScript model
    std::unique_ptr<ComputationGraph> load_torchscript(const std::string& path);
    
    // Load state dict
    void load_state_dict(ComputationGraph& graph, const std::string& path);
};

// Custom binary format loader
class CustomModelLoader : public ModelLoader {
public:
    std::unique_ptr<ComputationGraph> load(const std::string& path) override;
    void save(const ComputationGraph& graph, const std::string& path) override;
    ModelFormat format() const override { return ModelFormat::CUSTOM; }
    
    // Version control
    void set_version(int version) { version_ = version; }
    int get_version() const { return version_; }
    
private:
    int version_ = 1;
    
    void write_header(std::ostream& os, const ComputationGraph& graph);
    void read_header(std::istream& is, int& version, int& num_nodes);
    
    void write_tensor(std::ostream& os, const Tensor& tensor);
    Tensor read_tensor(std::istream& is);
    
    void write_layer(std::ostream& os, const Layer& layer);
    std::shared_ptr<Layer> read_layer(std::istream& is);
};

// Model loader factory
class ModelLoaderFactory {
public:
    static std::unique_ptr<ModelLoader> create(ModelFormat format) {
        switch (format) {
            case ModelFormat::ONNX:
                return std::make_unique<ONNXModelLoader>();
            case ModelFormat::TENSORFLOW:
                return std::make_unique<TensorFlowModelLoader>();
            case ModelFormat::PYTORCH:
                return std::make_unique<PyTorchModelLoader>();
            case ModelFormat::CUSTOM:
                return std::make_unique<CustomModelLoader>();
            default:
                throw std::runtime_error("Unsupported model format");
        }
    }
    
    static std::unique_ptr<ModelLoader> create_from_file(const std::string& path);
};

// Model converter between formats
class ModelConverter {
public:
    static void convert(const std::string& input_path, ModelFormat input_format,
                       const std::string& output_path, ModelFormat output_format);
    
    static void convert(const ComputationGraph& graph,
                       const std::string& output_path, ModelFormat output_format);
};

// Model optimization during loading
class OptimizingModelLoader {
public:
    OptimizingModelLoader(std::unique_ptr<ModelLoader> base_loader)
        : base_loader_(std::move(base_loader)) {}
    
    std::unique_ptr<ComputationGraph> load(const std::string& path);
    
    // Optimization options
    void enable_quantization(bool enable = true) { quantize_ = enable; }
    void enable_fusion(bool enable = true) { fuse_ = enable; }
    void enable_pruning(bool enable = true) { prune_ = enable; }
    void set_precision(PrecisionMode mode) { precision_ = mode; }
    
private:
    std::unique_ptr<ModelLoader> base_loader_;
    bool quantize_ = false;
    bool fuse_ = true;
    bool prune_ = false;
    PrecisionMode precision_ = PrecisionMode::FP32;
};

// Model metadata
struct ModelMetadata {
    std::string name;
    std::string version;
    std::string description;
    std::vector<std::string> input_names;
    std::vector<std::vector<int>> input_shapes;
    std::vector<DataType> input_types;
    std::vector<std::string> output_names;
    std::vector<std::vector<int>> output_shapes;
    std::vector<DataType> output_types;
    std::unordered_map<std::string, std::string> custom_metadata;
};

// Model validation
class ModelValidator {
public:
    static bool validate(const ComputationGraph& graph);
    static bool validate_onnx(const std::string& path);
    static bool validate_tensorflow(const std::string& path);
    
    struct ValidationResult {
        bool valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    static ValidationResult validate_detailed(const ComputationGraph& graph);
};

// Weight initialization strategies
enum class InitStrategy {
    XAVIER_UNIFORM,
    XAVIER_NORMAL,
    KAIMING_UNIFORM,
    KAIMING_NORMAL,
    NORMAL,
    UNIFORM,
    ZEROS,
    ONES
};

class WeightInitializer {
public:
    static void initialize(Tensor& weight, InitStrategy strategy,
                          int fan_in = -1, int fan_out = -1);
    
    static void initialize_conv(Tensor& weight, InitStrategy strategy);
    static void initialize_linear(Tensor& weight, InitStrategy strategy);
    static void initialize_embedding(Tensor& weight, InitStrategy strategy);
};

} // namespace deep_engine