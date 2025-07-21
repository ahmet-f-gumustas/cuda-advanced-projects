#pragma once

#include "../core/tensor.h"
#include "../core/layer.h"
#include <memory>
#include <unordered_map>

namespace deep_engine {

// Quantization schemes
enum class QuantizationScheme {
    SYMMETRIC,      // Zero point = 0
    ASYMMETRIC,     // Arbitrary zero point
    DYNAMIC,        // Compute scale per batch
    STATIC          // Pre-computed scale
};

// Quantization configuration
struct QuantizationConfig {
    int bits = 8;
    QuantizationScheme scheme = QuantizationScheme::SYMMETRIC;
    bool per_channel = false;
    float percentile = 99.9f;  // For calibration
    int calibration_batches = 100;
    bool fake_quantize = false;  // For QAT
};

// Quantization parameters
struct QuantizationParams {
    float scale;
    int zero_point;
    float min_val;
    float max_val;
};

// Base quantizer class
class Quantizer {
public:
    virtual ~Quantizer() = default;
    
    virtual Tensor quantize(const Tensor& tensor) const = 0;
    virtual Tensor dequantize(const Tensor& tensor) const = 0;
    virtual QuantizationParams compute_params(const Tensor& tensor) const = 0;
};

// INT8 Quantizer
class Int8Quantizer : public Quantizer {
public:
    explicit Int8Quantizer(QuantizationScheme scheme = QuantizationScheme::SYMMETRIC)
        : scheme_(scheme) {}
    
    Tensor quantize(const Tensor& tensor) const override;
    Tensor dequantize(const Tensor& tensor) const override;
    QuantizationParams compute_params(const Tensor& tensor) const override;
    
private:
    QuantizationScheme scheme_;
};

// Dynamic quantizer
class DynamicQuantizer : public Quantizer {
public:
    explicit DynamicQuantizer(int bits = 8) : bits_(bits) {}
    
    Tensor quantize(const Tensor& tensor) const override;
    Tensor dequantize(const Tensor& tensor) const override;
    QuantizationParams compute_params(const Tensor& tensor) const override;
    
private:
    int bits_;
};

// Quantization-aware training support
class FakeQuantizer : public Layer {
public:
    FakeQuantizer(int bits = 8, float scale = 1.0f, int zero_point = 0,
                  const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "FakeQuantizer"; }
    
    void update_params(const QuantizationParams& params);
    
private:
    int bits_;
    float scale_;
    int zero_point_;
    float min_val_;
    float max_val_;
};

// Calibration for static quantization
class QuantizationCalibrator {
public:
    explicit QuantizationCalibrator(const QuantizationConfig& config)
        : config_(config) {}
    
    void collect_statistics(const std::string& tensor_name, const Tensor& tensor);
    QuantizationParams compute_params(const std::string& tensor_name);
    std::unordered_map<std::string, QuantizationParams> get_all_params();
    
private:
    QuantizationConfig config_;
    std::unordered_map<std::string, std::vector<float>> min_values_;
    std::unordered_map<std::string, std::vector<float>> max_values_;
    std::unordered_map<std::string, std::vector<float>> histograms_;
};

// Quantized layers
class QuantizedConv2d : public Layer {
public:
    QuantizedConv2d(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 0,
                    const QuantizationParams& input_params = {},
                    const QuantizationParams& weight_params = {},
                    const QuantizationParams& output_params = {},
                    const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "QuantizedConv2d"; }
    
    size_t num_params() const override;
    
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    QuantizationParams input_params_;
    QuantizationParams weight_params_;
    QuantizationParams output_params_;
};

class QuantizedLinear : public Layer {
public:
    QuantizedLinear(int in_features, int out_features,
                    const QuantizationParams& input_params = {},
                    const QuantizationParams& weight_params = {},
                    const QuantizationParams& output_params = {},
                    bool use_bias = true,
                    const std::string& name = "");
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { return "QuantizedLinear"; }
    
    size_t num_params() const override;
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    QuantizationParams input_params_;
    QuantizationParams weight_params_;
    QuantizationParams output_params_;
};

// Mixed precision support
class MixedPrecisionWrapper : public Layer {
public:
    MixedPrecisionWrapper(std::unique_ptr<Layer> layer,
                         DataType compute_dtype = DataType::FP16,
                         DataType storage_dtype = DataType::FP32)
        : layer_(std::move(layer)), 
          compute_dtype_(compute_dtype),
          storage_dtype_(storage_dtype) {}
    
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override;
    std::string type() const override { 
        return "MixedPrecision[" + layer_->type() + "]"; 
    }
    
    size_t num_params() const override { return layer_->num_params(); }
    
private:
    std::unique_ptr<Layer> layer_;
    DataType compute_dtype_;
    DataType storage_dtype_;
};

} // namespace deep_engine