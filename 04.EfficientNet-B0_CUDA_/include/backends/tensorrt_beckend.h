#pragma once

#include "backend_base.h"
#include <memory>

class TensorRTBackend : public InferenceBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend();
    
    void load_model(const std::string& model_path) override;
    void infer(const void* input, void* output, cudaStream_t stream = 0) override;
    
    size_t get_input_size() const override;
    size_t get_output_size() const override;
    std::vector<int64_t> get_input_shape() const override;
    std::vector<int64_t> get_output_shape() const override;
    
    bool supports_fp16() const override { return true; }
    void enable_fp16(bool enable) override;
    
    // TensorRT spesifik
    void enable_int8(bool enable, const std::string& calib_data = "");
    void set_workspace_size(size_t size_mb);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    void build_engine_from_onnx(const std::string& onnx_path);
    void save_engine(const std::string& path);
};