#pragma once

#include "backend_base.h"
#include <memory>

class OrtBackend : public InferenceBackend {
public:
    explicit OrtBackend(int device_id = 0);
    ~OrtBackend();
    
    void load_model(const std::string& model_path) override;
    void infer(const void* input, void* output, cudaStream_t stream = 0) override;
    void infer_batch(const void* input, void* output, 
                    int batch_size, cudaStream_t stream = 0) override;
    
    size_t get_input_size() const override;
    size_t get_output_size() const override;
    std::vector<int64_t> get_input_shape() const override;
    std::vector<int64_t> get_output_shape() const override;
    
    bool supports_fp16() const override { return true; }
    void enable_fp16(bool enable) override;
    
    // ORT spesifik ayarlar
    void enable_cuda_graphs(bool enable);
    void set_cudnn_conv_algo_search(const std::string& mode);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};