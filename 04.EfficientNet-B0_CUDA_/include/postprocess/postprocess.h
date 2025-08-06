#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct ClassificationResult {
    int class_id;
    float probability;
    std::string class_name;
};

class Postprocessor {
public:
    Postprocessor(const std::string& class_names_file = "");
    ~Postprocessor();
    
    // Softmax + top-k
    std::vector<ClassificationResult> process(
        const float* d_logits, int num_classes, int top_k = 5,
        cudaStream_t stream = 0);
    
    // FP16 varyant
    std::vector<ClassificationResult> process_fp16(
        const __half* d_logits, int num_classes, int top_k = 5,
        cudaStream_t stream = 0);
    
    // Batch işleme
    std::vector<std::vector<ClassificationResult>> process_batch(
        const float* d_logits, int batch_size, int num_classes, int top_k = 5,
        cudaStream_t stream = 0);
    
private:
    std::vector<std::string> class_names_;
    
    // Geçici bufferlar
    float* d_softmax_;
    int* d_indices_;
    float* d_topk_probs_;
    
    size_t softmax_size_;
    size_t indices_size_;
    
    void ensure_buffers(int batch_size, int num_classes, int top_k);
    void load_class_names(const std::string& file);
};

// CUDA kernel fonksiyonları (softmax_topk.cu'da)
void softmax_kernel(const float* d_input, float* d_output, 
                   int batch_size, int num_classes, cudaStream_t stream);

void topk_kernel(const float* d_probs, int* d_indices, float* d_topk_probs,
                int batch_size, int num_classes, int k, cudaStream_t stream);