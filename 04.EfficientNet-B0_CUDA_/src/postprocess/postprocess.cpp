#include "postprocess/postprocess.h"
#include "common/cuda_utils.h"
#include "common/logger.h"
#include <fstream>
#include <algorithm>

Postprocessor::Postprocessor(const std::string& class_names_file)
    : d_softmax_(nullptr), d_indices_(nullptr), d_topk_probs_(nullptr),
      softmax_size_(0), indices_size_(0) {
    
    if (!class_names_file.empty()) {
        load_class_names(class_names_file);
    }

std::vector<ClassificationResult> Postprocessor::process(
    const float* d_logits, int num_classes, int top_k, cudaStream_t stream) {
    
    ensure_buffers(1, num_classes, top_k);
    
    // Softmax
    softmax_kernel(d_logits, d_softmax_, 1, num_classes, stream);
    
    // Top-k
    topk_kernel(d_softmax_, d_indices_, d_topk_probs_, 1, num_classes, top_k, stream);
    
    // Sonuçları host'a kopyala
    std::vector<int> indices(top_k);
    std::vector<float> probs(top_k);
    
    CUDA_CHECK(cudaMemcpyAsync(indices.data(), d_indices_, 
                               top_k * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(probs.data(), d_topk_probs_,
                               top_k * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Sonuçları oluştur
    std::vector<ClassificationResult> results;
    for (int i = 0; i < top_k; ++i) {
        ClassificationResult result;
        result.class_id = indices[i];
        result.probability = probs[i];
        
        if (result.class_id < class_names_.size()) {
            result.class_name = class_names_[result.class_id];
        } else {
            result.class_name = "Class " + std::to_string(result.class_id);
        }
        
        results.push_back(result);
    }
    
    return results;
}

std::vector<ClassificationResult> Postprocessor::process_fp16(
    const __half* d_logits, int num_classes, int top_k, cudaStream_t stream) {
    
    // FP16'dan FP32'ye dönüştür
    ensure_buffers(1, num_classes, top_k);
    
    // Dönüşüm kernel'i çağır (implement edilmeli)
    // convert_fp16_to_fp32(d_logits, d_softmax_, num_classes, stream);
    
    // Sonra normal process'i çağır
    return process(d_softmax_, num_classes, top_k, stream);
}

std::vector<std::vector<ClassificationResult>> Postprocessor::process_batch(
    const float* d_logits, int batch_size, int num_classes, int top_k,
    cudaStream_t stream) {
    
    ensure_buffers(batch_size, num_classes, top_k);
    
    // Batch softmax ve top-k
    softmax_kernel(d_logits, d_softmax_, batch_size, num_classes, stream);
    topk_kernel(d_softmax_, d_indices_, d_topk_probs_, batch_size, num_classes, top_k, stream);
    
    // Sonuçları kopyala
    std::vector<int> all_indices(batch_size * top_k);
    std::vector<float> all_probs(batch_size * top_k);
    
    CUDA_CHECK(cudaMemcpyAsync(all_indices.data(), d_indices_,
                               batch_size * top_k * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(all_probs.data(), d_topk_probs_,
                               batch_size * top_k * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Batch sonuçlarını oluştur
    std::vector<std::vector<ClassificationResult>> batch_results;
    for (int b = 0; b < batch_size; ++b) {
        std::vector<ClassificationResult> results;
        for (int i = 0; i < top_k; ++i) {
            int idx = b * top_k + i;
            ClassificationResult result;
            result.class_id = all_indices[idx];
            result.probability = all_probs[idx];
            
            if (result.class_id < class_names_.size()) {
                result.class_name = class_names_[result.class_id];
            } else {
                result.class_name = "Class " + std::to_string(result.class_id);
            }
            
            results.push_back(result);
        }
        batch_results.push_back(results);
    }
    
    return batch_results;
}

void Postprocessor::ensure_buffers(int batch_size, int num_classes, int top_k) {
    size_t needed_softmax = batch_size * num_classes * sizeof(float);
    size_t needed_indices = batch_size * top_k * sizeof(int);
    
    if (needed_softmax > softmax_size_) {
        if (d_softmax_) cudaFree(d_softmax_);
        if (d_topk_probs_) cudaFree(d_topk_probs_);
        
        CUDA_CHECK(cudaMalloc(&d_softmax_, needed_softmax));
        CUDA_CHECK(cudaMalloc(&d_topk_probs_, batch_size * top_k * sizeof(float)));
        softmax_size_ = needed_softmax;
    }
    
    if (needed_indices > indices_size_) {
        if (d_indices_) cudaFree(d_indices_);
        
        CUDA_CHECK(cudaMalloc(&d_indices_, needed_indices));
        indices_size_ = needed_indices;
    }
}

void Postprocessor::load_class_names(const std::string& file) {
    std::ifstream in(file);
    if (!in.is_open()) {
        LOG_WARNING("Cannot open class names file:", file);
        return;
    }
    
    std::string line;
    while (std::getline(in, line)) {
        class_names_.push_back(line);
    }
    
    LOG_INFO("Loaded", class_names_.size(), "class names");
}
}

Postprocessor::~Postprocessor() {
    if (d_softmax_) cudaFree(d_softmax_);
    if (d_indices_) cudaFree(d_indices_);
    if (d_topk_probs_) cudaFree(d_topk_probs_);
}