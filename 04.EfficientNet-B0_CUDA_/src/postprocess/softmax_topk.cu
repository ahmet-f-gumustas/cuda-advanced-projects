#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <limits>
#include <float.h>

// Softmax kernel - her satır için softmax hesapla
__global__ void softmax_kernel_impl(const float* input, float* output,
                                   int batch_size, int num_classes) {
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* in_row = input + batch_idx * num_classes;
    float* out_row = output + batch_idx * num_classes;
    
    // Max değeri bul (numerik stabilite için)
    float max_val = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    
    // Block içinde max reduction
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    max_val = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    
    if (tid == 0) {
        shared_data[0] = max_val;
    }
    __syncthreads();
    max_val = shared_data[0];
    
    // Exp hesapla ve toplamı bul
    float sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(in_row[i] - max_val);
        if (i < num_classes) {
            out_row[i] = exp_val;
        }
        sum += exp_val;
    }
    
    // Sum reduction
    __syncthreads();
    sum = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
        shared_data[0] = sum;
    }
    __syncthreads();
    sum = shared_data[0];
    
    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        out_row[i] /= sum;
    }
}

// Top-k selection kernel (basit versiyon)
__global__ void topk_kernel_impl(const float* probs, int* indices, float* top_probs,
                                int batch_size, int num_classes, int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* prob_row = probs + batch_idx * num_classes;
    int* idx_row = indices + batch_idx * k;
    float* topk_row = top_probs + batch_idx * k;
    
    // Basit selection sort (k küçük olduğu için yeterli)
    for (int i = 0; i < k; ++i) {
        float max_prob = -1.0f;
        int max_idx = -1;
        
        // En büyük elemanı bul
        for (int j = 0; j < num_classes; ++j) {
            bool already_selected = false;
            for (int m = 0; m < i; ++m) {
                if (idx_row[m] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && prob_row[j] > max_prob) {
                max_prob = prob_row[j];
                max_idx = j;
            }
        }
        
        idx_row[i] = max_idx;
        topk_row[i] = max_prob;
    }
}

// Host wrapper fonksiyonları
void softmax_kernel(const float* d_input, float* d_output,
                   int batch_size, int num_classes, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = batch_size;
    size_t shared_mem = sizeof(float) * 32;  // Max ve sum için
    
    softmax_kernel_impl<<<blocks, threads, shared_mem, stream>>>(
        d_input, d_output, batch_size, num_classes);
}

void topk_kernel(const float* d_probs, int* d_indices, float* d_topk_probs,
                int batch_size, int num_classes, int k, cudaStream_t stream) {
    const int blocks = batch_size;
    
    topk_kernel_impl<<<blocks, 1, 0, stream>>>(
        d_probs, d_indices, d_topk_probs, batch_size, num_classes, k);
}