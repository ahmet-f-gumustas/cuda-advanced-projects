#include "../include/kv_cache.h"
#include <cstring>

// ============================================================================
// KVCache implementation
// ============================================================================

KVCache KVCache::create(int n_layers, int n_kv_heads, int max_seq_len, int head_dim)
{
    KVCache kv;
    kv.n_layers    = n_layers;
    kv.n_kv_heads  = n_kv_heads;
    kv.max_seq_len = max_seq_len;
    kv.head_dim    = head_dim;
    kv.current_pos = 0;

    size_t per_buf = (size_t)n_layers * n_kv_heads * max_seq_len * head_dim;

    CUDA_CHECK(cudaMalloc(&kv.d_k, per_buf * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&kv.d_v, per_buf * sizeof(half)));
    CUDA_CHECK(cudaMemset(kv.d_k, 0, per_buf * sizeof(half)));
    CUDA_CHECK(cudaMemset(kv.d_v, 0, per_buf * sizeof(half)));

    return kv;
}

void KVCache::destroy()
{
    if (d_k) { CUDA_CHECK(cudaFree(d_k)); d_k = nullptr; }
    if (d_v) { CUDA_CHECK(cudaFree(d_v)); d_v = nullptr; }
    current_pos = 0;
}

void KVCache::reset()
{
    current_pos = 0;
    // Optionally zero the cache to avoid stale data (cheap at init time)
    size_t per_buf = (size_t)n_layers * n_kv_heads * max_seq_len * head_dim;
    CUDA_CHECK(cudaMemset(d_k, 0, per_buf * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_v, 0, per_buf * sizeof(half)));
}
