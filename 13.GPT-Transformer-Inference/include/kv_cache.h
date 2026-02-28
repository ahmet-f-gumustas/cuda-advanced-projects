#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.h"

// ============================================================================
// KVCache - GPU memory for key/value tensors across all layers
// ============================================================================
// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
// Stored as two flat contiguous buffers (K and V separately).
//
// Accessing position pos in layer l, head kv_h:
//   k_ptr(l) + kv_h * max_seq_len * head_dim + pos * head_dim
// ============================================================================

struct KVCache {
    half* d_k;         // [n_layers, n_kv_heads, max_seq_len, head_dim]
    half* d_v;         // [n_layers, n_kv_heads, max_seq_len, head_dim]

    int current_pos;   // Number of tokens processed so far (= next write position)
    int n_layers;
    int n_kv_heads;
    int max_seq_len;
    int head_dim;

    // Create and allocate GPU memory
    // Returns an initialized (zeroed) KVCache
    static KVCache create(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);

    // Free GPU memory
    void destroy();

    // Reset for a new generation (does NOT free memory — just resets position)
    void reset();

    // Pointer to the start of layer l's K buffer:
    //   size per layer = n_kv_heads * max_seq_len * head_dim  (half elements)
    half* k_ptr(int layer) const {
        return d_k + (long long)layer * n_kv_heads * max_seq_len * head_dim;
    }

    half* v_ptr(int layer) const {
        return d_v + (long long)layer * n_kv_heads * max_seq_len * head_dim;
    }

    // Total bytes used by both K and V buffers
    size_t bytes() const {
        size_t per_buf = (size_t)n_layers * n_kv_heads * max_seq_len * head_dim * sizeof(half);
        return 2 * per_buf;
    }
};

#endif // KV_CACHE_H
