#include "../include/transformer_kernels.cuh"
#include <cuda_fp16.h>
#include <float.h>

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float half2float(half x) { return __half2float(x); }
__device__ __forceinline__ half  float2half(float x) { return __float2half(x); }

// Warp-level reduction (sum)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = PT_WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Warp-level reduction (max)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = PT_WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ============================================================================
// RMSNorm kernel
// ============================================================================
// Each block handles one vector of length d.
// Uses warp shuffles for reduction; d must be <= 1024.

__global__ void rmsnorm_kernel(half*       d_out,
                               const half* d_in,
                               const half* d_weight,
                               int         d,
                               float       eps)
{
    // blockIdx.x is always 0 (single vector per call)
    int tid = threadIdx.x;

    // Step 1: compute mean of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float v = half2float(d_in[i]);
        sum_sq += v * v;
    }

    // Reduce within warp
    sum_sq = warp_reduce_sum(sum_sq);

    // Cross-warp reduce via shared memory
    extern __shared__ float smem[];
    int lane   = tid % PT_WARP_SIZE;
    int warp_id = tid / PT_WARP_SIZE;
    int n_warps = (blockDim.x + PT_WARP_SIZE - 1) / PT_WARP_SIZE;

    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (tid < n_warps) ? smem[tid] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        if (tid == 0) smem[0] = sum_sq;
    }
    __syncthreads();

    float rms = rsqrtf(smem[0] / (float)d + eps);

    // Step 2: normalize and scale
    for (int i = tid; i < d; i += blockDim.x) {
        float v = half2float(d_in[i]) * rms * half2float(d_weight[i]);
        d_out[i] = float2half(v);
    }
}

// ============================================================================
// Add residual
// ============================================================================

__global__ void add_residual_kernel(half*       d_inout,
                                    const half* d_src,
                                    int         d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        d_inout[i] = float2half(half2float(d_inout[i]) + half2float(d_src[i]));
    }
}

// ============================================================================
// RoPE kernel
// ============================================================================
// x: [n_heads, head_dim]
// Each thread handles one (cos,sin) pair at offset 2*threadIdx.x within a head.

__global__ void rope_kernel(half* d_x,
                             int   pos,
                             int   n_heads,
                             int   head_dim,
                             float theta)
{
    int head = blockIdx.x;
    int i    = threadIdx.x;   // 0 .. head_dim/2 - 1

    if (i >= head_dim / 2) return;

    float freq     = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float angle    = (float)pos * freq;
    float cos_val  = cosf(angle);
    float sin_val  = sinf(angle);

    int base = head * head_dim + 2 * i;
    float x0 = half2float(d_x[base]);
    float x1 = half2float(d_x[base + 1]);

    d_x[base]     = float2half(x0 * cos_val - x1 * sin_val);
    d_x[base + 1] = float2half(x0 * sin_val + x1 * cos_val);
}

// ============================================================================
// Softmax kernel (over seq_len, per head)
// ============================================================================
// scores: [n_heads, seq_len]   (float, pre-softmax → post-softmax in-place)

__global__ void softmax_kernel(float* d_scores,
                                int    n_heads,
                                int    seq_len)
{
    int head = blockIdx.x;
    int tid  = threadIdx.x;

    float* row = d_scores + head * seq_len;

    // Step 1: max for numerical stability
    float max_val = -FLT_MAX;
    for (int t = tid; t < seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, row[t]);
    max_val = warp_reduce_max(max_val);

    extern __shared__ float smax[];
    int lane   = tid % PT_WARP_SIZE;
    int warpid = tid / PT_WARP_SIZE;
    int nwarps = (blockDim.x + PT_WARP_SIZE - 1) / PT_WARP_SIZE;

    if (lane == 0) smax[warpid] = max_val;
    __syncthreads();
    if (warpid == 0) {
        max_val = (tid < nwarps) ? smax[tid] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
        if (tid == 0) smax[0] = max_val;
    }
    __syncthreads();
    max_val = smax[0];

    // Step 2: exp and sum
    float sum_exp = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float e = expf(row[t] - max_val);
        row[t]  = e;
        sum_exp += e;
    }
    sum_exp = warp_reduce_sum(sum_exp);

    if (lane == 0) smax[warpid] = sum_exp;
    __syncthreads();
    if (warpid == 0) {
        sum_exp = (tid < nwarps) ? smax[tid] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
        if (tid == 0) smax[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = smax[0];

    // Step 3: normalize
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int t = tid; t < seq_len; t += blockDim.x)
        row[t] *= inv_sum;
}

// ============================================================================
// SwiGLU kernel
// ============================================================================
// hidden[i] = SiLU(gate[i]) * up[i]   where SiLU(x) = x * sigmoid(x)

__global__ void swiglu_kernel(half*       d_hidden,
                               const half* d_gate,
                               const half* d_up,
                               int         ff_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff_dim) return;

    float g = half2float(d_gate[i]);
    float u = half2float(d_up[i]);
    float silu = g / (1.0f + expf(-g));  // SiLU(g)
    d_hidden[i] = float2half(silu * u);
}

// ============================================================================
// Embedding lookup
// ============================================================================

__global__ void embedding_lookup_kernel(half*       d_out,
                                        const half* d_table,
                                        int         token_id,
                                        int         d_model)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_model) {
        d_out[i] = d_table[(long long)token_id * d_model + i];
    }
}

// ============================================================================
// Write KV cache
// ============================================================================
// k, v: [n_kv_heads, head_dim]  (computed for current token)
// k_cache, v_cache: layer's slice [n_kv_heads, max_seq_len, head_dim]

__global__ void write_kv_cache_kernel(half*       d_k_cache,
                                       half*       d_v_cache,
                                       const half* d_k,
                                       const half* d_v,
                                       int         pos,
                                       int         n_kv_heads,
                                       int         max_seq_len,
                                       int         head_dim)
{
    int kv_h = blockIdx.x;   // KV head index
    int i    = threadIdx.x;  // dimension index

    if (kv_h >= n_kv_heads || i >= head_dim) return;

    long long cache_offset = (long long)kv_h * max_seq_len * head_dim
                           + (long long)pos  * head_dim + i;
    long long src_offset   = (long long)kv_h * head_dim + i;

    d_k_cache[cache_offset] = d_k[src_offset];
    d_v_cache[cache_offset] = d_v[src_offset];
}

// ============================================================================
// Attention scores kernel
// ============================================================================
// scores[h, t] = dot(Q[h], K_cache[kv_h][t]) * scale
// GQA: kv_h = h / kv_groups
// Q:       [n_heads,    head_dim]
// K_cache: [n_kv_heads, max_seq_len, head_dim]  (layer slice)
// scores:  [n_heads,    seq_len]  float

__global__ void attention_scores_kernel(float*      d_scores,
                                         const half* d_q,
                                         const half* d_k_cache,
                                         int         n_heads,
                                         int         n_kv_heads,
                                         int         head_dim,
                                         int         seq_len,
                                         int         max_seq_len,
                                         float       scale)
{
    int h = blockIdx.x;   // query head
    int t = blockIdx.y * blockDim.x + threadIdx.x;  // token position

    if (h >= n_heads || t >= seq_len) return;

    int kv_h      = h / (n_heads / n_kv_heads);  // GQA mapping
    const half* q = d_q + (long long)h * head_dim;
    const half* k = d_k_cache + (long long)kv_h * max_seq_len * head_dim
                               + (long long)t * head_dim;

    float dot = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < head_dim; ++i) {
        dot += half2float(q[i]) * half2float(k[i]);
    }

    d_scores[(long long)h * seq_len + t] = dot * scale;
}

// ============================================================================
// Attention output kernel
// ============================================================================
// out[h, i] = sum_t probs[h, t] * V_cache[kv_h][t][i]
// probs:   [n_heads, seq_len]  float (post-softmax)
// V_cache: [n_kv_heads, max_seq_len, head_dim]  (layer slice)
// out:     [n_heads, head_dim]

__global__ void attention_output_kernel(half*        d_out,
                                         const float* d_probs,
                                         const half*  d_v_cache,
                                         int          n_heads,
                                         int          n_kv_heads,
                                         int          seq_len,
                                         int          max_seq_len,
                                         int          head_dim)
{
    int h = blockIdx.x;   // query head
    int i = threadIdx.x;  // dimension index

    if (h >= n_heads || i >= head_dim) return;

    int kv_h = h / (n_heads / n_kv_heads);

    const float* probs = d_probs + (long long)h * seq_len;
    const half*  V     = d_v_cache + (long long)kv_h * max_seq_len * head_dim;

    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        acc += probs[t] * half2float(V[(long long)t * head_dim + i]);
    }

    d_out[(long long)h * head_dim + i] = float2half(acc);
}

// ============================================================================
// FP16 logits → FP32
// ============================================================================

__global__ void logits_fp16_to_fp32_kernel(const half* d_src, float* d_dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_dst[i] = half2float(d_src[i]);
}

// ============================================================================
// FP32 → FP16
// ============================================================================

__global__ void fp32_to_fp16_kernel(const float* d_src, half* d_dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_dst[i] = float2half(d_src[i]);
}

// ============================================================================
// INT8 quantization
// ============================================================================

__global__ void quantize_fp16_to_int8_kernel(const half* d_src,
                                              int8_t*     d_dst,
                                              float*      d_scales,
                                              int         rows,
                                              int         cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* src_row = d_src + (long long)row * cols;

    // Step 1: find row max (reduction within block)
    float row_max = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        row_max = fmaxf(row_max, fabsf(half2float(src_row[c])));

    row_max = warp_reduce_max(row_max);

    extern __shared__ float smem_q[];
    int lane   = threadIdx.x % PT_WARP_SIZE;
    int warpid = threadIdx.x / PT_WARP_SIZE;
    int nwarps = (blockDim.x + PT_WARP_SIZE - 1) / PT_WARP_SIZE;

    if (lane == 0) smem_q[warpid] = row_max;
    __syncthreads();
    if (warpid == 0) {
        row_max = (threadIdx.x < nwarps) ? smem_q[threadIdx.x] : 0.0f;
        row_max = warp_reduce_max(row_max);
        if (threadIdx.x == 0) smem_q[0] = row_max;
    }
    __syncthreads();
    row_max = smem_q[0];

    float scale = (row_max < 1e-10f) ? 1.0f : row_max / 127.0f;
    if (threadIdx.x == 0) d_scales[row] = scale;

    // Step 2: quantize
    int8_t* dst_row = d_dst + (long long)row * cols;
    float inv_scale = 1.0f / scale;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = half2float(src_row[c]) * inv_scale;
        v = fminf(127.0f, fmaxf(-127.0f, v));
        dst_row[c] = static_cast<int8_t>(__float2int_rn(v));
    }
}

__global__ void dequantize_int8_to_fp16_kernel(const int8_t* d_src,
                                                half*         d_dst,
                                                const float*  d_scales,
                                                int           rows,
                                                int           cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    float scale = d_scales[row];
    const int8_t* src_row = d_src + (long long)row * cols;
    half*         dst_row = d_dst + (long long)row * cols;

    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        dst_row[c] = float2half((float)src_row[c] * scale);
}

// ============================================================================
// Sampling kernels
// ============================================================================

// Temperature scaling
__global__ void scale_logits_kernel(float* d_logits, float inv_temp, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_logits[i] *= inv_temp;
}

// Greedy argmax — single block, shared memory reduction
// Assumes vocab_size <= 65536 and a single block with 512 threads
__global__ void argmax_kernel(const float* d_logits,
                               int*         d_out_token,
                               float*       d_max_val,
                               int          vocab_size)
{
    extern __shared__ float sdata[];
    int* sindex = (int*)(sdata + blockDim.x);

    int tid = threadIdx.x;
    float my_max = -FLT_MAX;
    int   my_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (d_logits[i] > my_max) {
            my_max = d_logits[i];
            my_idx = i;
        }
    }

    sdata[tid]  = my_max;
    sindex[tid] = my_idx;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid]  = sdata[tid + s];
            sindex[tid] = sindex[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_out_token = sindex[0];
        if (d_max_val) *d_max_val = sdata[0];
    }
}

// Top-K sampling
// 1. Partial sort to find top-K values (via selection)
// 2. Softmax over top-K
// 3. Sample using curand
__global__ void top_k_sampling_kernel(float*       d_logits,
                                       int*         d_out_token,
                                       int          vocab_size,
                                       int          k,
                                       float        temperature,
                                       curandState* d_rng_state)
{
    // Run single-threaded (k is small, e.g. 40)
    if (threadIdx.x != 0) return;

    curandState local_state = *d_rng_state;

    // Step 1: find k-th largest via partial selection (O(vocab*k), k≤40 is fine)
    // We store the top-k indices and values
    float top_vals[64];
    int   top_idxs[64];
    if (k > 64) k = 64;

    // Initialize with first k elements
    for (int i = 0; i < k && i < vocab_size; ++i) {
        top_vals[i] = d_logits[i];
        top_idxs[i] = i;
    }
    // Simple insertion into top-k for remaining elements
    float kth_val = -FLT_MAX;
    // Find minimum in initial top-k
    int min_pos = 0;
    for (int i = 0; i < k && i < vocab_size; ++i)
        if (top_vals[i] < top_vals[min_pos]) min_pos = i;
    kth_val = top_vals[min_pos];

    for (int i = k; i < vocab_size; ++i) {
        if (d_logits[i] > kth_val) {
            top_vals[min_pos] = d_logits[i];
            top_idxs[min_pos] = i;
            // Recompute minimum
            min_pos = 0;
            for (int j = 1; j < k; ++j)
                if (top_vals[j] < top_vals[min_pos]) min_pos = j;
            kth_val = top_vals[min_pos];
        }
    }

    // Step 2: softmax over top-k with temperature
    float max_v = -FLT_MAX;
    for (int i = 0; i < k; ++i) max_v = fmaxf(max_v, top_vals[i] / temperature);

    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        top_vals[i] = expf(top_vals[i] / temperature - max_v);
        sum_exp += top_vals[i];
    }
    for (int i = 0; i < k; ++i)
        top_vals[i] /= sum_exp;

    // Step 3: sample
    float r = curand_uniform(&local_state);
    float cum = 0.0f;
    int sampled = top_idxs[0];
    for (int i = 0; i < k; ++i) {
        cum += top_vals[i];
        if (r <= cum) { sampled = top_idxs[i]; break; }
    }

    *d_out_token  = sampled;
    *d_rng_state  = local_state;
}

// ============================================================================
// curand state initialization
// ============================================================================

__global__ void init_curand_state_kernel(curandState* d_state,
                                          unsigned long long seed,
                                          int                n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &d_state[i]);
}
