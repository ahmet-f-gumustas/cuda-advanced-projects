#include "unet.h"
#include "diffusion_kernels.cuh"

// ============================================================
// Conv2D implementation (cuDNN)
// ============================================================

void Conv2D::init(int in_ch, int out_ch, int k, int s, int p, bool use_bias) {
    in_channels = in_ch;
    out_channels = out_ch;
    kernel_size = k;
    stride = s;
    padding = p;

    d_weight = cudaMallocDevice<half>(out_ch * in_ch * k * k);
    d_bias = use_bias ? cudaMallocDevice<half>(out_ch) : nullptr;

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc,
        CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
        out_ch, in_ch, k, k));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
        p, p, s, s, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
}

void Conv2D::initRandom(float scale, unsigned& seed) {
    init_random_fp16(d_weight, out_channels * in_channels * kernel_size * kernel_size, scale, seed++);
    if (d_bias) init_zeros_fp16(d_bias, out_channels);
}

size_t Conv2D::getWorkspaceSize(cudnnHandle_t handle, int H, int W) {
    cudnnTensorDescriptor_t in_desc, out_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
        1, in_channels, H, W));

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
        1, out_channels, H_out, W_out));

    size_t ws = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
        in_desc, filter_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &ws));

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    return ws;
}

void Conv2D::forward(cudnnHandle_t handle,
                     const half* d_input, half* d_output,
                     int H, int W,
                     void* d_workspace, size_t ws_size)
{
    cudnnTensorDescriptor_t in_desc, out_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
        1, in_channels, H, W));

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
        1, out_channels, H_out, W_out));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
        &alpha,
        in_desc, d_input,
        filter_desc, d_weight,
        conv_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        d_workspace, ws_size,
        &beta,
        out_desc, d_output));

    // Add bias if present
    if (d_bias) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            1, out_channels, 1, 1));
        float one = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle, &one, bias_desc, d_bias, &one, out_desc, d_output));
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}

void Conv2D::destroy() {
    cudaFree(d_weight);
    if (d_bias) cudaFree(d_bias);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// ============================================================
// GroupNormLayer implementation
// ============================================================

void GroupNormLayer::init(int C, int groups, float epsilon) {
    channels = C;
    num_groups = groups;
    eps = epsilon;
    d_gamma = cudaMallocDevice<half>(C);
    d_beta = cudaMallocDevice<half>(C);
}

void GroupNormLayer::initRandom(unsigned& seed) {
    (void)seed;
    // gamma = 1, beta = 0
    std::vector<half> ones(channels), zeros(channels);
    for (int i = 0; i < channels; i++) {
        ones[i] = __float2half(1.0f);
        zeros[i] = __float2half(0.0f);
    }
    cudaMemcpyH2D(d_gamma, ones.data(), channels);
    cudaMemcpyH2D(d_beta, zeros.data(), channels);
}

void GroupNormLayer::forward(const half* d_input, half* d_output, int H, int W) {
    int threads = std::min(channels / num_groups * H * W, 1024);
    threads = std::max(threads, 32);
    group_norm_kernel<<<num_groups, threads>>>(
        d_output, d_input, d_gamma, d_beta,
        channels, H, W, num_groups, eps);
    CUDA_CHECK_LAST_ERROR();
}

void GroupNormLayer::destroy() {
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// ============================================================
// ResBlock implementation
// ============================================================

void ResBlock::init(int in_ch, int out_ch, int temb_dim, int num_groups) {
    in_channels = in_ch;
    out_channels = out_ch;
    time_embed_dim = temb_dim;
    has_skip = (in_ch != out_ch);

    norm1.init(in_ch, num_groups);
    conv1.init(in_ch, out_ch, 3, 1, 1);

    norm2.init(out_ch, num_groups);
    conv2.init(out_ch, out_ch, 3, 1, 1);

    // Time embedding projection
    d_time_proj_w = cudaMallocDevice<half>(temb_dim * out_ch);
    d_time_proj_b = cudaMallocDevice<half>(out_ch);

    if (has_skip) {
        skip_conv.init(in_ch, out_ch, 1, 1, 0);
    }
}

void ResBlock::initRandom(float scale, unsigned& seed) {
    norm1.initRandom(seed);
    conv1.initRandom(scale, seed);
    norm2.initRandom(seed);
    conv2.initRandom(scale, seed);
    init_random_fp16(d_time_proj_w, time_embed_dim * out_channels, scale, seed++);
    init_zeros_fp16(d_time_proj_b, out_channels);
    if (has_skip) skip_conv.initRandom(scale, seed);
}

void ResBlock::forward(cudnnHandle_t cudnn, cublasHandle_t cublas,
                       const half* d_input, const half* d_temb,
                       half* d_output, int H, int W,
                       void* d_workspace, size_t ws_size,
                       half* d_temp1, half* d_temp2)
{
    int spatial = H * W;

    // Path 1: norm1 -> silu -> conv1 -> + time_emb -> norm2 -> silu -> conv2
    norm1.forward(d_input, d_temp1, H, W);

    int size1 = in_channels * spatial;
    silu_inplace_kernel<<<(size1 + 255) / 256, 256>>>(d_temp1, size1);
    CUDA_CHECK_LAST_ERROR();

    conv1.forward(cudnn, d_temp1, d_temp2, H, W, d_workspace, ws_size);

    // Add time embedding: project temb [1, temb_dim] -> [1, out_ch] then broadcast to spatial
    // temb_proj = temb @ W^T + b -> [1, out_ch]
    {
        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);
        // d_temp1 reuse as temb projection output [1, out_channels]
        CUBLAS_CHECK(cublasHgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            out_channels, 1, time_embed_dim,
            &alpha_h,
            d_time_proj_w, time_embed_dim,
            d_temb, time_embed_dim,
            &beta_h,
            d_temp1, out_channels));

        // Add bias
        int total = out_channels;
        add_tensors_inplace_kernel<<<(total + 255) / 256, 256>>>(d_temp1, d_time_proj_b, total);
        CUDA_CHECK_LAST_ERROR();

        // Broadcast add: d_temp2[c, h, w] += d_temp1[c] for all h, w
        // Simple kernel: each thread handles one element
        int out_size = out_channels * spatial;
        // We need a small broadcast kernel - use a lambda-style approach
        // For simplicity, iterate in a flat kernel
        // d_temp2 has layout [out_channels, H, W]
        // d_temp1 has [out_channels] values to add per channel
        // We'll add using a simple loop
        for (int c = 0; c < out_channels; c++) {
            half* channel_ptr = d_temp2 + c * spatial;
            half temb_val;
            CUDA_CHECK(cudaMemcpy(&temb_val, d_temp1 + c, sizeof(half), cudaMemcpyDeviceToHost));
            // Create a temp filled with temb_val and add
            // Actually, let's use scale+add: much simpler to just do on device
            // We'll launch a kernel per channel (small overhead, clean code)
            scale_tensor_kernel<<<(spatial + 255) / 256, 256>>>(
                d_temp1 + out_channels,  // reuse temp space
                channel_ptr, 1.0f, spatial);  // copy
        }
        // Better approach: use a single broadcast-add kernel inline
        // Actually simplest: memcpy temb to host, create full tensor, upload
        std::vector<half> h_temb(out_channels);
        cudaMemcpyD2H(h_temb.data(), d_temp1, out_channels);
        std::vector<half> h_broadcast(out_size);
        for (int c = 0; c < out_channels; c++) {
            for (int s = 0; s < spatial; s++) {
                h_broadcast[c * spatial + s] = h_temb[c];
            }
        }
        half* d_broadcast_temp = d_temp1;  // reuse (we need out_size elements)
        // Check if d_temp1 has enough space - it should since temp buffers are sized for max
        cudaMemcpyH2D(d_broadcast_temp, h_broadcast.data(), out_size);
        add_tensors_inplace_kernel<<<(out_size + 255) / 256, 256>>>(d_temp2, d_broadcast_temp, out_size);
        CUDA_CHECK_LAST_ERROR();
    }

    norm2.forward(d_temp2, d_temp1, H, W);

    int size2 = out_channels * spatial;
    silu_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_temp1, size2);
    CUDA_CHECK_LAST_ERROR();

    conv2.forward(cudnn, d_temp1, d_temp2, H, W, d_workspace, ws_size);

    // Skip connection
    if (has_skip) {
        skip_conv.forward(cudnn, d_input, d_output, H, W, d_workspace, ws_size);
        add_tensors_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_output, d_temp2, size2);
    } else {
        add_tensors_kernel<<<(size2 + 255) / 256, 256>>>(d_output, d_input, d_temp2, size2);
    }
    CUDA_CHECK_LAST_ERROR();
}

void ResBlock::destroy() {
    norm1.destroy(); norm2.destroy();
    conv1.destroy(); conv2.destroy();
    cudaFree(d_time_proj_w);
    cudaFree(d_time_proj_b);
    if (has_skip) skip_conv.destroy();
}

// ============================================================
// AttentionBlock implementation
// ============================================================

void AttentionBlock::init(int C, int ctx_dim, int heads, int num_groups) {
    channels = C;
    context_dim = ctx_dim;
    n_heads = heads;

    self_norm.init(C, num_groups);
    d_self_wq = cudaMallocDevice<half>(C * C);
    d_self_wk = cudaMallocDevice<half>(C * C);
    d_self_wv = cudaMallocDevice<half>(C * C);
    d_self_wo = cudaMallocDevice<half>(C * C);

    cross_norm.init(C, num_groups);
    d_cross_wq = cudaMallocDevice<half>(C * C);
    d_cross_wk = cudaMallocDevice<half>(ctx_dim * C);
    d_cross_wv = cudaMallocDevice<half>(ctx_dim * C);
    d_cross_wo = cudaMallocDevice<half>(C * C);
}

void AttentionBlock::initRandom(float scale, unsigned& seed) {
    self_norm.initRandom(seed);
    init_random_fp16(d_self_wq, channels * channels, scale, seed++);
    init_random_fp16(d_self_wk, channels * channels, scale, seed++);
    init_random_fp16(d_self_wv, channels * channels, scale, seed++);
    init_random_fp16(d_self_wo, channels * channels, scale, seed++);

    cross_norm.initRandom(seed);
    init_random_fp16(d_cross_wq, channels * channels, scale, seed++);
    init_random_fp16(d_cross_wk, context_dim * channels, scale, seed++);
    init_random_fp16(d_cross_wv, context_dim * channels, scale, seed++);
    init_random_fp16(d_cross_wo, channels * channels, scale, seed++);
}

void AttentionBlock::gemm(cublasHandle_t cublas, const half* A, const half* B, half* C,
                          int m, int n, int k, float alpha, float beta) {
    half alpha_h = __float2half(alpha);
    half beta_h = __float2half(beta);
    CUBLAS_CHECK(cublasHgemm(cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha_h, B, k, A, k,
        &beta_h, C, n));
}

void AttentionBlock::forward(cublasHandle_t cublas,
                             const half* d_input,
                             const half* d_context,
                             int context_len,
                             half* d_output,
                             int H, int W,
                             half* d_q, half* d_k, half* d_v,
                             half* d_attn_out, float* d_scores)
{
// Clean AttentionBlock::forward implementation
    int spatial = H * W;
    int C = channels;
    int head_dim = C / n_heads;
    int total = C * spatial;

    // --- Self-Attention ---
    // GroupNorm on input (NCHW)
    self_norm.forward(d_input, d_output, H, W);

    // Data is [C, spatial] in NCHW memory layout.
    // Project Q, K, V: Y[C, spatial] = W[C, C] @ X[C, spatial]
    // cuBLAS col-major: Y_cm[spatial, C] = X_cm[spatial, C] @ W_cm[C, C]
    // Since X_rm[C, spatial] = X_cm[spatial, C] (reinterpret), and
    // W_rm[C, C] stored contiguously, W_cm = W_rm^T for square matrix.
    // We use: Y_cm = X_cm @ W_cm^T
    // cublasHgemm(T, N, C, spatial, C, W, C, X, C, Y, C)
    {
        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);

        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            C, spatial, C, &alpha_h, d_self_wq, C, d_output, C, &beta_h, d_q, C));
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            C, spatial, C, &alpha_h, d_self_wk, C, d_output, C, &beta_h, d_k, C));
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            C, spatial, C, &alpha_h, d_self_wv, C, d_output, C, &beta_h, d_v, C));
    }

    // Q, K, V are [C, spatial] = [n_heads * head_dim, spatial]
    // = [n_heads, head_dim, spatial] (contiguous per head)
    // Each Q_h is [head_dim, spatial] rm = [spatial, head_dim] cm
    // Scores: Q_h^T @ K_h = [spatial, spatial]
    float scale = 1.0f / sqrtf((float)head_dim);
    {
        float zero_f = 0.0f;
        long long strideQK = (long long)head_dim * spatial;
        long long strideS = (long long)spatial * spatial;

        CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            spatial, spatial, head_dim,
            &scale,
            d_q, CUDA_R_16F, spatial, strideQK,
            d_k, CUDA_R_16F, spatial, strideQK,
            &zero_f,
            d_scores, CUDA_R_32F, spatial, strideS,
            n_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Softmax
    {
        dim3 grid(n_heads, spatial);
        softmax_2d_kernel<<<grid, std::min(spatial, 256)>>>(d_scores, n_heads, spatial, spatial);
        CUDA_CHECK_LAST_ERROR();
    }

    // Attention output: scores @ V -> [n_heads, head_dim, spatial]
    // scores_cm[spatial, spatial], V_cm[spatial, head_dim]
    // out_cm[spatial, head_dim] = scores_cm^T @ V_cm (but scores is not symmetric)
    // We want out_rm[head_dim, spatial] = V_rm[head_dim, spatial] @ scores_rm^T[spatial, spatial]
    // Convert float scores to half (reuse d_k buffer which is no longer needed)
    {
        int scores_size = n_heads * spatial * spatial;
        float_to_half_kernel<<<(scores_size + 255) / 256, 256>>>(
            d_k, d_scores, scores_size);
        CUDA_CHECK_LAST_ERROR();
    }

    // out = scores_h @ V : both FP16
    // out_cm[spatial, hd] = scores_cm^T[spatial, spatial] @ V_cm[spatial, hd]
    {
        half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
        long long strideScores = (long long)spatial * spatial;
        long long strideV = (long long)head_dim * spatial;
        long long strideO = (long long)head_dim * spatial;

        CUBLAS_CHECK(cublasHgemmStridedBatched(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            spatial, head_dim, spatial,
            &one_h,
            d_k, spatial, strideScores,
            d_v, spatial, strideV,
            &zero_h,
            d_attn_out, spatial, strideO,
            n_heads));
    }

    // Output projection: Wo @ attn_out -> d_q (reuse buffer)
    {
        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            C, spatial, C, &alpha_h, d_self_wo, C, d_attn_out, C, &beta_h, d_q, C));
    }

    // Self-attention residual
    add_tensors_kernel<<<(total + 255) / 256, 256>>>(d_output, d_input, d_q, total);
    CUDA_CHECK_LAST_ERROR();

    // --- Cross-Attention ---
    cross_norm.forward(d_output, d_attn_out, H, W);

    // Q from image: [C, spatial]
    {
        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            C, spatial, C, &alpha_h, d_cross_wq, C, d_attn_out, C, &beta_h, d_q, C));
    }

    // K, V from context: context is [context_len, context_dim] rm
    // K[C, context_len] = Wk_eff[C, ctx_dim] @ context^T[ctx_dim, ctxlen]
    // Wk stored as [ctx_dim, C] rm = [C, ctx_dim] cm
    // context rm[ctxlen, ctx_dim] = cm[ctx_dim, ctxlen]
    // K_cm[ctxlen, C] = context_cm^T[ctxlen, ctx_dim] @ Wk_cm^T[ctx_dim, C]
    // m=ctxlen, n=C, k=ctx_dim, transa=T, transb=T
    {
        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_T,
            context_len, C, context_dim,
            &alpha_h,
            d_context, context_dim,
            d_cross_wk, C,
            &beta_h,
            d_k, context_len));
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_T,
            context_len, C, context_dim,
            &alpha_h,
            d_context, context_dim,
            d_cross_wv, C,
            &beta_h,
            d_v, context_len));
    }

    // K, V are now [C, context_len] rm = [n_heads, head_dim, context_len]
    // K_h cm = [context_len, head_dim]
    // Cross-attention scores: Q_h^T @ K_h
    // Q_h cm = [spatial, head_dim], K_h cm = [context_len, head_dim]
    // scores_cm[context_len, spatial] = K_h_cm @ Q_h_cm^T... wait
    // We want scores_rm[spatial, context_len] = scores_cm[context_len, spatial]
    // scores = Q_h^T_rm @ K_h_rm = [spatial, hd] @ [hd, ctxlen] = [spatial, ctxlen] (rm)
    // cm: [ctxlen, spatial]
    // cuBLAS: C_cm[ctxlen, spatial] = op(A)[ctxlen, k] @ op(B)[k, spatial]
    // k = head_dim
    // op(A) = K_h: K_h_cm[ctxlen, hd], op=N -> [ctxlen, hd], m=ctxlen
    // op(B) = Q_h^T: Q_h_cm[spatial, hd], op=T -> [hd, spatial], n=spatial
    {
        float zero_f = 0.0f;
        long long strideQ = (long long)head_dim * spatial;
        long long strideK = (long long)head_dim * context_len;
        long long strideS = (long long)spatial * context_len;

        CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            context_len, spatial, head_dim,
            &scale,
            d_k, CUDA_R_16F, context_len, strideK,
            d_q, CUDA_R_16F, spatial, strideQ,
            &zero_f,
            d_scores, CUDA_R_32F, context_len, strideS,
            n_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // scores_cm[ctxlen, spatial] per head = scores_rm[spatial, ctxlen] per head
    // Softmax over context_len (last dim in rm)
    {
        dim3 grid(n_heads, spatial);
        softmax_2d_kernel<<<grid, std::min(context_len, 256)>>>(d_scores, n_heads, spatial, context_len);
        CUDA_CHECK_LAST_ERROR();
    }

    // Cross-attention output: scores @ V
    // scores_rm[spatial, ctxlen], V_rm[hd, ctxlen] per head
    // out_rm[hd, spatial] = V_rm @ scores_rm^T = [hd, ctxlen] @ [ctxlen, spatial]
    // Convert cross-attention float scores to half (reuse d_q buffer)
    {
        int scores_size = n_heads * spatial * context_len;
        float_to_half_kernel<<<(scores_size + 255) / 256, 256>>>(
            d_q, d_scores, scores_size);
        CUDA_CHECK_LAST_ERROR();
    }

    // Cross-attention output: scores_h @ V (both FP16)
    {
        half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
        long long strideScores = (long long)spatial * context_len;
        long long strideV = (long long)head_dim * context_len;
        long long strideO = (long long)head_dim * spatial;

        CUBLAS_CHECK(cublasHgemmStridedBatched(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            spatial, head_dim, context_len,
            &one_h,
            d_q, context_len, strideScores,
            d_v, context_len, strideV,
            &zero_h,
            d_attn_out, spatial, strideO,
            n_heads));
    }

    // Cross-attention output projection
    {
        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            C, spatial, C, &alpha_h, d_cross_wo, C, d_attn_out, C, &beta_h, d_q, C));
    }

    // Cross-attention residual
    add_tensors_inplace_kernel<<<(total + 255) / 256, 256>>>(d_output, d_q, total);
    CUDA_CHECK_LAST_ERROR();
}

void AttentionBlock::destroy() {
    self_norm.destroy();
    cudaFree(d_self_wq); cudaFree(d_self_wk); cudaFree(d_self_wv); cudaFree(d_self_wo);
    cross_norm.destroy();
    cudaFree(d_cross_wq); cudaFree(d_cross_wk); cudaFree(d_cross_wv); cudaFree(d_cross_wo);
}

// ============================================================
// UNet implementation
// ============================================================

UNet::UNet(const UNetConfig& cfg)
    : cfg_(cfg), d_workspace_(nullptr), workspace_size_(0),
      d_time_embed_(nullptr), d_time_sinusoidal_(nullptr),
      d_buf1_(nullptr), d_buf2_(nullptr), d_buf3_(nullptr),
      d_temp1_(nullptr), d_temp2_(nullptr),
      d_q_buf_(nullptr), d_k_buf_(nullptr), d_v_buf_(nullptr),
      d_attn_out_buf_(nullptr), d_scores_buf_(nullptr),
      max_buf_size_(0), buffers_allocated_(false)
{
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH));

    int num_levels = cfg_.num_levels;
    int temb_dim = cfg_.time_embed_dim;
    int ctx_dim = cfg_.context_dim;
    int ng = cfg_.num_groups;
    int n_res = cfg_.n_res_blocks;

    // Input conv: latent_ch -> base_ch
    input_conv_.init(cfg_.latent_channels, cfg_.base_channels, 3, 1, 1);

    // Time embedding MLP: base_ch -> temb_dim -> temb_dim
    d_time_mlp_w1_ = cudaMallocDevice<half>(cfg_.base_channels * temb_dim);
    d_time_mlp_b1_ = cudaMallocDevice<half>(temb_dim);
    d_time_mlp_w2_ = cudaMallocDevice<half>(temb_dim * temb_dim);
    d_time_mlp_b2_ = cudaMallocDevice<half>(temb_dim);

    d_time_sinusoidal_ = cudaMallocDevice<half>(cfg_.base_channels);
    d_time_embed_ = cudaMallocDevice<half>(temb_dim);

    // Down blocks
    down_res_.resize(num_levels);
    down_attn_.resize(num_levels);
    for (int lvl = 0; lvl < num_levels; lvl++) {
        int ch_in = (lvl == 0) ? cfg_.base_channels : cfg_.channels(lvl - 1);
        int ch_out = cfg_.channels(lvl);

        down_res_[lvl].resize(n_res);
        down_attn_[lvl].resize(n_res);

        for (int r = 0; r < n_res; r++) {
            int in_ch = (r == 0) ? ch_in : ch_out;
            down_res_[lvl][r].init(in_ch, ch_out, temb_dim, ng);
            down_attn_[lvl][r].init(ch_out, ctx_dim, cfg_.n_heads, ng);
        }
    }

    // Downsamplers (stride-2 conv) for all levels except last
    downsamplers_.resize(num_levels - 1);
    for (int lvl = 0; lvl < num_levels - 1; lvl++) {
        int ch = cfg_.channels(lvl);
        downsamplers_[lvl].init(ch, ch, 3, 2, 1);
    }

    // Mid block
    int mid_ch = cfg_.channels(num_levels - 1);
    mid_res1_.init(mid_ch, mid_ch, temb_dim, ng);
    mid_attn_.init(mid_ch, ctx_dim, cfg_.n_heads, ng);
    mid_res2_.init(mid_ch, mid_ch, temb_dim, ng);

    // Up blocks
    up_res_.resize(num_levels);
    up_attn_.resize(num_levels);
    for (int lvl = num_levels - 1; lvl >= 0; lvl--) {
        int ch_out = cfg_.channels(lvl);
        // First ResBlock takes concatenated skip (ch_out + skip_ch)
        int skip_ch = ch_out;  // skip from down block at same level
        int ch_below = (lvl < num_levels - 1) ? cfg_.channels(lvl + 1) : mid_ch;

        // Extra ResBlock for downsample skip, except at the topmost level (lvl=0)
        int n_up_res = (lvl > 0) ? n_res + 1 : n_res;
        up_res_[lvl].resize(n_up_res);
        up_attn_[lvl].resize(n_up_res);

        for (int r = 0; r < n_up_res; r++) {
            int in_ch;
            if (r == 0) {
                // Input is from level below (or mid) upsampled, concatenated with skip
                in_ch = ch_below + skip_ch;
            } else {
                in_ch = ch_out + skip_ch;  // each takes a skip
            }
            up_res_[lvl][r].init(in_ch, ch_out, temb_dim, ng);
            up_attn_[lvl][r].init(ch_out, ctx_dim, cfg_.n_heads, ng);
        }
    }

    // Upsample convs (after nearest-neighbor upsample)
    upsample_convs_.resize(num_levels - 1);
    for (int lvl = num_levels - 1; lvl >= 1; lvl--) {
        int ch = cfg_.channels(lvl);
        upsample_convs_[lvl - 1].init(ch, ch, 3, 1, 1);
    }

    // Output: GroupNorm + SiLU + Conv
    output_norm_.init(cfg_.base_channels, ng);
    output_conv_.init(cfg_.base_channels, cfg_.latent_channels, 3, 1, 1);
}

UNet::~UNet() {
    freeWeights();
    freeBuffers();
    cudnnDestroy(cudnn_handle_);
    cublasDestroy(cublas_handle_);
}

void UNet::freeWeights() {
    input_conv_.destroy();
    cudaFree(d_time_mlp_w1_); cudaFree(d_time_mlp_b1_);
    cudaFree(d_time_mlp_w2_); cudaFree(d_time_mlp_b2_);
    cudaFree(d_time_sinusoidal_); cudaFree(d_time_embed_);

    for (auto& lvl : down_res_) for (auto& r : lvl) r.destroy();
    for (auto& lvl : down_attn_) for (auto& a : lvl) a.destroy();
    for (auto& d : downsamplers_) d.destroy();

    mid_res1_.destroy(); mid_res2_.destroy(); mid_attn_.destroy();

    for (auto& lvl : up_res_) for (auto& r : lvl) r.destroy();
    for (auto& lvl : up_attn_) for (auto& a : lvl) a.destroy();
    for (auto& u : upsample_convs_) u.destroy();

    output_norm_.destroy();
    output_conv_.destroy();
}

void UNet::freeBuffers() {
    if (!buffers_allocated_) return;
    cudaFree(d_workspace_);
    cudaFree(d_buf1_); cudaFree(d_buf2_); cudaFree(d_buf3_);
    cudaFree(d_temp1_); cudaFree(d_temp2_);
    cudaFree(d_q_buf_); cudaFree(d_k_buf_); cudaFree(d_v_buf_);
    cudaFree(d_attn_out_buf_); cudaFree(d_scores_buf_);
    for (auto* p : skip_buffers_) cudaFree(p);
    skip_buffers_.clear();
    buffers_allocated_ = false;
}

void UNet::allocBuffers(int H, int W) {
    if (buffers_allocated_) freeBuffers();

    // Find max channel count and spatial size
    int max_ch = cfg_.base_channels;
    for (int lvl = 0; lvl < cfg_.num_levels; lvl++) {
        max_ch = std::max(max_ch, cfg_.channels(lvl));
    }
    // Max spatial size at input level
    int max_spatial = H * W;
    max_buf_size_ = max_ch * 2 * max_spatial;  // 2x for concat

    // Workspace for cuDNN
    workspace_size_ = 64 * 1024 * 1024;  // 64 MB
    CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_size_));

    // Activation buffers
    d_buf1_ = cudaMallocDevice<half>(max_buf_size_);
    d_buf2_ = cudaMallocDevice<half>(max_buf_size_);
    d_buf3_ = cudaMallocDevice<half>(max_buf_size_);
    d_temp1_ = cudaMallocDevice<half>(max_buf_size_);
    d_temp2_ = cudaMallocDevice<half>(max_buf_size_);

    // Attention buffers
    d_q_buf_ = cudaMallocDevice<half>(max_ch * max_spatial);
    d_k_buf_ = cudaMallocDevice<half>(max_ch * max_spatial);
    d_v_buf_ = cudaMallocDevice<half>(max_ch * max_spatial);
    d_attn_out_buf_ = cudaMallocDevice<half>(max_ch * max_spatial);
    d_scores_buf_ = cudaMallocDevice<float>(max_spatial * max_spatial);

    // Skip connection storage
    // Down path produces: n_res skips per level + 1 downsample skip (except last level)
    int num_skips = 0;
    skip_channels_.clear();
    skip_heights_.clear();
    skip_widths_.clear();
    int h = H, w = W;
    for (int lvl = 0; lvl < cfg_.num_levels; lvl++) {
        int ch = cfg_.channels(lvl);
        for (int r = 0; r < cfg_.n_res_blocks; r++) {
            skip_channels_.push_back(ch);
            skip_heights_.push_back(h);
            skip_widths_.push_back(w);
            num_skips++;
        }
        if (lvl < cfg_.num_levels - 1) {
            // Downsample skip
            h /= 2; w /= 2;
            skip_channels_.push_back(ch);
            skip_heights_.push_back(h);
            skip_widths_.push_back(w);
            num_skips++;
        }
    }

    skip_buffers_.resize(num_skips);
    for (int i = 0; i < num_skips; i++) {
        skip_buffers_[i] = cudaMallocDevice<half>(skip_channels_[i] * skip_heights_[i] * skip_widths_[i]);
    }

    buffers_allocated_ = true;
}

void UNet::initRandom(unsigned long long seed) {
    unsigned s = (unsigned)seed;
    float scale = 0.02f;

    input_conv_.initRandom(scale, s);
    init_random_fp16(d_time_mlp_w1_, cfg_.base_channels * cfg_.time_embed_dim, scale, s++);
    init_zeros_fp16(d_time_mlp_b1_, cfg_.time_embed_dim);
    init_random_fp16(d_time_mlp_w2_, cfg_.time_embed_dim * cfg_.time_embed_dim, scale, s++);
    init_zeros_fp16(d_time_mlp_b2_, cfg_.time_embed_dim);

    for (auto& lvl : down_res_) for (auto& r : lvl) r.initRandom(scale, s);
    for (auto& lvl : down_attn_) for (auto& a : lvl) a.initRandom(scale, s);
    for (auto& d : downsamplers_) d.initRandom(scale, s);

    mid_res1_.initRandom(scale, s);
    mid_attn_.initRandom(scale, s);
    mid_res2_.initRandom(scale, s);

    for (auto& lvl : up_res_) for (auto& r : lvl) r.initRandom(scale, s);
    for (auto& lvl : up_attn_) for (auto& a : lvl) a.initRandom(scale, s);
    for (auto& u : upsample_convs_) u.initRandom(scale, s);

    output_norm_.initRandom(s);
    output_conv_.initRandom(scale, s);

    printf("[UNet] Initialized with random weights (base_ch=%d, levels=%d, heads=%d)\n",
           cfg_.base_channels, cfg_.num_levels, cfg_.n_heads);
}

void UNet::computeTimeEmbedding(int timestep) {
    int base_ch = cfg_.base_channels;
    int temb_dim = cfg_.time_embed_dim;

    // Sinusoidal embedding: timestep -> [base_ch]
    sinusoidal_embedding_kernel<<<1, base_ch / 2>>>(
        d_time_sinusoidal_, timestep, base_ch, 10000.0f);
    CUDA_CHECK_LAST_ERROR();

    // MLP: Linear -> SiLU -> Linear
    // Layer 1: [1, base_ch] -> [1, temb_dim]
    half alpha_h = __float2half(1.0f);
    half beta_h = __float2half(0.0f);
    CUBLAS_CHECK(cublasHgemm(cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        temb_dim, 1, base_ch,
        &alpha_h,
        d_time_mlp_w1_, base_ch,
        d_time_sinusoidal_, base_ch,
        &beta_h,
        d_time_embed_, temb_dim));

    add_tensors_inplace_kernel<<<(temb_dim + 255) / 256, 256>>>(
        d_time_embed_, d_time_mlp_b1_, temb_dim);
    silu_inplace_kernel<<<(temb_dim + 255) / 256, 256>>>(d_time_embed_, temb_dim);
    CUDA_CHECK_LAST_ERROR();

    // Layer 2: [1, temb_dim] -> [1, temb_dim]
    // Layer 2 uses d_temp1_ as output buffer
    CUBLAS_CHECK(cublasHgemm(cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        temb_dim, 1, temb_dim,
        &alpha_h,
        d_time_mlp_w2_, temb_dim,
        d_time_embed_, temb_dim,
        &beta_h,
        d_temp1_, temb_dim));

    // Copy back + add bias
    cudaMemcpyD2D(d_time_embed_, d_temp1_, temb_dim);
    add_tensors_inplace_kernel<<<(temb_dim + 255) / 256, 256>>>(
        d_time_embed_, d_time_mlp_b2_, temb_dim);
    CUDA_CHECK_LAST_ERROR();
}

void UNet::forward(const half* d_latent,
                   const half* d_context, int context_len,
                   int timestep,
                   half* d_output,
                   int H, int W)
{
    if (!buffers_allocated_) allocBuffers(H, W);

    // 1. Compute time embedding
    computeTimeEmbedding(timestep);

    // 2. Input conv: [1, 4, H, W] -> [1, base_ch, H, W]
    input_conv_.forward(cudnn_handle_, d_latent, d_buf1_, H, W,
                        d_workspace_, workspace_size_);

    // 3. Down path
    int h = H, w = W;
    int skip_idx = 0;
    half* d_current = d_buf1_;

    for (int lvl = 0; lvl < cfg_.num_levels; lvl++) {
        for (int r = 0; r < cfg_.n_res_blocks; r++) {
            // ResBlock
            down_res_[lvl][r].forward(cudnn_handle_, cublas_handle_,
                d_current, d_time_embed_, d_buf2_, h, w,
                d_workspace_, workspace_size_, d_temp1_, d_temp2_);

            // Attention
            down_attn_[lvl][r].forward(cublas_handle_,
                d_buf2_, d_context, context_len,
                d_buf3_, h, w,
                d_q_buf_, d_k_buf_, d_v_buf_, d_attn_out_buf_, d_scores_buf_);

            // Save skip
            int skip_size = cfg_.channels(lvl) * h * w;
            cudaMemcpyD2D(skip_buffers_[skip_idx], d_buf3_, skip_size);
            skip_idx++;

            // Swap buffers
            d_current = d_buf3_;
            std::swap(d_buf1_, d_buf3_);
        }

        // Downsample (except last level)
        if (lvl < cfg_.num_levels - 1) {
            downsamplers_[lvl].forward(cudnn_handle_, d_current, d_buf2_,
                h, w, d_workspace_, workspace_size_);
            h /= 2; w /= 2;

            int skip_size = cfg_.channels(lvl) * h * w;
            cudaMemcpyD2D(skip_buffers_[skip_idx], d_buf2_, skip_size);
            skip_idx++;

            d_current = d_buf2_;
            std::swap(d_buf1_, d_buf2_);
        }
    }

    // 4. Mid block
    mid_res1_.forward(cudnn_handle_, cublas_handle_,
        d_current, d_time_embed_, d_buf2_, h, w,
        d_workspace_, workspace_size_, d_temp1_, d_temp2_);

    mid_attn_.forward(cublas_handle_,
        d_buf2_, d_context, context_len,
        d_buf3_, h, w,
        d_q_buf_, d_k_buf_, d_v_buf_, d_attn_out_buf_, d_scores_buf_);

    mid_res2_.forward(cudnn_handle_, cublas_handle_,
        d_buf3_, d_time_embed_, d_buf1_, h, w,
        d_workspace_, workspace_size_, d_temp1_, d_temp2_);

    d_current = d_buf1_;

    // 5. Up path (reverse order)
    for (int lvl = cfg_.num_levels - 1; lvl >= 0; lvl--) {
        int ch_out = cfg_.channels(lvl);
        int n_up_res = (lvl > 0) ? cfg_.n_res_blocks + 1 : cfg_.n_res_blocks;

        for (int r = 0; r < n_up_res; r++) {
            // Pop skip connection
            skip_idx--;
            int skip_ch = skip_channels_[skip_idx];
            int skip_h = skip_heights_[skip_idx];
            int skip_w = skip_widths_[skip_idx];

            // Concatenate current with skip along channels
            int current_ch;
            if (lvl == cfg_.num_levels - 1 && r == 0) {
                current_ch = cfg_.channels(cfg_.num_levels - 1);  // from mid block
            } else {
                current_ch = ch_out;  // from previous up resblock
            }

            // If spatial dims don't match, we need to handle it
            // (they should match since skips are stored at correct resolution)

            int cat_size = (current_ch + skip_ch) * skip_h * skip_w;
            concat_channels_kernel<<<(cat_size + 255) / 256, 256>>>(
                d_buf2_,
                d_current, current_ch,
                skip_buffers_[skip_idx], skip_ch,
                skip_h, skip_w);
            CUDA_CHECK_LAST_ERROR();

            // ResBlock on concatenated features
            up_res_[lvl][r].forward(cudnn_handle_, cublas_handle_,
                d_buf2_, d_time_embed_, d_buf3_, skip_h, skip_w,
                d_workspace_, workspace_size_, d_temp1_, d_temp2_);

            // Attention
            up_attn_[lvl][r].forward(cublas_handle_,
                d_buf3_, d_context, context_len,
                d_buf1_, skip_h, skip_w,
                d_q_buf_, d_k_buf_, d_v_buf_, d_attn_out_buf_, d_scores_buf_);

            d_current = d_buf1_;
        }

        // Upsample (except first level going to output)
        if (lvl > 0) {
            int ch = cfg_.channels(lvl);
            int up_size = ch * h * 2 * w * 2;
            upsample_nearest_2x_kernel<<<(up_size + 255) / 256, 256>>>(
                d_buf2_, d_current, ch, h, w);
            CUDA_CHECK_LAST_ERROR();

            upsample_convs_[lvl - 1].forward(cudnn_handle_, d_buf2_, d_buf3_,
                h * 2, w * 2, d_workspace_, workspace_size_);

            h *= 2; w *= 2;
            d_current = d_buf3_;
            std::swap(d_buf1_, d_buf3_);
        }
    }

    // 6. Output: GroupNorm -> SiLU -> Conv
    int out_ch = cfg_.base_channels;
    output_norm_.forward(d_current, d_buf2_, h, w);

    int out_size = out_ch * h * w;
    silu_inplace_kernel<<<(out_size + 255) / 256, 256>>>(d_buf2_, out_size);
    CUDA_CHECK_LAST_ERROR();

    output_conv_.forward(cudnn_handle_, d_buf2_, d_output, h, w,
                         d_workspace_, workspace_size_);
}
