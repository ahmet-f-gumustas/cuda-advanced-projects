#include "../kernels/conv_kernels.cuh"
#include <cuda_fp16.h>
#include <mma.h>

namespace deep_engine {
namespace kernels {

// Shared memory tile sizes for different architectures
constexpr int TILE_SIZE_M = 16;
constexpr int TILE_SIZE_N = 16;
constexpr int TILE_SIZE_K = 16;

// Warp-level primitive kullanarak matrix multiply
// Ampere ve üstü için tensor core kullanımı
template<typename T>
__global__ void gemm_kernel_optimized(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Shared memory allocation
    extern __shared__ char shared_mem[];
    T* shared_A = reinterpret_cast<T*>(shared_mem);
    T* shared_B = reinterpret_cast<T*>(shared_mem + TILE_SIZE_M * TILE_SIZE_K * sizeof(T));
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    
    const int row = bid_y * TILE_SIZE_M + tid_y;
    const int col = bid_x * TILE_SIZE_N + tid_x;
    
    T accumulator = 0;
    
    // Tile-based computation
    for (int tile_k = 0; tile_k < (K + TILE_SIZE_K - 1) / TILE_SIZE_K; ++tile_k) {
        // Collaborative loading into shared memory
        if (row < M && tile_k * TILE_SIZE_K + tid_x < K) {
            shared_A[tid_y * TILE_SIZE_K + tid_x] = A[row * K + tile_k * TILE_SIZE_K + tid_x];
        } else {
            shared_A[tid_y * TILE_SIZE_K + tid_x] = 0;
        }
        
        if (col < N && tile_k * TILE_SIZE_K + tid_y < K) {
            shared_B[tid_y * TILE_SIZE_N + tid_x] = B[(tile_k * TILE_SIZE_K + tid_y) * N + col];
        } else {
            shared_B[tid_y * TILE_SIZE_N + tid_x] = 0;
        }
        
        __syncthreads();
        
        // Compute on tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_K; ++k) {
            accumulator += shared_A[tid_y * TILE_SIZE_K + k] * shared_B[k * TILE_SIZE_N + tid_x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * accumulator + beta * C[row * N + col];
    }
}

// Im2col implementation - convolution'ı matrix multiply'a çevirmek için
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    float* __restrict__ data_col,
    int output_h, int output_w)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = channels * kernel_h * kernel_w * output_h * output_w;
    
    if (index >= total_elements) return;
    
    // Hangi output pixel'i için çalışıyoruz
    const int w_out = index % output_w;
    const int h_out = (index / output_w) % output_h;
    const int channel_in = (index / (output_w * output_h)) % channels;
    const int kernel_idx = index / (output_w * output_h * channels);
    
    const int kh = kernel_idx / kernel_w;
    const int kw = kernel_idx % kernel_w;
    
    // Input koordinatları
    const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
    const int w_in = w_out * stride_w - pad_w + kw * dilation_w;
    
    // Boundary check ve data kopyalama
    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
        data_col[index] = data_im[channel_in * height * width + h_in * width + w_in];
    } else {
        data_col[index] = 0.0f;
    }
}

// 1x1 convolution için optimize edilmiş kernel
// Bu aslında sadece matrix multiply, overhead'i azaltıyoruz
template<int BLOCK_SIZE>
__global__ void conv1x1_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width)
{
    __shared__ float shared_input[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 bank conflict için
    __shared__ float shared_weight[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_ch = blockIdx.y * BLOCK_SIZE + ty;
    const int spatial_idx = blockIdx.x * BLOCK_SIZE + tx;
    
    const int h = spatial_idx / width;
    const int w = spatial_idx % width;
    const int batch = blockIdx.z;
    
    float sum = 0.0f;
    
    // Tile üzerinden iterasyon
    for (int k = 0; k < (in_channels + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        // Input tile'ı yükle
        if (spatial_idx < height * width && k * BLOCK_SIZE + ty < in_channels) {
            shared_input[ty][tx] = input[batch * in_channels * height * width +
                                        (k * BLOCK_SIZE + ty) * height * width + spatial_idx];
        } else {
            shared_input[ty][tx] = 0.0f;
        }
        
        // Weight tile'ı yükle
        if (out_ch < out_channels && k * BLOCK_SIZE + tx < in_channels) {
            shared_weight[ty][tx] = weight[out_ch * in_channels + k * BLOCK_SIZE + tx];
        } else {
            shared_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += shared_weight[ty][i] * shared_input[i][tx];
        }
        
        __syncthreads();
    }
    
    // Bias ekle ve sonucu yaz
    if (out_ch < out_channels && spatial_idx < height * width) {
        sum += (bias != nullptr) ? bias[out_ch] : 0.0f;
        output[batch * out_channels * height * width + out_ch * height * width + spatial_idx] = sum;
    }
}

// Depthwise convolution kernel - MobileNet tarzı modeller için kritik
__global__ void depthwise_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    const int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z % channels;
    const int batch = blockIdx.z / channels;
    
    if (out_w >= out_width || out_h >= out_height) return;
    
    float sum = 0.0f;
    
    // Convolution computation
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int in_h = out_h * stride_h - pad_h + kh;
            const int in_w = out_w * stride_w - pad_w + kw;
            
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                const int input_idx = ((batch * channels + channel) * in_height + in_h) * in_width + in_w;
                const int weight_idx = channel * kernel_h * kernel_w + kh * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add bias and write output
    sum += (bias != nullptr) ? bias[channel] : 0.0f;
    const int output_idx = ((batch * channels + channel) * out_height + out_h) * out_width + out_w;
    output[output_idx] = sum;
}

// INT8 quantized convolution için özel kernel
__global__ void conv_int8_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ output,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int pad)
{
    // DP4A instruction kullanarak INT8 dot product
    // RTX serisi GPU'larda bu çok hızlı çalışır
    
    extern __shared__ int32_t shared_mem[];
    int32_t* shared_input = shared_mem;
    int32_t* shared_weight = shared_mem + blockDim.x * blockDim.y;
    
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z;
    
    if (out_x >= width || out_y >= height) return;
    
    int32_t sum = 0;
    
    // Her 4 int8 değeri bir int32'ye pack'leyip DP4A kullanabiliriz
    for (int in_ch = 0; in_ch < in_channels; in_ch += 4) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_y = out_y * stride - pad + ky;
                const int in_x = out_x * stride - pad + kx;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    // 4 kanal birden yükle
                    int32_t packed_input = 0;
                    int32_t packed_weight = 0;
                    
                    #pragma unroll
                    for (int c = 0; c < 4 && (in_ch + c) < in_channels; ++c) {
                        int8_t in_val = input[((in_ch + c) * height + in_y) * width + in_x];
                        int8_t w_val = weight[(out_ch * in_channels + in_ch + c) * kernel_size * kernel_size + 
                                             ky * kernel_size + kx];
                        
                        packed_input |= (in_val & 0xFF) << (c * 8);
                        packed_weight |= (w_val & 0xFF) << (c * 8);
                    }
                    
                    // DP4A instruction
                    sum = __dp4a(packed_input, packed_weight, sum);
                }
            }
        }
    }
    
    output[out_ch * height * width + out_y * width + out_x] = sum;
}

// Winograd F(2x2, 3x3) implementation
// 3x3 convolution'ı 2.25x hızlandırır
__device__ void winograd_transform_input_tile(const float* tile, float* transformed) {
    // Input transform matrix BT * d * B
    const float BT[4][4] = {
        { 1,  0, -1,  0},
        { 0,  1,  1,  0},
        { 0, -1,  1,  0},
        { 0,  1,  0, -1}
    };
    
    float temp[4][4];
    
    // BT * d
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            temp[i][j] = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                temp[i][j] += BT[i][k] * tile[k * 4 + j];
            }
        }
    }
    
    // (BT * d) * B
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            transformed[i * 4 + j] = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                transformed[i * 4 + j] += temp[i][k] * BT[j][k];  // B = BT^T
            }
        }
    }
}

__global__ void winograd_conv3x3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight_transformed,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int pad)
{
    // Her thread bir output tile hesaplar
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z;
    
    const int num_tiles_x = (width + 1) / 2;  // 2x2 output tiles
    const int num_tiles_y = (height + 1) / 2;
    
    if (tile_x >= num_tiles_x || tile_y >= num_tiles_y) return;
    
    __shared__ float shared_input[64][17];  // 16 channels + padding
    float input_tile[16];
    float output_tile[4];
    
    // Her output channel için
    memset(output_tile, 0, sizeof(output_tile));
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        // 4x4 input tile'ı yükle
        #pragma unroll
        for (int dy = 0; dy < 4; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 4; ++dx) {
                int y = tile_y * 2 - pad + dy;
                int x = tile_x * 2 - pad + dx;
                
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    input_tile[dy * 4 + dx] = input[(in_ch * height + y) * width + x];
                } else {
                    input_tile[dy * 4 + dx] = 0.0f;
                }
            }
        }
        
        // Transform input
        float transformed_input[16];
        winograd_transform_input_tile(input_tile, transformed_input);
        
        // Element-wise multiply with transformed weights
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            output_tile[i % 4] += transformed_input[i] * 
                weight_transformed[(out_ch * in_channels + in_ch) * 16 + i];
        }
    }
    
    // Output transform ve yazma
    // AT * output_tile * A
    const float AT[2][4] = {
        {1, 1, 1, 0},
        {0, 1, -1, -1}
    };
    
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int out_y = tile_y * 2 + dy;
            int out_x = tile_x * 2 + dx;
            
            if (out_y < height && out_x < width) {
                float val = 0;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    val += AT[dy][k] * output_tile[k] * AT[dx][k];
                }
                output[out_ch * height * width + out_y * width + out_x] = val;
            }
        }
    }
}

// Tensor Core kullanarak convolution (Ampere ve üstü)
#if __CUDA_ARCH__ >= 800
__global__ void conv_tensorcore_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int M, int N, int K)
{
    using namespace nvcuda;
    
    // Warp-level matrix fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Initialize output to zero
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    // Main GEMM loop
    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * 16;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, input + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, weight + bRow * N + bCol, N);
            
            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store the output
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(output + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
#endif

} // namespace kernels
} // namespace deep_engine