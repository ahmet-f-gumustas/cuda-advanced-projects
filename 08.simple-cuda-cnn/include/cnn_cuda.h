#ifndef CNN_CUDA_H
#define CNN_CUDA_H

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, \
                    cudaGetErrorString(error)); \
        } \
    } while(0)

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel launcher declarations
void launch_conv2d(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
);

void launch_relu(float* d_data, int size);

void launch_sigmoid(float* d_data, int size);

void launch_max_pool(
    const float* d_input,
    float* d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
);

void launch_avg_pool(
    const float* d_input,
    float* d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
);

// C++ API
typedef struct {
    float* d_input;
    float* d_filter;
    float* d_bias;         // Bias için GPU memory
    float* d_output;
    float* d_temp;
    int input_width;
    int input_height;
    int output_width;
    int output_height;
    float last_inference_time_ms;  // Son inference süresi (ms)
} CNN_Context;

CNN_Context* create_cnn_context(int input_width, int input_height);
void destroy_cnn_context(CNN_Context* ctx);

void set_input_data(CNN_Context* ctx, const float* h_input, int size);
void set_filter_data(CNN_Context* ctx, const float* h_filter, int size);
void set_bias_data(CNN_Context* ctx, const float* h_bias, int size);
void get_output_data(CNN_Context* ctx, float* h_output, int size);

void run_cnn_forward(CNN_Context* ctx);
float get_last_inference_time(CNN_Context* ctx);

#ifdef __cplusplus
}
#endif

#endif // CNN_CUDA_H
