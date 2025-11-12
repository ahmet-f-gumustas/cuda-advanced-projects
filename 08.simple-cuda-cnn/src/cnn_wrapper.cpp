#include "cnn_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" {

CNN_Context* create_cnn_context(int input_width, int input_height) {
    CNN_Context* ctx = (CNN_Context*)malloc(sizeof(CNN_Context));

    ctx->input_width = input_width;
    ctx->input_height = input_height;
    ctx->output_width = input_width - 2;  // 3x3 conv reduces size by 2
    ctx->output_height = input_height - 2;
    ctx->last_inference_time_ms = 0.0f;

    // Allocate GPU memory
    int input_size = input_width * input_height * sizeof(float);
    int output_size = ctx->output_width * ctx->output_height * sizeof(float);
    int filter_size = 3 * 3 * sizeof(float);

    cudaMalloc(&ctx->d_input, input_size);
    cudaMalloc(&ctx->d_filter, filter_size);
    cudaMalloc(&ctx->d_output, output_size);
    cudaMalloc(&ctx->d_temp, output_size);

    return ctx;
}

void destroy_cnn_context(CNN_Context* ctx) {
    if (ctx) {
        cudaFree(ctx->d_input);
        cudaFree(ctx->d_filter);
        cudaFree(ctx->d_output);
        cudaFree(ctx->d_temp);
        free(ctx);
    }
}

void set_input_data(CNN_Context* ctx, const float* h_input, int size) {
    cudaMemcpy(ctx->d_input, h_input, size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void set_filter_data(CNN_Context* ctx, const float* h_filter, int size) {
    cudaMemcpy(ctx->d_filter, h_filter, size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void get_output_data(CNN_Context* ctx, float* h_output, int size) {
    cudaMemcpy(h_output, ctx->d_output, size * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void run_cnn_forward(CNN_Context* ctx) {
    // CUDA events ile timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Step 1: Convolution
    launch_conv2d(
        ctx->d_input,
        ctx->d_filter,
        ctx->d_temp,
        ctx->input_width,
        ctx->input_height,
        ctx->output_width,
        ctx->output_height
    );

    // Step 2: ReLU activation
    int output_size = ctx->output_width * ctx->output_height;
    launch_relu(ctx->d_temp, output_size);

    // Step 3: Max pooling (2x2)
    int pooled_width = ctx->output_width / 2;
    int pooled_height = ctx->output_height / 2;

    launch_max_pool(
        ctx->d_temp,
        ctx->d_output,
        ctx->output_width,
        ctx->output_height,
        pooled_width,
        pooled_height
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Süreyi hesapla (milisaniye)
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    ctx->last_inference_time_ms = milliseconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float get_last_inference_time(CNN_Context* ctx) {
    return ctx->last_inference_time_ms;
}

} // extern "C"
