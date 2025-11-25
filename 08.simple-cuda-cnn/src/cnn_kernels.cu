#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/cnn_cuda.h"

// Basit 2D Convolution Kernel (3x3 filter)
__global__ void conv2d_kernel(
    const float* input,
    const float* filter,
    float* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        float sum = 0.0f;

        // 3x3 convolution
        for (int fy = 0; fy < 3; fy++) {
            for (int fx = 0; fx < 3; fx++) {
                int input_x = x + fx;
                int input_y = y + fy;

                if (input_x < input_width && input_y < input_height) {
                    sum += input[input_y * input_width + input_x] *
                           filter[fy * 3 + fx];
                }
            }
        }

        output[y * output_width + x] = sum;
    }
}

// ReLU Activation Kernel
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Sigmoid Activation Kernel
__global__ void sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// Max Pooling Kernel (2x2)
__global__ void max_pool_kernel(
    const float* input,
    float* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        float max_val = -1e9f;

        // 2x2 pooling
        for (int py = 0; py < 2; py++) {
            for (int px = 0; px < 2; px++) {
                int input_x = x * 2 + px;
                int input_y = y * 2 + py;

                if (input_x < input_width && input_y < input_height) {
                    float val = input[input_y * input_width + input_x];
                    max_val = fmaxf(max_val, val);
                }
            }
        }

        output[y * output_width + x] = max_val;
    }
}

// Average Pooling Kernel (2x2)
__global__ void avg_pool_kernel(
    const float* input,
    float* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        float sum = 0.0f;
        int count = 0;

        // 2x2 pooling
        for (int py = 0; py < 2; py++) {
            for (int px = 0; px < 2; px++) {
                int input_x = x * 2 + px;
                int input_y = y * 2 + py;

                if (input_x < input_width && input_y < input_height) {
                    sum += input[input_y * input_width + input_x];
                    count++;
                }
            }
        }

        output[y * output_width + x] = (count > 0) ? (sum / count) : 0.0f;
    }
}

// C API Functions
extern "C" {

void launch_conv2d(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y
    );

    conv2d_kernel<<<grid, block>>>(
        d_input, d_filter, d_output,
        input_width, input_height,
        output_width, output_height
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_relu(float* d_data, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(d_data, size);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_sigmoid(float* d_data, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    sigmoid_kernel<<<blocks, threads>>>(d_data, size);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_max_pool(
    const float* d_input,
    float* d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y
    );

    max_pool_kernel<<<grid, block>>>(
        d_input, d_output,
        input_width, input_height,
        output_width, output_height
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_avg_pool(
    const float* d_input,
    float* d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y
    );

    avg_pool_kernel<<<grid, block>>>(
        d_input, d_output,
        input_width, input_height,
        output_width, output_height
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // extern "C"
