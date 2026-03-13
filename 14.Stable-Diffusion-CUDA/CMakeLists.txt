cmake_minimum_required(VERSION 3.18)
project(StableDiffusionCUDA LANGUAGES CXX CUDA)

# ============================================================
# Standards
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================
# CUDA Architectures: Turing(75) Ampere(80,86) Ada(89)
# ============================================================
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# ============================================================
# Compiler flags
# ============================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr -lineinfo")

# ============================================================
# Dependencies
# ============================================================
find_package(CUDAToolkit REQUIRED)

# Find cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
          ${CUDA_TOOLKIT_ROOT_DIR}/include
          /usr/include
          /usr/local/cuda/include)

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          /usr/lib/x86_64-linux-gnu
          /usr/local/cuda/lib64)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found! Set CUDNN_INCLUDE_DIR and CUDNN_LIBRARY.")
endif()

message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")

# ============================================================
# Static library: sd_cuda
# ============================================================
add_library(sd_cuda STATIC
    src/diffusion_kernels.cu
    src/scheduler.cu
    src/clip.cu
    src/unet.cu
    src/vae.cu
    src/pipeline.cu
)

target_include_directories(sd_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDNN_INCLUDE_DIR}
)

target_link_libraries(sd_cuda PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUDNN_LIBRARY}
)

set_target_properties(sd_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================
# Executables
# ============================================================

# Inference executable
add_executable(sd_inference src/inference.cu)
target_link_libraries(sd_inference PRIVATE sd_cuda)
target_include_directories(sd_inference PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(sd_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Benchmark executable
add_executable(sd_benchmark src/benchmark.cu)
target_link_libraries(sd_benchmark PRIVATE sd_cuda)
target_include_directories(sd_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(sd_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Test executable
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE sd_cuda)
target_include_directories(test_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================
# Output directories
# ============================================================
set_target_properties(sd_inference sd_benchmark test_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
