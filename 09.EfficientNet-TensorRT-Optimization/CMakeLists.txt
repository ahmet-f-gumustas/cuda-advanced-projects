cmake_minimum_required(VERSION 3.18)
project(EfficientNetTensorRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA Architecture (RTX 4070 = 89, RTX 3090 = 86, Jetson Orin = 87)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_ONNXPARSER nvonnxparser
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT or install TensorRT.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Source files
set(SOURCES
    src/trt_engine.cpp
    src/calibrator.cpp
    src/cuda_preprocess.cu
    src/main.cpp
)

# Executable
add_executable(efficientnet_trt ${SOURCES})

# Set CUDA properties
set_target_properties(efficientnet_trt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(efficientnet_trt
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER}
    ${TENSORRT_PLUGIN}
    CUDA::cudart
    CUDA::cuda_driver
)

# Benchmark executable (optional, separate build)
add_executable(benchmark_trt
    src/trt_engine.cpp
    src/calibrator.cpp
    src/cuda_preprocess.cu
    src/benchmark.cpp
)

set_target_properties(benchmark_trt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(benchmark_trt
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER}
    ${TENSORRT_PLUGIN}
    CUDA::cudart
    CUDA::cuda_driver
)

# Output directory
set_target_properties(efficientnet_trt benchmark_trt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin
)

# Installation
install(TARGETS efficientnet_trt benchmark_trt
    RUNTIME DESTINATION bin
)
