cmake_minimum_required(VERSION 3.18)
project(DeepInferenceEngine CUDA CXX)

# CUDA ayarları
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architectures - RTX 4070 için
set(CMAKE_CUDA_ARCHITECTURES "89")  # Ada Lovelace

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -use_fast_math")

# Debug vs Release
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
endif()

# Find packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include
    PATH_SUFFIXES cuda)

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    PATH_SUFFIXES cuda)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found!")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

# Source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/tensor.cu
    src/core/layer.cpp
    src/core/graph.cpp
    src/core/allocator.cu
)

set(LAYER_SOURCES
    src/layers/convolution.cpp
    src/layers/convolution.cu
    src/layers/pooling.cpp
    src/layers/pooling.cu
    src/layers/activation.cpp
    src/layers/activation.cu
    src/layers/batchnorm.cpp
    src/layers/dense.cpp
    src/layers/dense.cu
    src/layers/softmax.cu
)

set(KERNEL_SOURCES
    src/kernels/conv_kernels.cu
    src/kernels/gemm_kernels.cu
    src/kernels/activation_kernels.cu
    src/kernels/reduction_kernels.cu
)

set(OPTIMIZATION_SOURCES
    src/optimizations/quantization.cpp
    src/optimizations/fusion.cpp
    src/optimizations/memory_pool.cpp
)

set(UTIL_SOURCES
    src/utils/profiler.cpp
    src/utils/logger.cpp
    src/utils/model_loader.cpp
)

# Main library
add_library(deep_engine SHARED
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${KERNEL_SOURCES}
    ${OPTIMIZATION_SOURCES}
    ${UTIL_SOURCES}
)

# Link libraries
target_link_libraries(deep_engine
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    stdc++fs  # filesystem
)

# Optional: TensorRT support
find_library(TENSORRT_LIB nvinfer)
if(TENSORRT_LIB)
    message(STATUS "TensorRT found, enabling TensorRT backend")
    target_compile_definitions(deep_engine PRIVATE HAVE_TENSORRT)
    target_link_libraries(deep_engine ${TENSORRT_LIB})
endif()

# Test executable
add_executable(test_engine
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_convolution.cpp
    tests/test_graph.cpp
)

target_link_libraries(test_engine deep_engine)

# Benchmark executable
add_executable(benchmark_engine
    benchmarks/benchmark_main.cpp
    benchmarks/benchmark_conv.cpp
    benchmarks/benchmark_gemm.cpp
)

target_link_libraries(benchmark_engine deep_engine)

# Example applications
add_executable(resnet_inference examples/resnet_inference.cpp)
target_link_libraries(resnet_inference deep_engine)

add_executable(yolo_inference examples/yolo_inference.cpp)
target_link_libraries(yolo_inference deep_engine)

# Installation
install(TARGETS deep_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/deep_engine
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DeepEngineConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

# Enable testing
enable_testing()
add_test(NAME tensor_test COMMAND test_engine --test-tensor)
add_test(NAME conv_test COMMAND test_engine --test-conv)
add_test(NAME graph_test COMMAND test_engine --test-graph)