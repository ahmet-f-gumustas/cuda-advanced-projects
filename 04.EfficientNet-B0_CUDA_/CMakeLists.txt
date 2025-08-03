cmake_minimum_required(VERSION 3.18)
project(EfficientNetB0_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

# Find cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include
    PATHS /usr/include /usr/local/cuda/include)

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
    PATHS /usr/lib64 /usr/lib /usr/local/cuda/lib64 /usr/local/cuda/lib)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found")
endif()

# Find OpenCV for image loading
find_package(OpenCV REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# CUDA architectures - RTX 4070 is Ada Lovelace (8.9)
set(CMAKE_CUDA_ARCHITECTURES "89")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/tensor_utils.cpp
    src/layers.cpp
    src/mbconv.cpp
    src/efficientnet_b0.cpp
    src/activations.cu
)

# Create executable
add_executable(effnet_cuda ${SOURCES})

# Link libraries
target_link_libraries(effnet_cuda 
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
    ${OpenCV_LIBS}
    cublas
    curand
)

# Set CUDA properties
set_property(TARGET effnet_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Add compile definitions
target_compile_definitions(effnet_cuda PRIVATE 
    CUDNN_MAJOR=8
    USE_TENSOR_CORES=1
)

# Installation
install(TARGETS effnet_cuda DESTINATION bin)