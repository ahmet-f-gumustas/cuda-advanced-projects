cmake_minimum_required(VERSION 3.18)
project(SimpleCudaCNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/cnn_wrapper.cpp
    src/cnn_kernels.cu
)

# Create shared library (.so)
add_library(cnn_cuda SHARED ${SOURCES})

# Set CUDA properties
set_target_properties(cnn_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA libraries
target_link_libraries(cnn_cuda
    ${CUDA_LIBRARIES}
)

# Set output directory
set_target_properties(cnn_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)

# Installation
install(TARGETS cnn_cuda
    LIBRARY DESTINATION lib
)
