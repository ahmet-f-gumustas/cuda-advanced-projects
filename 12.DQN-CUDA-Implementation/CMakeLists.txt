cmake_minimum_required(VERSION 3.18)
project(DQN_CUDA_Implementation CUDA CXX)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Detect GPU architecture automatically, fallback to common ones
include(FindCUDA/select_compute_arch)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create static library from CUDA sources
add_library(dqn_cuda STATIC
    src/dqn_kernels.cu
    src/dqn_network.cu
)

set_target_properties(dqn_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Training executable
add_executable(train src/train.cu)
target_link_libraries(train dqn_cuda CUDA::cudart)
set_target_properties(train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Prediction/Test executable
add_executable(predict src/predict.cu)
target_link_libraries(predict dqn_cuda CUDA::cudart)
set_target_properties(predict PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Print build info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
