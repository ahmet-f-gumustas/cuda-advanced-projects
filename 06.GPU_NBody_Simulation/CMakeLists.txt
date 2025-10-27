cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPU_NBody_Simulation LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package (require 12.6+)
find_package(CUDAToolkit 12.6 REQUIRED)

# CUDA architecture - RTX 4070 = sm_89
# Set multiple architectures for compatibility
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cu
    src/nbody_kernels.cu
)

# Executable
add_executable(nbody_sim ${SOURCES})

# Link CUDA libraries
target_link_libraries(nbody_sim CUDA::cudart CUDA::cuda_driver)

# Set CUDA properties
set_target_properties(nbody_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install target
install(TARGETS nbody_sim DESTINATION bin)

# Print configuration
message(STATUS "==================================================")
message(STATUS "GPU N-Body Simulation Configuration")
message(STATUS "==================================================")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "==================================================")
