cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(real_time_video_stabilization LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# CUDA architecture for RTX 4070 (Ada Lovelace - Compute 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Source files
set(CUDA_SOURCES
    src/cuda/optical_flow.cu
    src/cuda/motion_estimation.cu
    src/cuda/frame_warping.cu
    src/cuda/gaussian_pyramid.cu
)

set(CPP_SOURCES
    src/cpp/video_stabilizer.cpp
    src/cpp/trajectory_smoother.cpp
    src/cpp/memory_manager.cpp
    src/main.cpp
)

# Create the executable
add_executable(video_stabilizer ${CPP_SOURCES} ${CUDA_SOURCES})

# Set CUDA properties
set_property(TARGET video_stabilizer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET video_stabilizer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
target_link_libraries(video_stabilizer
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cublas
)

# Compiler-specific flags for RTX 4070 (sm_89)
target_compile_options(video_stabilizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler=-Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3
    >
)

# Install targets
install(TARGETS video_stabilizer
    RUNTIME DESTINATION bin
)

# Copy test data
install(DIRECTORY data/
    DESTINATION share/real_time_video_stabilization/data
)

# Print configuration summary
message(STATUS "======================================")
message(STATUS "Video Stabilization Build Configuration")
message(STATUS "======================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES} (RTX 4070)")
message(STATUS "======================================")
