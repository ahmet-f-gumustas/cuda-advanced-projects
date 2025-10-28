cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPU_NBody_Simulation LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package (require 12.6+)
find_package(CUDAToolkit 12.6 REQUIRED)

# Find OpenGL and GLFW for visualization
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

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

# ============================================================================
# Visual N-Body Simulation with OpenGL
# ============================================================================

# Visual application sources
set(VISUAL_SOURCES
    src/main_visual.cu
    src/nbody_kernels.cu
    src/renderer.cpp
)

# Visual executable
add_executable(nbody_visual ${VISUAL_SOURCES})

# Link libraries for visual version
target_link_libraries(nbody_visual
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glfw
    ${CMAKE_DL_LIBS}
)

# Set CUDA properties
set_target_properties(nbody_visual PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install visual target
install(TARGETS nbody_visual DESTINATION bin)

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
