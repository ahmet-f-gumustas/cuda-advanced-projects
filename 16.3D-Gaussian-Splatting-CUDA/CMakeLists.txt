cmake_minimum_required(VERSION 3.18)
project(GaussianSplattingCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Source files for static library
set(LIB_SOURCES
    src/gaussian.cu
    src/splatting_kernels.cu
    src/renderer.cu
    src/ply_reader.cpp
)

# Static library
add_library(gs_cuda STATIC ${LIB_SOURCES})
target_include_directories(gs_cuda PUBLIC include)
target_link_libraries(gs_cuda PUBLIC
    CUDA::cudart
    CUDA::curand
)
set_target_properties(gs_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Main viewer executable (OpenGL)
add_executable(gs_viewer src/viewer.cu src/main.cu)
target_link_libraries(gs_viewer PRIVATE
    gs_cuda
    OpenGL::GL
    glfw
    ${CMAKE_DL_LIBS}
)
set_target_properties(gs_viewer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Benchmark executable
add_executable(gs_benchmark src/benchmark.cu)
target_link_libraries(gs_benchmark PRIVATE gs_cuda)
set_target_properties(gs_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Test executable
add_executable(gs_test tests/test_splatting.cu)
target_link_libraries(gs_test PRIVATE gs_cuda)
set_target_properties(gs_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
