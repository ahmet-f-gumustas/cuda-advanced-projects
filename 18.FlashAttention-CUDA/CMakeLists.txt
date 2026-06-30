cmake_minimum_required(VERSION 3.18)
project(flash_attention_cuda LANGUAGES CXX CUDA)

# ============================================================================
# Standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# Compiler flags
# ============================================================================
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG   "-G -g")

find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Static library: the kernels
# ============================================================================
add_library(flash_cuda STATIC src/flash_kernels.cu)
target_include_directories(flash_cuda PUBLIC include)
target_link_libraries(flash_cuda PUBLIC CUDA::cudart)
set_target_properties(flash_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Executables
# ============================================================================
add_executable(flash_demo src/main.cu)
target_link_libraries(flash_demo PRIVATE flash_cuda)
set_target_properties(flash_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(flash_benchmark src/benchmark.cu)
target_link_libraries(flash_benchmark PRIVATE flash_cuda)
set_target_properties(flash_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_flash tests/test_flash.cu)
target_link_libraries(test_flash PRIVATE flash_cuda)
set_target_properties(test_flash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
message(STATUS "")
message(STATUS "=== FlashAttention-CUDA Build ===")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Targets            : flash_cuda (lib), flash_demo, flash_benchmark, test_flash")
message(STATUS "")
