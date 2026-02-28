cmake_minimum_required(VERSION 3.18)
project(GPTTransformerInference LANGUAGES CXX CUDA)

# ============================================================================
# Standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA architectures (RTX 3000-4000 series + Ampere data-center)
# ============================================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# ============================================================================
# Compiler flags
# ============================================================================
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr -lineinfo")

# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Static library: all inference logic
# ============================================================================
add_library(gpt_cuda STATIC
    src/transformer_kernels.cu
    src/transformer.cu
    src/kv_cache.cu
    src/quantization.cu
    src/speculative.cu)

target_include_directories(gpt_cuda PUBLIC include)
target_link_libraries(gpt_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand)

set_target_properties(gpt_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Executable 1: inference (interactive text generation)
# ============================================================================
add_executable(inference src/inference.cu)
target_include_directories(inference PRIVATE include)
target_link_libraries(inference gpt_cuda CUDA::cudart)
set_target_properties(inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Executable 2: benchmark (tokens/sec comparison)
# ============================================================================
add_executable(benchmark src/benchmark.cu)
target_include_directories(benchmark PRIVATE include)
target_link_libraries(benchmark gpt_cuda CUDA::cudart)
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Tests (optional, built by default)
# ============================================================================
add_executable(test_attention tests/test_attention.cu)
target_include_directories(test_attention PRIVATE include)
target_link_libraries(test_attention gpt_cuda CUDA::cudart)
set_target_properties(test_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Build summary
# ============================================================================
message(STATUS "")
message(STATUS "=== GPT Transformer Inference Build ===")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Targets            : gpt_cuda (lib), inference, benchmark, test_attention")
message(STATUS "")
