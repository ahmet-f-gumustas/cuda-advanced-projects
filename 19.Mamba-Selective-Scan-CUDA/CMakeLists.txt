cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

project(mamba_selective_scan_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

find_package(CUDAToolkit REQUIRED)

add_library(mamba_cuda STATIC
    src/selective_scan.cu
    src/mamba_block.cu)
target_include_directories(mamba_cuda PUBLIC include)
target_link_libraries(mamba_cuda PUBLIC CUDA::cudart)
set_target_properties(mamba_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(mamba_demo src/main.cu)
target_link_libraries(mamba_demo PRIVATE mamba_cuda)
set_target_properties(mamba_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mamba_benchmark src/benchmark.cu)
target_link_libraries(mamba_benchmark PRIVATE mamba_cuda)
set_target_properties(mamba_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_mamba tests/test_mamba.cu)
target_link_libraries(test_mamba PRIVATE mamba_cuda)
set_target_properties(test_mamba PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME mamba_correctness COMMAND test_mamba)
set_tests_properties(mamba_correctness PROPERTIES SKIP_RETURN_CODE 77)

message(STATUS "")
message(STATUS "=== Mamba-Selective-Scan-CUDA Build ===")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Targets            : mamba_demo, mamba_benchmark, test_mamba")
message(STATUS "")
