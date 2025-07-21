# Benchmarks CMakeLists.txt

# Find Google Benchmark
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    # Download and build Google Benchmark
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# Helper function to add benchmarks
function(add_deep_engine_benchmark benchmark_name)
    add_executable(${benchmark_name} ${ARGN})
    target_link_libraries(${benchmark_name}
        deep_engine
        benchmark::benchmark
        benchmark::benchmark_main
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_cudnn_LIBRARY}
    )
    
    target_include_directories(${benchmark_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
    )
endfunction()

# Add benchmark executables
add_deep_engine_benchmark(benchmark_tensor benchmark_tensor.cpp)
add_deep_engine_benchmark(benchmark_layers benchmark_layers.cpp)
add_deep_engine_benchmark(benchmark_kernels benchmark_kernels.cpp)
add_deep_engine_benchmark(benchmark_models benchmark_models.cpp)

# Combined benchmark executable
add_deep_engine_benchmark(benchmark_engine
    benchmark_tensor.cpp
    benchmark_layers.cpp
    benchmark_kernels.cpp
    benchmark_models.cpp
)