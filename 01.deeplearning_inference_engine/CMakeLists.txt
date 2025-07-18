cmake_minimum_required(VERSION 3.18)
project(DeepInferenceEngine CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_TENSORRT "Use TensorRT for optimization" OFF)
option(ENABLE_PROFILING "Enable profiling support" ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

# Optional packages
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_definitions(-DUSE_OPENCV)
endif()

if(USE_TENSORRT)
    find_package(TensorRT)
    if(TensorRT_FOUND)
        add_definitions(-DUSE_TENSORRT)
    endif()
endif()

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp" "src/core/*.cu")
file(GLOB_RECURSE LAYER_SOURCES "src/layers/*.cpp" "src/layers/*.cu")
file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")
file(GLOB_RECURSE OPT_SOURCES "src/optimizations/*.cpp" "src/optimizations/*.cu")
file(GLOB_RECURSE UTIL_SOURCES "src/utils/*.cpp" "src/utils/*.cu")

# Create main library
add_library(deep_engine SHARED
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${KERNEL_SOURCES}
    ${OPT_SOURCES}
    ${UTIL_SOURCES}
)

target_link_libraries(deep_engine
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cudnn_LIBRARY}
    Threads::Threads
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(deep_engine OpenMP::OpenMP_CXX)
endif()

# Install targets
install(TARGETS deep_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/deep_engine
)

# Build subdirectories
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DeepEngineConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)