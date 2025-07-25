cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(real_time_image_denoising LANGUAGES CXX CUDA)

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
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

# Source files
set(CUDA_SOURCES
    src/cuda/bilateral_filter.cu
    src/cuda/nlm_denoiser.cu
    src/cuda/gaussian_filter.cu
)

set(CPP_SOURCES
    src/cpp/cuda_denoiser.cpp
    src/cpp/memory_manager.cpp
    src/cpp/image_processor.cpp
    src/main.cpp
)

# Create the executable
add_executable(image_denoiser ${CPP_SOURCES} ${CUDA_SOURCES})

# Set CUDA properties
set_property(TARGET image_denoiser PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET image_denoiser PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
target_link_libraries(image_denoiser 
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${Python3_LIBRARIES}
    cudart
    cublas
    cufft
)

# Compiler-specific flags
target_compile_options(image_denoiser PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v
        -Xcompiler -Wall
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_75,code=sm_75
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3
    >
)

# Create Python module (optional)
add_subdirectory(python)

# Install targets
install(TARGETS image_denoiser
    RUNTIME DESTINATION bin
)

# Copy Python scripts
install(DIRECTORY python/
    DESTINATION share/real_time_image_denoising/python
    FILES_MATCHING PATTERN "*.py"
)

# Copy test data
install(DIRECTORY data/
    DESTINATION share/real_time_image_denoising/data
)

# Print configuration summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
