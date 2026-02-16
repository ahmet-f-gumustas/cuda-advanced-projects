cmake_minimum_required(VERSION 3.18)
project(LSTM_CUDA_Implementation CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Source files
set(CUDA_SOURCES
    src/lstm_kernels.cu
    src/lstm_layer.cu
)

# Create LSTM library
add_library(lstm_cuda STATIC ${CUDA_SOURCES})
target_include_directories(lstm_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(lstm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Training executable
add_executable(train src/train.cu)
target_link_libraries(train lstm_cuda ${CUDA_LIBRARIES})
set_target_properties(train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Testing executable
add_executable(test_lstm src/test.cu)
target_link_libraries(test_lstm lstm_cuda ${CUDA_LIBRARIES})
set_target_properties(test_lstm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Print configuration
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Installation
install(TARGETS train test_lstm lstm_cuda
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)
