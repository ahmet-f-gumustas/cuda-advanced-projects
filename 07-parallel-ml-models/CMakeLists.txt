cmake_minimum_required(VERSION 3.18)
project(ParallelMLModels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
)

# Source files
set(CUDA_SOURCES
    src/linear_regression.cu
    src/kmeans.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/model_manager.cpp
    src/visualizer.cpp
)

# CUDA compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -O3
    -arch=sm_75
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    --expt-relaxed-constexpr
)

# Create executable
add_executable(parallel_ml_models
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Set CUDA architectures
set_target_properties(parallel_ml_models PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89"
)

# Link libraries
target_link_libraries(parallel_ml_models
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    pthread
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(parallel_ml_models PRIVATE
        -Wall
        -Wextra
        -O3
    )
endif()

# Print configuration
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "OpenGL Libraries: ${OPENGL_LIBRARIES}")
message(STATUS "GLEW Libraries: ${GLEW_LIBRARIES}")
