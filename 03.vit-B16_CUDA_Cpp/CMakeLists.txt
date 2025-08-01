cmake_minimum_required(VERSION 3.18)
project(vit_cuda_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA architecture for RTX 4070
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Source files
set(CUDA_SOURCES
    src/tensor.cu
    src/ops_layernorm.cu
    src/ops_softmax.cu
    src/ops_gelu.cu
    src/ops_fuse.cu
    src/ops_patch_embed.cu
    src/ops_position_embed.cu
    src/vit_block.cu
    src/vit_model.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/weights.cpp
    src/timers.cpp
    src/cublaslt_utils.cpp
)

# Main executable
add_executable(vit_infer ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(vit_infer 
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cublasLt_LIBRARY}
    ${OpenCV_LIBS}
    cudnn
)

# Tools
add_executable(gen_dummy_weights tools/gen_dummy_weights.cpp src/weights.cpp src/tensor.cu)
target_link_libraries(gen_dummy_weights ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Set RPATH
set_target_properties(vit_infer PROPERTIES
    BUILD_RPATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)