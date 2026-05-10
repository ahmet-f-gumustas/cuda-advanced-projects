cmake_minimum_required(VERSION 3.18)
project(yolov8_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# cuDNN (system-installed v9, Debian/Ubuntu layout)
find_library(CUDNN_LIB cudnn HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
if(NOT CUDNN_LIB)
    message(FATAL_ERROR "cuDNN library not found")
endif()
find_path(CUDNN_INCLUDE cudnn.h HINTS /usr/include /usr/local/cuda/include)

# Optional: LibTorch (used by yolo_camera). Picked up from a Python torch install
# unless overridden by -DTorch_DIR=...
option(BUILD_CAMERA "Build the LibTorch+OpenCV webcam demo (yolo_camera)" ON)
if(BUILD_CAMERA)
    if(NOT Torch_DIR)
        execute_process(
            COMMAND python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake/Torch'))"
            OUTPUT_VARIABLE _TORCH_CMAKE OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _TORCH_RC)
        if(_TORCH_RC EQUAL 0 AND EXISTS "${_TORCH_CMAKE}")
            set(Torch_DIR "${_TORCH_CMAKE}")
            message(STATUS "Discovered Torch_DIR=${Torch_DIR}")
        endif()
    endif()
    find_package(Torch QUIET)
    find_package(OpenCV QUIET COMPONENTS core imgproc highgui videoio)
    if(NOT Torch_FOUND OR NOT OpenCV_FOUND)
        message(WARNING "yolo_camera disabled — Torch_FOUND=${Torch_FOUND} OpenCV_FOUND=${OpenCV_FOUND}")
        set(BUILD_CAMERA OFF)
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

set(LIB_SOURCES
    src/yolo_kernels.cu
    src/conv2d.cu
    src/backbone.cu
    src/neck.cu
    src/head.cu
    src/yolov8.cu
    src/postprocess.cu
    src/pipeline.cu
    src/image_io.cpp
)

add_library(yolov8_lib STATIC ${LIB_SOURCES})
target_include_directories(yolov8_lib PUBLIC include ${CUDNN_INCLUDE})
target_link_libraries(yolov8_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    ${CUDNN_LIB}
)
set_target_properties(yolov8_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Test executable
add_executable(test_yolo tests/test_yolo.cu)
target_link_libraries(test_yolo PRIVATE yolov8_lib)
set_target_properties(test_yolo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Main demo executable
add_executable(yolo_detect src/main.cu)
target_link_libraries(yolo_detect PRIVATE yolov8_lib)
set_target_properties(yolo_detect PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Benchmark executable
add_executable(yolo_benchmark src/benchmark.cu)
target_link_libraries(yolo_benchmark PRIVATE yolov8_lib)
set_target_properties(yolo_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Live webcam demo (LibTorch + OpenCV + custom CUDA pre/post)
if(BUILD_CAMERA)
    add_executable(yolo_camera src/yolo_camera.cpp src/ts_pipeline.cpp)
    target_link_libraries(yolo_camera PRIVATE
        yolov8_lib
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
    )
    target_include_directories(yolo_camera PRIVATE ${OpenCV_INCLUDE_DIRS})
    set_target_properties(yolo_camera PROPERTIES
        CXX_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
