cmake_minimum_required(VERSION 3.23)
project(efficientnet-cpp LANGUAGES CXX CUDA)

# C++ ve CUDA standartları
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Varsayılan build tipi
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA mimarileri
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
    else()
        set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070
    endif()
endif()

# Backend seçimi
set(BACKEND "ORT" CACHE STRING "Backend seçimi: ORT, TRT, CUDNN")
set_property(CACHE BACKEND PROPERTY STRINGS ORT TRT CUDNN)

# Opsiyonlar
option(ENABLE_NVJPEG "NVJPEG desteğini etkinleştir" ON)

# Optimizasyon bayrakları
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
endif()

# Module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# CUDA
find_package(CUDAToolkit REQUIRED)

# cuDNN
find_package(CUDNN REQUIRED)

# NVJPEG
if(ENABLE_NVJPEG)
    if(TARGET CUDA::nvjpeg)
        set(HAVE_NVJPEG ON)
        message(STATUS "NVJPEG bulundu")
    else()
        message(WARNING "NVJPEG bulunamadı, CPU fallback kullanılacak")
        set(HAVE_NVJPEG OFF)
    endif()
endif()

# Include dizinleri
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

# Ortak kütüphane
add_library(efficientnet_common STATIC
    src/common/cuda_utils.cpp
    src/common/timer.cpp
    src/common/logger.cpp
    src/io/image_decoder.cpp
    src/io/nvjpeg_decoder.cpp
    src/preprocess/preprocess.cpp
    src/preprocess/resize_norm.cu
    src/postprocess/postprocess.cpp
    src/postprocess/softmax_topk.cu
)

target_link_libraries(efficientnet_common
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARIES}
)

if(HAVE_NVJPEG)
    target_link_libraries(efficientnet_common PUBLIC CUDA::nvjpeg)
    target_compile_definitions(efficientnet_common PUBLIC HAVE_NVJPEG)
endif()

# Backend'e göre derleme
if(BACKEND STREQUAL "ORT")
    find_package(ONNXRuntime REQUIRED)
    
    add_library(efficientnet_ort_backend STATIC
        src/backends/ort/ort_backend.cpp
    )
    
    target_link_libraries(efficientnet_ort_backend
        PUBLIC
            efficientnet_common
            ${ONNXRUNTIME_LIBRARIES}
    )
    
    target_include_directories(efficientnet_ort_backend
        PUBLIC
            ${ONNXRUNTIME_INCLUDE_DIRS}
    )
    
    add_executable(efficientnet_ort apps/classify_ort.cpp)
    target_link_libraries(efficientnet_ort efficientnet_ort_backend)
    
elseif(BACKEND STREQUAL "TRT")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        find_package(TensorRT)
        if(TensorRT_FOUND)
            add_library(efficientnet_trt_backend STATIC
                src/backends/tensorrt/trt_backend.cpp
                src/backends/tensorrt/int8_calibrator.cpp
            )
            
            target_link_libraries(efficientnet_trt_backend
                PUBLIC
                    efficientnet_common
                    ${TensorRT_LIBRARIES}
            )
            
            target_include_directories(efficientnet_trt_backend
                PUBLIC
                    ${TensorRT_INCLUDE_DIRS}
            )
            
            add_executable(efficientnet_trt apps/classify_trt.cpp)
            target_link_libraries(efficientnet_trt efficientnet_trt_backend)
        else()
            message(WARNING "TensorRT bulunamadı, TRT backend devre dışı")
        endif()
    else()
        message(STATUS "TensorRT backend sadece aarch64'de desteklenir")
    endif()
    
elseif(BACKEND STREQUAL "CUDNN")
    add_library(efficientnet_cudnn_backend STATIC
        src/backends/cudnn/cudnn_kernels.cpp
    )
    
    target_link_libraries(efficientnet_cudnn_backend
        PUBLIC
            efficientnet_common
    )
    
    add_executable(bench_cudnn apps/bench_cudnn.cpp)
    target_link_libraries(bench_cudnn efficientnet_cudnn_backend)
endif()