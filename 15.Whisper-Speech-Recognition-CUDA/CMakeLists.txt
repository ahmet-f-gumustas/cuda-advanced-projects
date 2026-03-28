cmake_minimum_required(VERSION 3.18)
project(whisper_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(CUDAToolkit REQUIRED)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Source files
set(KERNEL_SOURCES
    src/whisper_kernels.cu
    src/mel_spectrogram.cu
    src/encoder.cu
    src/decoder.cu
    src/beam_search.cu
    src/pipeline.cu
    src/wav_reader.cpp
    src/tokenizer.cpp
)

# Static library
add_library(whisper_cuda_lib STATIC ${KERNEL_SOURCES})
target_include_directories(whisper_cuda_lib PUBLIC include)
target_link_libraries(whisper_cuda_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)
set_target_properties(whisper_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Test executable
add_executable(test_whisper src/test_whisper.cu)
target_link_libraries(test_whisper PRIVATE whisper_cuda_lib)
set_target_properties(test_whisper PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Main executable
add_executable(whisper_inference src/main.cu)
target_link_libraries(whisper_inference PRIVATE whisper_cuda_lib)
set_target_properties(whisper_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
