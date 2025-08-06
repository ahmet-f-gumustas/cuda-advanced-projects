# FindONNXRuntime.cmake
# ONNXRUNTIME_ROOT ortam değişkeninden veya third_party dizininden ORT'yi bulur

if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
else()
    set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIRS
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include
          ${ONNXRUNTIME_ROOT}/include/onnxruntime
          ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARIES
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime DEFAULT_MSG
    ONNXRUNTIME_LIBRARIES ONNXRUNTIME_INCLUDE_DIRS)

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIRS ONNXRUNTIME_LIBRARIES)