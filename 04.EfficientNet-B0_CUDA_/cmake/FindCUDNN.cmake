# FindCUDNN.cmake
# Locate cuDNN library

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${CUDNN_ROOT}/include
          ${CUDA_TOOLKIT_ROOT_DIR}/include
          /usr/local/cuda/include
          /usr/include
)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    HINTS ${CUDNN_ROOT}/lib64
          ${CUDNN_ROOT}/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
          /usr/lib/x86_64-linux-gnu
          /usr/lib/aarch64-linux-gnu
          /usr/lib
)

# Get version
if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn_version.h CUDNN_VERSION_FILE_CONTENTS)
    if(NOT CUDNN_VERSION_FILE_CONTENTS)
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
    endif()
    
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
    
    if(NOT CUDNN_VERSION_MAJOR)
        set(CUDNN_VERSION "?")
    else()
        set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG
    CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

if(CUDNN_FOUND)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY} (version ${CUDNN_VERSION})")
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)