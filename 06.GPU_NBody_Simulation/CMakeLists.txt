cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPU_NBody_Simulation LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package (require 12.6+)
find_package(CUDAToolkit 12.6 REQUIRED)

# Find OpenGL and GLFW for visualization
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# CUDA architecture - RTX 4070 = sm_89
# Set multiple architectures for compatibility
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cu
    src/nbody_kernels.cu
)

# Executable
add_executable(nbody_sim ${SOURCES})

# Link CUDA libraries
target_link_libraries(nbody_sim CUDA::cudart CUDA::cuda_driver)

# Set CUDA properties
set_target_properties(nbody_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install target
install(TARGETS nbody_sim DESTINATION bin)

# ============================================================================
# Visual N-Body Simulation with OpenGL
# ============================================================================

# Visual application sources
set(VISUAL_SOURCES
    src/main_visual.cu
    src/nbody_kernels.cu
    src/renderer.cpp
)

# Visual executable
add_executable(nbody_visual ${VISUAL_SOURCES})

# Link libraries for visual version
target_link_libraries(nbody_visual
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glfw
    ${CMAKE_DL_LIBS}
)

# Set CUDA properties
set_target_properties(nbody_visual PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install visual target
install(TARGETS nbody_visual DESTINATION bin)

# ============================================================================
# Python Bindings with pybind11
# ============================================================================

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Find pybind11
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Try to find pybind11 via Python
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_VERSION}")

        # Python module sources (need to be compiled separately without main.cu)
        set(PYTHON_BINDINGS_SOURCES
            python/nbody_bindings.cpp
            src/nbody_kernels.cu
            src/nbody_helper.cpp
        )

        # Create shared library for Python bindings
        pybind11_add_module(nbody_python MODULE ${PYTHON_BINDINGS_SOURCES})

        # Include directories
        target_include_directories(nbody_python PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        # Link CUDA libraries
        target_link_libraries(nbody_python PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
        )

        # Set properties
        set_target_properties(nbody_python PROPERTIES
            OUTPUT_NAME "nbody_cuda"
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
            POSITION_INDEPENDENT_CODE ON
            LINKER_LANGUAGE CXX
        )

        # Ensure CUDA runtime is linked properly
        set_property(TARGET nbody_python PROPERTY CUDA_RUNTIME_LIBRARY Static)

        # Install to Python site-packages
        install(TARGETS nbody_python
                LIBRARY DESTINATION ${Python3_SITEARCH})

        message(STATUS "Python bindings will be built")
        message(STATUS "Python executable: ${Python3_EXECUTABLE}")
        message(STATUS "Python version: ${Python3_VERSION}")
    else()
        message(WARNING "pybind11 not found. Python bindings will not be built.")
    endif()
endif()

# Print configuration
message(STATUS "==================================================")
message(STATUS "GPU N-Body Simulation Configuration")
message(STATUS "==================================================")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "==================================================")
