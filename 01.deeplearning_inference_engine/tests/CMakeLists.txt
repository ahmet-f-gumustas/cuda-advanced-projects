# Tests CMakeLists.txt

# Enable testing
enable_testing()

# Find GTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Download and build GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Helper function to add tests
function(add_deep_engine_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} 
        deep_engine 
        gtest 
        gtest_main
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_cudnn_LIBRARY}
    )
    
    # Add include directories
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
    )
    
    # Register test
    add_test(NAME ${test_name} COMMAND ${test_name})
    
    # Set test properties
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 120
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
    )
endfunction()

# Main test executable combining all unit tests
file(GLOB_RECURSE TEST_SOURCES "unit/*.cpp")
add_deep_engine_test(test_engine ${TEST_SOURCES})

# Individual test executables for faster iteration
add_deep_engine_test(test_tensor unit/test_tensor.cpp)
add_deep_engine_test(test_layers unit/test_layers.cpp)
add_deep_engine_test(test_graph unit/test_graph.cpp)
add_deep_engine_test(test_optimizations unit/test_optimizations.cpp)

# Benchmarks subdirectory
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()