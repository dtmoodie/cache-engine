cmake_minimum_required(VERSION 3.5)
CMAKE_POLICY(SET CMP0048 NEW)
project(cache-engine 
    VERSION 1.0.0
    LANGUAGES CXX
)

option(BUILD_TESTS "Build tests" ON)

set(CMAKE_DEBUG_POSTFIX "d")

FILE(GLOB_RECURSE hdr "include/ce/*.hpp")
FILE(GLOB_RECURSE src "include/ce/*.cpp")

ADD_LIBRARY(ce SHARED ${hdr} ${src})
TARGET_INCLUDE_DIRECTORIES(ce
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(ce
    PROPERTIES
        CXX_VERSION 11
        CXX_VERSION_REQUIRED ON
)

if(NOT TARGET ct)
    add_subdirectory(external/ct)
endif()

target_link_libraries(ce ct)

FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    if(CUDA_TOOLKIT_INCLUDE)
    #    TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${CUDA_TOOLKIT_INCLUDE}>)
    endif()
    TARGET_LINK_LIBRARIES(ce ${CUDA_CUDART_LIBRARY})
    TARGET_COMPILE_DEFINITIONS(ce PUBLIC HAVE_CUDA)
ENDIF()

FIND_PACKAGE(OpenCV QUIET COMPONENTS core cudaimgproc cudaarithm cudawarping imgcodecs)
IF(OpenCV_FOUND)
    TARGET_COMPILE_DEFINITIONS(ce PUBLIC HAVE_OPENCV)
    TARGET_LINK_LIBRARIES(ce ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)

add_executable(example_ce_main main.cpp)
TARGET_LINK_LIBRARIES(example_ce_main ce)

if(BUILD_TESTS)
    enable_testing()
    IF(CUDA_FOUND)
        file(GLOB_RECURSE src "tests/cuda/*.cpp" "tests/cuda/*.cu")
        cuda_add_executable(test_ce_cuda_sync ${src})
        TARGET_LINK_LIBRARIES(test_ce_cuda_sync ce)
        add_test(test_ce_cuda_sync test_ce_cuda_sync)
    ENDIF()

    if(OpenCV_FOUND)
        add_executable(test_ce_opencv_sync "tests/opencv/main.cpp")
        TARGET_LINK_LIBRARIES(test_ce_opencv_sync ce)
        add_test(test_ce_opencv_sync test_ce_opencv_sync)
    endif()

    file(GLOB_RECURSE src "tests/static/*.cpp")
    add_executable(test_ce_static_function ${src})
    target_link_libraries(test_ce_static_function ce gtest gtest_main)
    add_test(test_ce_static_function test_ce_static_function)

    file(GLOB_RECURSE src "tests/member/*.cpp")
    add_executable(test_ce_member_function ${src})
    target_link_libraries(test_ce_member_function ce gtest gtest_main)
    add_test(test_ce_member_function test_ce_member_function)

    file(GLOB_RECURSE src "tests/shared_ptr/*.cpp")
    add_executable(test_ce_shared_ptr ${src})
    target_link_libraries(test_ce_shared_ptr ce gtest gtest_main)
    add_test(test_ce_shared_ptr test_ce_shared_ptr)
ENDIF()

ADD_EXECUTABLE(constexpr_test "tests/constexpr/main.cpp")
target_link_libraries(constexpr_test ce)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

export(TARGETS ce
    FILE ce-targets.cmake
)

install(TARGETS ce
      EXPORT ce
      DESTINATION lib
)

install(EXPORT ce FILE ce-config.cmake
    DESTINATION share/ce
)

install(DIRECTORY include/ce DESTINATION include)

