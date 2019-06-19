cmake_minimum_required(VERSION 2.8)
project(cache-engine)
set(CMAKE_DEBUG_POSTFIX "d")

FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(OpenCV QUIET COMPONENTS core cudaimgproc cudaarithm cudawarping imgcodecs)

FILE(GLOB_RECURSE hdr "include/ce/*.hpp")
FILE(GLOB_RECURSE src "include/ce/*.cpp")

ADD_LIBRARY(ce SHARED ${hdr} ${src})
TARGET_INCLUDE_DIRECTORIES(ce
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/external/ct/include>
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

IF(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)
    TARGET_LINK_LIBRARIES(ce ${CUDA_CUDART_LIBRARY})
    TARGET_COMPILE_DEFINITIONS(ce PUBLIC HAVE_CUDA)
ENDIF()

IF(OpenCV_FOUND)
    TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>)
    TARGET_COMPILE_DEFINITIONS(ce PUBLIC HAVE_OPENCV)
    TARGET_LINK_LIBRARIES(ce ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)

add_executable(example_ce_main main.cpp)
TARGET_LINK_LIBRARIES(example_ce_main ce)

IF(CUDA_FOUND)
    file(GLOB_RECURSE src "tests/cuda/*.cpp" "tests/cuda/*.cu")
    cuda_add_executable(test_ce_cuda_sync ${src})
    TARGET_LINK_LIBRARIES(test_ce_cuda_sync ce)
ENDIF()

if(OpenCV_FOUND)
    add_executable(test_ce_opencv_sync "tests/opencv/main.cpp")
    TARGET_LINK_LIBRARIES(test_ce_opencv_sync ce)
endif()

find_package(Boost QUIET COMPONENTS unit_test_framework)
if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
    file(GLOB_RECURSE src "tests/static/*.cpp")
    link_directories(${BOOST_LIBRARYDIR})
    add_executable(test_ce_static_function ${src})
    target_link_libraries(test_ce_static_function ce ${Boost_LIBRARIES})
    target_include_directories(test_ce_static_function PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)

    file(GLOB_RECURSE src "tests/member/*.cpp")
    link_directories(${BOOST_LIBRARYDIR})
    add_executable(test_ce_member_function ${src})
    target_link_libraries(test_ce_member_function ce ${Boost_LIBRARIES})
    target_include_directories(test_ce_member_function PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
ENDIF()

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY external/ct/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

