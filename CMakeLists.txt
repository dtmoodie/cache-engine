cmake_minimum_required(VERSION 2.8)
set(CMAKE_DEBUG_POSTFIX "d")
if(NOT MSVC)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
endif()

FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(OpenCV QUIET COMPONENTS core cudaimgproc cudaarithm cudawarping imgcodecs)

FILE(GLOB_RECURSE hdr "include/ce/*.hpp")
FILE(GLOB_RECURSE src "include/ce/*.cpp")

ADD_LIBRARY(ce SHARED ${hdr} ${src})
TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/external/ct/include>)

file(GLOB_RECURSE ct_hdrs ${CMAKE_CURRENT_LIST_DIR}/external/ct/include/*.hpp)
add_custom_target(ct SOURCES ${ct_hdrs})

IF(CUDA_FOUND)
  TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)
  TARGET_LINK_LIBRARIES(ce ${CUDA_CUDART_LIBRARY})
  TARGET_COMPILE_DEFINITIONS(ce PUBLIC HAVE_CUDA)
ENDIF()
IF(OpenCV_FOUND)
    TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>)
    TARGET_COMPILE_DEFINITIONS(ce PUBLIC HAVE_OPENCV)
    TARGET_LINK_LIBRARIES(ce ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)

add_executable(main main.cpp)
TARGET_LINK_LIBRARIES(main ce)

IF(CUDA_FOUND)
  file(GLOB_RECURSE src "tests/cuda/*.cpp" "tests/cuda/*.cu")
  cuda_add_executable(cuda_sync_test ${src})
  TARGET_LINK_LIBRARIES(cuda_sync_test ce)
ENDIF()

if(OpenCV_FOUND)
	add_executable(opencv_sync_test "tests/opencv/main.cpp")
	TARGET_LINK_LIBRARIES(opencv_sync_test ce)
endif()

find_package(Boost QUIET COMPONENTS unit_test_framework)
if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
	file(GLOB_RECURSE src "tests/static/*.cpp")
	link_directories(${BOOST_LIBRARYDIR})
	add_executable(static_function_test ${src})
        target_link_libraries(static_function_test ce ${Boost_LIBRARIES})
	target_include_directories(static_function_test PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
	
	file(GLOB_RECURSE src "tests/member/*.cpp")
	link_directories(${BOOST_LIBRARYDIR})
	add_executable(member_function_test ${src})
        target_link_libraries(member_function_test ce ${Boost_LIBRARIES})
	target_include_directories(member_function_test PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
ENDIF()

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY external/ct/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

