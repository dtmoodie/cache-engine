cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11")

FILE(GLOB_RECURSE hdr "include/ce/*.hpp")
FILE(GLOB_RECURSE src "include/ce/*.cpp")

ADD_LIBRARY(ce SHARED ${hdr} ${src})
TARGET_INCLUDE_DIRECTORIES(ce PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

add_executable(main main.cpp)
TARGET_LINK_LIBRARIES(main ce)