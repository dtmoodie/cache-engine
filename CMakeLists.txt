cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
INCLUDE_DIRECTORIES(include)
add_executable(main main.cpp)
