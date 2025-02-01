cmake_minimum_required(VERSION 3.10)
project(HartreeFock)

# Use C++11 standard.
set(CMAKE_CXX_STANDARD 11)

# Find and include Eigen3 (make sure it is installed on your system).
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Build the executable.
add_executable(hf main.cpp)
