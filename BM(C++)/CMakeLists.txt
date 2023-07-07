cmake_minimum_required(VERSION 3.17)
project(BMTest)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR C:\\opencv\\build) # xxxx目录包含OpenCVConfig.cmake
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} BM.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})