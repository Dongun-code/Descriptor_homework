cmake_minimum_required(VERSION 3.0)
project(cvfeature)

set(CMAKE_CXX_COMPILER g++)

find_package(OpenCV REQUIRED PATHS $ENV{HOME}/lib/deploy/opencv/lib/cmake NO_DEFAULT_PATH)
message("OpenCV_INCLUDE_DIR: " ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
