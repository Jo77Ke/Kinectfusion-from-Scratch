cmake_minimum_required(VERSION 3.22.1)
project(kinect_fusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
        "src/main.cpp"
        "src/bilateral_filter.h"
        "src/rgbd_frame_stream.h"
        "src/utils.h"
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

