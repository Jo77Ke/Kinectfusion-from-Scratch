cmake_minimum_required(VERSION 3.22.1)
project(kinectfusion)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)

find_path(FREEIMAGE_INCLUDE_DIR NAMES FreeImage.h)
find_library(FREEIMAGE_LIBRARY NAMES freeimage)

add_executable(${PROJECT_NAME}
        "libs/Eigen.h"
        "libs/FreeImageHelper.cpp"
        "src/main.cpp"
        "src/VirtualSensor.h"
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${FREEIMAGE_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/libs
)

target_link_libraries(${PROJECT_NAME}
        Eigen3::Eigen
        ${FREEIMAGE_LIBRARY}
)

