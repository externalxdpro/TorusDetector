set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
cmake_minimum_required(VERSION 2.8)
project( TorusDetector )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( TorusDetector TorusDetector.cpp )
target_link_libraries( TorusDetector ${OpenCV_LIBS} )
find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})
target_link_libraries(TorusDetector fmt::fmt)
