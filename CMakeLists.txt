cmake_minimum_required(VERSION 3.12)
project(hello)


set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_LINK_WHAT_YOU_USE TRUE)
#find_program(TIDY "clang-tidy-6.0")
#set(CMAKE_CXX_CLANG_TIDY ${TIDY} "-checks=*")
#find_program(CPPCHECK cppcheck "--std=c++14")
#set(CMAKE_CXX_CPPCHECK ${CPPCHECK})
#find_program(IWYU include-what-you-use)
#set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE  ${IWYU})

find_package(OpenCV REQUIRED)
add_executable(hello main.cpp)
target_include_directories(hello PRIVATE ${OpenCV_INCLUDE_DIRS})
 target_link_libraries(hello PRIVATE opencv_imgproc opencv_core opencv_imgcodecs opencv_highgui)
#target_link_libraries(hello PRIVATE opencv_imgcodecs opencv_core)