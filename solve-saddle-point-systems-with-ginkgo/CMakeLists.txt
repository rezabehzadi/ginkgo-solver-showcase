cmake_minimum_required(VERSION 3.12)
project(saddle_point_solver)

set(CMAKE_CXX_STANDARD 17)

find_package(Ginkgo REQUIRED PATHS /path/to/ginkgo/build/cmake)

add_executable(saddle_solver src/main.cpp)
target_include_directories(saddle_solver PRIVATE include)
target_link_libraries(saddle_solver Ginkgo::ginkgo)