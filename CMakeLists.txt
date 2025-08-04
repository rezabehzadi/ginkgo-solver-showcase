cmake_minimum_required(VERSION 3.12)
project(HelloGinkgo)

find_package(Ginkgo REQUIRED PATHS "/mnt/c/ginkgo/ginkgo/build")

# add_executable(01_hello_ginkgo 01_hello_ginkgo.cpp)
# target_link_libraries(01_hello_ginkgo Ginkgo::ginkgo)

add_executable(02_dense_random_solver 02_dense_random_solver.cpp)
target_link_libraries(02_dense_random_solver Ginkgo::ginkgo)
