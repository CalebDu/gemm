cmake_minimum_required(VERSION 3.10)
project(gemm CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)

#option(Xcudafe "--diag_suppress=unrecognized_gcc_pragma")

find_package(fmt REQUIRED)
include_directories(/home/caleb/cutlass/include)
include_directories(/home/caleb/cutlass/tools/util/include)
include_directories(include)
include_directories(/home/caleb/include/fmt/include)

add_executable(gemm mm_run.cu)
#add_executable(reduce reduce.cu)
#add_executable(shfl shfl_test.cu)

#target_link_libraries(reduce -lcublas fmt::fmt)
#set_target_properties(reduce PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gemm -lcublas fmt::fmt)
set_target_properties(gemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
