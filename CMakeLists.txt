cmake_minimum_required(VERSION 3.10)
project(gemm CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)

include_directories(/home/caleb/cutlass/include)
include_directories(/home/caleb/cutlass/tools/util/include)
include_directories(include)
include_directories(/home/caleb/include/fmt/include)

add_executable(gemm mm_run.cu)

target_link_libraries(gemm -lcublas)
set_target_properties(gemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
