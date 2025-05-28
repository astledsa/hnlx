cmake_minimum_required(VERSION 3.16)
project(HNLX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MLX_ROOT "${CMAKE_SOURCE_DIR}/../../mlx")

include_directories(${MLX_ROOT})

link_directories(${MLX_ROOT}/build)

add_library(hnlx STATIC hnlx.cpp)

add_executable(exec main.cpp)

target_link_libraries(exec
    PRIVATE
        hnlx
        mlx
        "-framework Accelerate"
        "-framework Metal"
        "-framework Foundation"
)
