cmake_minimum_required(VERSION 3.14)

project(NimbusRT_C LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(Dependencies)
add_subdirectory(_C)
add_subdirectory(Nimbus)