cmake_minimum_required(VERSION 3.20)
project(PlateTectonics LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories("$ENV{CUDA_PATH}/include")

find_package(PNG REQUIRED)

add_executable(PlateTectonics
        main.cpp
        flux_erosion.h
        flux_erosion.cu
        texture_save.cpp
        texture_save.h
)


set_target_properties(PlateTectonics PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(PlateTectonics PRIVATE PNG::PNG)