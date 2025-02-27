cmake_minimum_required(VERSION 3.20)
project(PlateTectonics CUDA CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(PlateTectonics
        main.cpp
        random_texture.h
        random_texture.cu
)

set_target_properties(PlateTectonics PROPERTIES
                        LINKER_LANGUAGE CXX
                        CUDA_SEPARABLE_COMPILATION ON
)