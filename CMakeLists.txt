cmake_minimum_required(VERSION 3.20)
project(MyCUDAProject LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories("$ENV{CUDA_PATH}/include")

# Find the package (if stb installed via vcpkg)
find_package(Stb REQUIRED)

add_executable(PlateTectonics
        main.cpp
        random_texture.h
        random_texture.cu
        erosion.h
        erosion.cu)

set_target_properties(PlateTectonics PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(PlateTectonics PRIVATE ${Stb_INCLUDE_DIR})
