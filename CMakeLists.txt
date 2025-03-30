cmake_minimum_required(VERSION 3.20)
project(PlateTectonics LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories("$ENV{CUDA_PATH}/include")

find_package(PNG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(glm REQUIRED)

add_executable(PlateTectonics
        src/main.cpp
        src/renderer/background.cpp
        src/renderer/background.h
        src/renderer/camera.cpp
        src/renderer/camera.h
        src/renderer/gl_misc.cpp
        src/renderer/gl_misc.h
        src/renderer/gui.cpp
        src/renderer/gui.h
        src/renderer/render_component.cpp
        src/renderer/render_component.h
        src/renderer/renderer.cpp
        src/renderer/renderer.h
        src/renderer/shader.cpp
        src/renderer/shader.h
        src/renderer/terrain.cpp
        src/renderer/terrain.h
        src/renderer/textures/image.cpp
        src/renderer/textures/image.h
        src/renderer/textures/mipmap.cpp
        src/renderer/textures/mipmap.h
        src/renderer/textures/texture.cpp
        src/renderer/textures/texture.h
        src/renderer/imgui/imgui_impl_glfw.cpp
        src/renderer/imgui/imgui_impl_glfw.h
        src/renderer/imgui/imgui_impl_opengl3.cpp
        src/renderer/imgui/imgui_impl_opengl3.h
        src/renderer/imgui/imgui_impl_opengl3_loader.h
        src/cuda_compute/plate_tectonic_sim.cu
        src/cuda_compute/plate_tectonic_sim.h
        src/cuda_compute/random_texture.cu
        src/cuda_compute/random_texture.h
        src/cuda_compute/texture_save.cpp
        src/cuda_compute/texture_save.h
        src/cuda_compute/flux_erosion.h
        src/cuda_compute/flux_erosion.cu
        src/cuda_compute/cuda_noise.cuh
        src/config.h
        src/cuda_compute/vec2.cuh
        src/cuda_compute/plate_data.h
        src/cuda_compute/plate_tectonics_kernel.cu
        src/cuda_compute/plate_tectonics_kernel.cuh
        src/cuda_compute/cuda_helper.cuh
        src/cuda_compute/cuda_gl_interop_manager.cpp
        src/cuda_compute/cuda_gl_interop_manager.h
        src/cuda_compute/cuda_texture.cuh
        src/generation_settings.cpp
        src/generation_settings.h
        src/cuda_compute/erosion_kernel.cu
        src/cuda_compute/erosion_kernel.cuh
)

set_target_properties(PlateTectonics PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(PlateTectonics PRIVATE
        PNG::PNG
        GLEW::GLEW
        glfw
        imgui::imgui
        glm::glm
        OpenGL::GL
)

# Include directories (if needed)
target_include_directories(PlateTectonics PRIVATE
        ${STB_INCLUDE_DIRS}
)


file(GLOB SHADER_FILES "src/shaders/*")

add_custom_target(copy_shaders ALL COMMENT "Updating shaders if they changed")

foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(FILE_NAME ${SHADER_FILE} NAME)
    add_custom_command(
            TARGET copy_shaders PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_FILE} ${CMAKE_BINARY_DIR}/shaders/${FILE_NAME}
            COMMENT "Copying updated shader: ${FILE_NAME}"
    )
endforeach()

add_dependencies(PlateTectonics copy_shaders)

file(GLOB_RECURSE ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/*")

add_custom_target(copy_assets ALL COMMENT "Updating assets if they changed")

foreach(ASSET_FILE ${ASSET_FILES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/assets ${ASSET_FILE})
    add_custom_command(
            TARGET copy_assets PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ASSET_FILE} ${CMAKE_BINARY_DIR}/assets/${REL_PATH}
            COMMENT "Copying updated asset: ${REL_PATH}"
    )
endforeach()

add_dependencies(PlateTectonics copy_assets)


# Add custom target to move config.json if it changes
add_custom_target(copy_config_json ALL COMMENT "Updating config.json if it changed")

add_custom_command(
        TARGET copy_config_json PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/config.json ${CMAKE_BINARY_DIR}/config.json
        COMMENT "Copying updated config.json"
)

add_dependencies(PlateTectonics copy_config_json)
