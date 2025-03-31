

#include <iostream>

#include "config.h"
#include "cuda_compute/cuda_gl_interop_manager.h"
#include "cuda_compute/plate_tectonic_sim.h"
#include "renderer/background.h"
#include "renderer/renderer.h"
#include "renderer/terrain.h"

void loadShaders()
{
    Renderer::addShader("background", new Shader("./shaders/fullscreenQuad.vert", "./shaders/background.frag"));
    Renderer::addShader("heightMap", new Shader("./shaders/heightMap.vert", "./shaders/heightMap.frag"));
    Renderer::addShader("plates", new Shader("./shaders/heightMap.vert", "./shaders/plates.frag"));
}

void loadTextures(CudaGlInteropManager &interopManager, const PlateTectonicSim &tectonicSim)
{
    const glm::ivec2 heightmapDimensions = Config::getInstance().HEIGHTMAP_DIMENSIONS;

    auto *environmentMap = new Texture();
    environmentMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    environmentMap->load2DImage("./assets/textures/001.hdr", GL_RGB32F, GL_RGB, GL_FLOAT, 3);
    Renderer::addTexture("environmentMap", environmentMap);

    auto *heightMap = new Texture();
    heightMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    heightMap->load2DImage("./assets/textures/heightMap.png", GL_R32F, GL_RED, GL_FLOAT, 1);
    Renderer::addTexture("heightMap", heightMap);

    auto *cudaHeightMap = new Texture();
    cudaHeightMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    cudaHeightMap->load2DEmpty(GL_R32F, GL_RED, GL_FLOAT, heightmapDimensions);
    Renderer::addTexture("cudaHeightMap", cudaHeightMap);

    interopManager.addConnection(heightMap->id(), tectonicSim.heightMapDevice(), sizeof(float), heightmapDimensions.x, heightmapDimensions.y);

    auto *cudaPlateMap = new Texture();
    cudaPlateMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    cudaPlateMap->load2DEmpty(GL_R8, GL_RED, GL_UNSIGNED_BYTE, heightmapDimensions);
    Renderer::addTexture("cudaPlateMap", cudaPlateMap);
    interopManager.addConnection(cudaPlateMap->id(), static_cast<void *>(tectonicSim.plateIdsDevice()), sizeof(uint8_t), heightmapDimensions.x, heightmapDimensions.y);
    std::cout << "copying" << "\n";
    interopManager.copyAllConnections();


}

void initRenderComponents(Renderer &renderer)
{
    renderer.addRenderComponent(new Background());
    const auto size = Config::getInstance().HEIGHTMAP_DIMENSIONS;
        // For now fix it to the size of the heightmap
    renderer.addRenderComponent(new Terrain(size.x, size.y));
}

int main()
{
    Config::getInstance().loadConfig("config.json");

    if (Config::getInstance().USE_RENDERER)
    {
        Renderer renderer;
        renderer.initRenderer();
        auto const size = Config::getInstance().HEIGHTMAP_DIMENSIONS;
        PlateTectonicSim tectonicSim(size.x, size.y, 128);
        tectonicSim.initialize(rand());

        CudaGlInteropManager interopManager;

        loadShaders();
        loadTextures(interopManager, tectonicSim);

        initRenderComponents(renderer);

        renderer.initRenderComponents();
        while (!renderer.shouldClose())
        {
            renderer.render();
        }
    }
    else
        std::cout << "Do some non-GUI terrain generation" << std::endl;

}