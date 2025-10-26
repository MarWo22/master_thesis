#include <iostream>

#include "config.h"
#include "generation_settings.h"
#include "cuda_compute/cuda_gl_interop_manager.h"
#include "cuda_compute/kernel_settings.cuh"
#include "cuda_compute/plate_tectonic_sim.h"
#include "cuda_compute/debug_logging/debug_logging.cuh"
#include "renderer/background.h"
#include "renderer/renderer.h"
#include "renderer/terrain.h"

extern RenderSettings renderSettings;
extern SimulationSettings simulationSettings;

void loadShaders()
{
    Renderer::addShader("background", new Shader("./shaders/fullscreenQuad.vert", "./shaders/background.frag"));
    Renderer::addShader("heightMap", new Shader("./shaders/heightMap.vert", "./shaders/heightMap.frag"));
    Renderer::addShader("platesColored", new Shader("./shaders/heightMap.vert", "./shaders/platesColored.frag"));
    Renderer::addShader("platesBorder", new Shader("./shaders/heightMap.vert", "./shaders/platesBorder.frag"));
    Renderer::addShader("region", new Shader("./shaders/heightMap.vert", "./shaders/region.frag"));
    Renderer::addShader("velocity", new Shader("./shaders/heightMap.vert", "./shaders/velocity.frag"));
    Renderer::addShader("direction", new Shader("./shaders/heightMap.vert", "./shaders/direction.frag"));
    Renderer::addShader("asthenosphere", new Shader("./shaders/heightMap.vert", "./shaders/asthenosphere.frag"));
}

void loadTextures(CudaGlInteropManager &interopManager)
{
    const glm::ivec2 heightmapDimensions = Config::getInstance().HEIGHTMAP_DIMENSIONS;

    auto *environmentMap = new Texture();
    environmentMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    environmentMap->load2DImage("./assets/textures/001.hdr", GL_RGB32F, GL_RGB, GL_FLOAT, 3);
    Renderer::addTexture("environmentMap", environmentMap);

    auto *arrowTexture = new Texture();
    arrowTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    arrowTexture->load2DImage("./assets/textures/arrow_texture.png", GL_RGBA32F, GL_RGBA, GL_FLOAT, 4);
    Renderer::addTexture("arrowTexture", arrowTexture);

    auto *cudaHeightMap = new Texture();
    cudaHeightMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    cudaHeightMap->load2DEmpty(GL_R32F, GL_RED, GL_FLOAT, heightmapDimensions);
    Renderer::addTexture("heightMap", cudaHeightMap);

    // interopManager.addConnection(cudaHeightMap->id(), tectonicSim.heightMapDevice(), sizeof(float), heightmapDimensions.x, heightmapDimensions.y);
    interopManager.addConnection("heightMap", cudaHeightMap->id(), sizeof(float), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R32F, GL_RED, GL_FLOAT);

    // The following textures are not initially allocated. They are allocated through toggling them on or off on the GUI
    // These are only for visualization purposes, and will draw unnecessary memory and computing power to allocate and copy constantly when not used
    auto *cudaPlateTexture = new Texture();
    cudaPlateTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    Renderer::addTexture("cudaPlateTexture", cudaPlateTexture);
    interopManager.addConnection("cudaPlateTexture", cudaPlateTexture->id(), sizeof(uint8_t), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    Renderer::addTexture("cudaPlateTexture", cudaPlateTexture);

    auto *collisionMap = new Texture();
    collisionMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    Renderer::addTexture("collisionMap", collisionMap);
    interopManager.addConnection("collisionMap", collisionMap->id(), sizeof(uint32_t), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
    Renderer::addTexture("collisionMap", collisionMap);

    auto *accretionTexture = new Texture();
    accretionTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    Renderer::addTexture("accretionTexture", accretionTexture);
    interopManager.addConnection("accretionTexture", accretionTexture->id(), sizeof(uint8_t), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    Renderer::addTexture("accretionTexture", accretionTexture);



    auto *velocityTexture = new Texture();
    velocityTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    Renderer::addTexture("velocityTexture", velocityTexture);
    interopManager.addConnection("velocityTexture", velocityTexture->id(), sizeof(float), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R32F, GL_RED, GL_FLOAT);
    Renderer::addTexture("velocityTexture", velocityTexture);

    auto *asthenosphereTexture = new Texture();
    asthenosphereTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    Renderer::addTexture("asthenosphereTexture", asthenosphereTexture);
    interopManager.addConnection("asthenosphereTexture", asthenosphereTexture->id(), sizeof(float2), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_RG32F, GL_RG, GL_FLOAT);
    Renderer::addTexture("asthenosphereTexture", asthenosphereTexture);

    auto *waterTexture = new Texture();
    waterTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    interopManager.addConnection("waterTexture", waterTexture->id(), sizeof(float), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R32F, GL_RED, GL_FLOAT);
    Renderer::addTexture("waterTexture", waterTexture);

    auto *directionTexture = new Texture();
    directionTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    interopManager.addConnection("directionTexture", directionTexture->id(), sizeof(float2), heightmapDimensions.x / 16,
                                 heightmapDimensions.y / 16, GL_RG32F, GL_RG, GL_FLOAT);
    Renderer::addTexture("directionTexture", directionTexture);

    auto *cclTexture = new Texture();
    cclTexture->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    interopManager.addConnection("cclTexture", cclTexture->id(), sizeof(uint8_t), heightmapDimensions.x,
                                 heightmapDimensions.y, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    Renderer::addTexture("cclTexture", cclTexture);
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
        CudaGlInteropManager interopManager;

        loadShaders();
        loadTextures(interopManager);

        auto const size = Config::getInstance().HEIGHTMAP_DIMENSIONS;
        PlateTectonicSim tectonicSim(size.x, size.y, simulationSettings.seed, &interopManager);
        tectonicSim.initialize(simulationSettings.numStartingPlates, simulationSettings.numVoronoiSeeds);

        initRenderComponents(renderer);

        renderer.initRenderComponents();

        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess)
            std::cerr << "Cuda error during initialization phase: " << cudaGetErrorString(err) << "\n";

        float time_between_executions = 0;
        while (!renderer.shouldClose())
        {
            if (renderSettings.isExecutingRealtime)
            {
                time_between_executions += renderer.deltaTime();
                if (time_between_executions > 1.f / static_cast<float>(renderSettings.iterationsPerSecond))
                {
                    time_between_executions = 0;
                    renderSettings.callCallback("executeIterations");
                } else
                    std::cout << "waiting\n";
            }
            renderer.render();
        }
    } else
        std::cout << "Do some non-GUI terrain generation" << std::endl;
}
