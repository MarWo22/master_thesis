

#include <iostream>

#include "config.h"
#include "cuda_compute/plate_tectonic_sim.h"
#include "renderer/background.h"
#include "renderer/renderer.h"
#include "renderer/terrain.h"

void loadShaders()
{
    Renderer::addShader("background", new Shader("./shaders/fullscreenQuad.vert", "./shaders/background.frag"));
    Renderer::addShader("heightMap", new Shader("./shaders/heightMap.vert", "./shaders/heightMap.frag"));
}

void loadTextures(PlateTectonicSim &tectonicsSim)
{
    auto *environmentMap = new Texture;
    environmentMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    environmentMap->load2DImage("./assets/textures/001.hdr", GL_RGB32F, GL_RGB, GL_FLOAT, 3);
    Renderer::addTexture("environmentMap", environmentMap);

    auto *heightMap = new Texture;
    heightMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    heightMap->load2DImage("./assets/textures/heightMap.png", GL_R32F, GL_RED, GL_FLOAT, 1);
    //Renderer::addTexture("heightMap", heightMap);

    auto *cudaHeightMap = new Texture;
    cudaHeightMap->init2D(GL_CLAMP_TO_EDGE, GL_LINEAR);
    cudaHeightMap->load2DEmpty(GL_R32F, GL_RED, GL_FLOAT, Config::getInstance().HEIGHTMAP_DIMENSIONS);
    tectonicsSim.connectToOpengl2DTexture(cudaHeightMap->id());
    tectonicsSim.copyToOpenGl();

    Renderer::addTexture("heightMap", cudaHeightMap);
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
        PlateTectonicSim tectonicSim(size.x, size.y);
        loadShaders();
        loadTextures(tectonicSim);

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