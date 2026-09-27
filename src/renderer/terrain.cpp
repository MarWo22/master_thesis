#include "terrain.h"

#include <iostream>

#include "../generation_settings.h"
#include "../cuda_compute/kernel_settings.cuh"

extern RenderSettings renderSettings;
extern KernelSettings kernelSettingsHost;


void Terrain::Vertex::init(const int xPos, const int zPos, const int sizeX, const int sizeZ)
{
    pos = glm::vec3(xPos, 0, zPos);
    uv = glm::vec2(static_cast<float>(xPos) / static_cast<float>(sizeX),
                   static_cast<float>(zPos) / static_cast<float>(sizeZ));
}

Terrain::Terrain(const int sizeX, const int sizeZ)
    : m_heightmapShader(nullptr)
      , m_platesColoredShader(nullptr)
      , m_platesBorderShader(nullptr)
      , m_regionShader(nullptr)
      , m_velocityShader(nullptr)
      , m_directionShader(nullptr)
      , m_asthenosphereShader(nullptr)
      , m_heightmapTexture(nullptr)
      , m_platesTexture(nullptr)
      , m_collisionTexture(nullptr)
      , m_velocityTexture(nullptr)
      , m_directionTexture(nullptr)
      , m_waterTexture(nullptr)
      , m_arrowTexture(nullptr)
      , m_accretionTexture(nullptr)
      , m_asthenosphereTextureX(nullptr)
      , m_asthenosphereTextureY(nullptr)
      , m_sizeX(sizeX)
      , m_sizeZ(sizeZ)
      , m_vao(0)
      , m_vb(0)
      , m_ib(0)
{}

void Terrain::init()
{
    m_heightmapShader = getShader("heightMap");
    m_platesColoredShader = getShader("platesColored");
    m_platesBorderShader = getShader("platesBorder");
    m_regionShader = getShader("region");
    m_velocityShader = getShader("velocity");
    m_directionShader = getShader("direction");
    m_asthenosphereShader = getShader("asthenosphere");
    m_heightmapTexture = getTexture("heightMap");
    m_platesTexture = getTexture("cudaPlateTexture");
    m_collisionTexture = getTexture("collisionMap");
    m_velocityTexture = getTexture("velocityTexture");
    m_directionTexture = getTexture("directionTexture");
    m_waterTexture = getTexture("waterTexture");
    m_arrowTexture = getTexture("arrowTexture");
    m_accretionTexture = getTexture("accretionTexture");
    m_asthenosphereTextureX = getTexture("asthenosphereTextureX");
    m_asthenosphereTextureY = getTexture("asthenosphereTextureY");
    createGlState();
    populateBuffers();
}

void Terrain::render(const Camera &camera)
{
    switch (renderSettings.renderMode)
    {
        case RenderSettings::RenderMode::NORMAL:
            m_heightmapShader->bind();
            m_heightmapShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapShader->setUniform("displaceHeight", renderSettings.renderHeight ? 1 : 0);
            m_heightmapShader->setUniform("borderRenderType", renderSettings.borderRenderMode);
            m_heightmapShader->setUniform("collisionRenderType", renderSettings.collisionRenderMode);
            m_heightmapShader->setUniform("showAccretion", renderSettings.renderAccretionPixels);
            m_heightmapShader->setUniform("showWater", renderSettings.renderWater ? 1 : 0);
            m_heightmapShader->setUniform("heightMultiplier", renderSettings.heightMultiplier);
            m_heightmapShader->setUniform("showDirectionArrows", renderSettings.renderDirections ? 1 : 0);
            m_heightmapShader->setUniform("shadingType", renderSettings.shadingMode);
            m_heightmapShader->setUniform("continentalCrustThreshold", kernelSettingsHost.continentalCrustThreshold);

            m_heightmapTexture->bind(GL_TEXTURE0);
            m_platesTexture->bind(GL_TEXTURE1);
            m_waterTexture->bind(GL_TEXTURE2);
            m_arrowTexture->bind(GL_TEXTURE3);
            m_directionTexture->bind(GL_TEXTURE4);
            m_collisionTexture->bind(GL_TEXTURE5);
            m_accretionTexture->bind(GL_TEXTURE6);
            break;
        case RenderSettings::RenderMode::SHOW_COLLISION_AREAS:
            m_regionShader->bind();
            m_regionShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_collisionTexture->bind(GL_TEXTURE1);
            break;
        case RenderSettings::RenderMode::SHOW_PLATE_DIRECTIONS:
            m_directionShader->bind();
            m_directionShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_directionTexture->bind(GL_TEXTURE1);
            break;
        case RenderSettings::RenderMode::SHOW_PLATE_VELOCITIES:
            m_velocityShader->bind();
            m_velocityShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_velocityTexture->bind(GL_TEXTURE1);
            break;
        case RenderSettings::RenderMode::SHOW_ASTHENOSPHERE:
            m_asthenosphereShader->bind();
            m_asthenosphereShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_asthenosphereTextureX->bind(GL_TEXTURE1);
            m_asthenosphereTextureY->bind(GL_TEXTURE2);
            break;
        default:
            std::cerr << "Undefined render mode" << std::endl;
    }

    glBindVertexArray(m_vao);

    glDrawElements(GL_TRIANGLES, m_sizeX * m_sizeZ * 6, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void Terrain::createGlState()
{
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vb);
    glGenBuffers(1, &m_ib);


    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vb);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ib);


    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);

    // glGenBuffers(1, &m_uvBuffer);
    // glBindBuffer(GL_ARRAY_BUFFER, m_uvBuffer);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(3 * sizeof(float)));
}

void Terrain::populateBuffers() const
{
    std::vector<Vertex> vertices((m_sizeX + 1) * (m_sizeZ + 1));
    initVertices(vertices);

    std::vector<unsigned int> indices(m_sizeX * m_sizeZ * 6);
    initIndices(indices);


    glBindBuffer(GL_ARRAY_BUFFER, m_vb);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(Vertex) * vertices.size()), vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ib);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(unsigned int) * indices.size()),
                 indices.data(), GL_STATIC_DRAW);
}

void Terrain::initVertices(std::vector<Vertex> &vertices) const
{
    int index = 0;
    for (int z = 0; z != m_sizeZ + 1; ++z)
        for (int x = 0; x != m_sizeX + 1; ++x)
            vertices[index++].init(x, z, m_sizeX, m_sizeZ);
}

void Terrain::initIndices(std::vector<unsigned int> &indices) const
{
    int index = 0;
    for (int z = 0; z != m_sizeZ; ++z)
        for (int x = 0; x != m_sizeX; ++x)
        {
            const unsigned int index_bottom_left = z * (m_sizeX + 1) + x;
            const unsigned int index_bottom_right = z * (m_sizeX + 1) + x + 1;

            const unsigned int index_top_left = (z + 1) * (m_sizeX + 1) + x;
            const unsigned int index_top_right = (z + 1) * (m_sizeX + 1) + x + 1;

            // top left tri
            indices[index++] = index_top_left;
            indices[index++] = index_top_right;
            indices[index++] = index_bottom_left;

            // bottom right tri
            indices[index++] = index_top_right;
            indices[index++] = index_bottom_right;
            indices[index++] = index_bottom_left;
        }
}
