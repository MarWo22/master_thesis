#include "terrain.h"

#include <iostream>

#include "../generation_settings.h"

extern GenerationSettings generationSettings;

void Terrain::Vertex::init(const int xPos, const int zPos, const int sizeX, const int sizeZ)
{
    pos = glm::vec3(xPos, 0, zPos);
    uv = glm::vec2(static_cast<float>(xPos) / static_cast<float>(sizeX),
                    static_cast<float>(zPos) / static_cast<float>(sizeZ));
}

Terrain::Terrain(const int sizeX, const int sizeZ)
    : m_heightmapShader(nullptr)
    , m_platesShader(nullptr)
    , m_regionShader(nullptr)
    , m_velocityShader(nullptr)
    , m_directionShader(nullptr)
    , m_heightmapTexture(nullptr)
    , m_platesTexture(nullptr)
    , m_collisionTexture(nullptr)
    , m_velocityTexture(nullptr)
    , m_directionTexture(nullptr)
    , m_upliftTexture(nullptr)
    , m_sizeX(sizeX)
    , m_sizeZ(sizeZ)
    , m_vao(0)
    , m_vb(0)
    , m_ib(0)
{}

void Terrain::init()
{
    m_heightmapShader = getShader("heightMap");
    m_platesShader = getShader("plates");
    m_regionShader = getShader("region");
    m_velocityShader = getShader("velocity");
    m_directionShader = getShader("direction");
    m_heightmapTexture = getTexture("heightMap");
    m_platesTexture = getTexture("cudaPlateTexture");
    m_collisionTexture = getTexture("collisionMap");
    m_velocityTexture = getTexture("velocityTexture");
    m_directionTexture = getTexture("directionTexture");
    m_upliftTexture = getTexture("upliftTexture");
    createGlState();
    populateBuffers();
}

void Terrain::render(const Camera &camera)
{
    switch (generationSettings.renderMode)
    {
        case GenerationSettings::RenderMode::NORMAL:
            m_heightmapShader->bind();
            m_heightmapShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            break;
        case GenerationSettings::RenderMode::SHOW_PLATES:
            m_platesShader->bind();
            m_platesShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_platesTexture->bind(GL_TEXTURE1);
            break;
        case GenerationSettings::RenderMode::SHOW_COLLISION_AREAS:
            m_regionShader->bind();
            m_regionShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_collisionTexture->bind(GL_TEXTURE1);
            break;
        case GenerationSettings::RenderMode::SHOW_PLATE_DIRECTIONS:
            m_directionShader->bind();
            m_directionShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_directionTexture->bind(GL_TEXTURE1);
            break;
        case GenerationSettings::RenderMode::SHOW_PLATE_VELOCITIES:
            m_velocityShader->bind();
            m_velocityShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_velocityTexture->bind(GL_TEXTURE1);
            break;
        case GenerationSettings::RenderMode::SHOW_UPLIFT_AREAS:
            m_velocityShader->bind();
            m_velocityShader->setUniform("vpMat", camera.ProjectionMatrix() * camera.ViewMatrix());
            m_heightmapTexture->bind(GL_TEXTURE0);
            m_upliftTexture->bind(GL_TEXTURE1);
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
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(Vertex) * vertices.size()), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ib);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(unsigned int) * indices.size()), indices.data(), GL_STATIC_DRAW);
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

