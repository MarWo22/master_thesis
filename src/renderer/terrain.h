//
// Created by marti on 19/03/2025.
//

#ifndef TERRAIN_H
#define TERRAIN_H

#include "render_component.h"
#include "shader.h"
#include "textures/texture.h"


class Terrain final : public RenderComponent
{
    Shader *m_heightmapShader;
    Shader *m_platesShader;
    Shader *m_regionShader;
    Shader *m_velocityShader;
    Shader *m_directionShader;
    Texture *m_heightmapTexture;
    Texture *m_platesTexture;
    Texture *m_collisionTexture;
    Texture *m_velocityTexture;
    Texture *m_directionTexture;
    int m_sizeX;
    int m_sizeZ;

    GLuint m_vao;
    GLuint m_vb;
    GLuint m_ib;

    struct Vertex
    {
        glm::vec3 pos;
        glm::vec2 uv;

        void init(int xPos, int zPos, int sizeX, int sizeZ);
    };

public:
    Terrain(int sizeX, int sizeZ);

    void init() override;
    void render(const Camera &camera) override;

private:
    void createGlState();

    void populateBuffers() const;

    void initVertices(std::vector<Vertex> &vertices) const;

    void initIndices(std::vector<unsigned int> &indices) const;

};


#endif //TERRAIN_H
