//
// Created by marti on 21/03/2025.
//

#include "background.h"

#include <iostream>

#include "camera.h"

Background::Background()
    : m_VAO(0)
    , m_shader(nullptr)
    , m_environmentMap(nullptr)
    , m_environmentMultiplier(1.5f)
{}

void Background::init()
{
    initVAO();
    m_shader = getShader("background");
    m_environmentMap = getTexture("environmentMap");
}

void Background::render(const Camera &camera)
{
    m_shader->bind();
    m_shader->setUniform("environment_multiplier", m_environmentMultiplier);
    m_shader->setUniform("inv_PV", glm::inverse(camera.ProjectionMatrix() * camera.ViewMatrix()));
    m_shader->setUniform("camera_pos", camera.Pos());
    m_environmentMap->bind(GL_TEXTURE6);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void Background::initVAO()
{
    glGenVertexArrays(1, &m_VAO);
    static const glm::vec2 positions[] = { { -1.0f, -1.0f }, { 1.0f, -1.0f }, { 1.0f, 1.0f },
                                           { -1.0f, -1.0f }, { 1.0f, 1.0f },  { -1.0f, 1.0f } };
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

    // Now attach buffer to vertex array object.
    glBindVertexArray(m_VAO);
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
