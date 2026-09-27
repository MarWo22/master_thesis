#include <iostream>
#include "texture.h"

#include <vector>

#include "stb_image.h"
#include "image.h"

Texture::Texture()
    : m_id(0)
    , m_size()
    , m_wrapMode(GL_CLAMP_TO_EDGE)
    , m_minMagMode(GL_LINEAR)
{}

Texture::~Texture()
{
    if (m_id)
        glDeleteTextures(1, &m_id);
}

void Texture::init2D(const GLint wrapMode, const GLint minMagMode)
{
    m_wrapMode = wrapMode;
    m_minMagMode = minMagMode;
    glGenTextures(1, &m_id);
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMode);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, minMagMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMagMode);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error after cudaHeightMap.init2D(): " << err << std::endl;
    }

}

void Texture::load2DImage(const std::string& filename, GLint internalFormat, GLenum format, GLenum type, int channels)
{
    glBindTexture(GL_TEXTURE_2D, m_id);
    const Image image(filename, channels);
    m_size = {image.width, image.height};
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image.width, image.height, 0, format, type, image.data);
}

void Texture::regenerate()
{
    if (m_id)
        glDeleteTextures(1, &m_id);

    glGenTextures(1, &m_id);
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, m_wrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, m_wrapMode);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_minMagMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_minMagMode);
    glBindTexture(GL_TEXTURE_2D, 0);

}

void Texture::load2DEmpty(const GLint internalFormat, const GLenum format, const GLenum type, const glm::ivec2 &size)
{
    m_size = size;
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, size.x, size.y, 0, format, type, nullptr);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        // Print or handle the error
        std::cerr << "OpenGL Error: " << error << std::endl;
    }
}

void Texture::bind(const GLenum texture_unit) const
{
    glActiveTexture(texture_unit);
    glBindTexture(GL_TEXTURE_2D, m_id);

}

Texture& Texture::operator=(Texture&& other) noexcept
{
    if (this != &other)
    {
        std::swap(m_id, other.m_id);
        other.m_id = 0;
    }

    return *this;
}