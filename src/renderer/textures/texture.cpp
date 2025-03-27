#include <iostream>
#include "texture.h"

#include <vector>

#include "stb_image.h"
#include "image.h"

Texture::Texture()
    : m_id(INT_MAX)
{ }

Texture::~Texture()
{
    if (m_id != INT_MAX)
        glDeleteTextures(1, &m_id);
}

void Texture::init2D(const GLint wrapMode, const GLint minMagMode)
{
    glGenTextures(1, &m_id);
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMode);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, minMagMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMagMode);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error after cudaHeightMap.init2D(): " << err << std::endl;
    }

}

void Texture::load2DImage(const std::string& filename, GLint internalFormat, GLenum format, GLenum type, int channels)
{
    glBindTexture(GL_TEXTURE_2D, m_id);
    const Image image(filename, channels);
    m_size = {image.width, image.height};
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image.width, image.height, 0, format, type, image.data);
}

void Texture::load2DEmpty(const GLint internalFormat, const GLenum format, const GLenum type, const glm::ivec2 &size)
{
    m_size = size;
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, size.x, size.y, 0, format, type, nullptr);
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