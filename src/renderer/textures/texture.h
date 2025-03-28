#ifndef TEXTURE_H_
#define TEXTURE_H_


#include <string>
#include <glm/vec2.hpp>

#include "GL/glew.h"

class Texture
{
    GLuint m_id;
    glm::ivec2 m_size;

public:

    Texture();

    ~Texture();

    void init2D(GLint wrapMode, GLint minMagMode);

    void load2DImage(const std::string& filename, GLint internalFormat, GLenum format, GLenum type, int channels);

    void load2DEmpty(GLint internalFormat, GLenum format, GLenum type, const glm::ivec2 &size);


    [[nodiscard]] GLuint id() const { return m_id; }
    [[nodiscard]] const glm::ivec2 &size() const {return m_size; }

    void bind(GLenum texture_unit) const;

    Texture &operator=(Texture &&other) noexcept;
};


#endif //TEXTURE_H_
