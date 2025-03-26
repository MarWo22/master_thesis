#ifndef SHADING_H
#define SHADING_H

#include <string>
#include <unordered_map>
#include <GL/glew.h>

class Shader
{
private:
    GLuint m_ShaderID;
    std::string m_vsPath;
    std::string m_fsPath;
    std::string m_gsPath;
    std::string m_compPath;
    mutable std::unordered_map<std::string, GLint> m_UniformLocationCache;


public:
    static int globalID;

public:
    Shader();
    Shader(const std::string& path, GLenum shaderType);
    Shader(const std::string& vertexShader, const std::string& fragmentShaders);
    ~Shader();
        // Needed, otherwise the destructor will delete the new ShaderProgram
    Shader& operator=(Shader&& other) noexcept;
    void addShader(const std::string& path, GLenum shaderType);
    void bind() const;
    void unbind() const;
    void createShaderProgram();
//    void SetUniform(const std::string &name, const glm::vec3 &value);
//    void SetUniform(const std::string &name, const glm::vec2 &value);
//    void SetUniform(const std::string &name, bool value);
//    void SetUniform(const std::string &name, GLuint value);
//    void SetUniform(const std::string &name, GLint value);
//    void SetUniform(const std::string &name, float value);
//    void SetUniform(const std::string &name, const glm::mat4 &value);
//    void SetUniform(const std::string &name, const std::vector<glm::vec3> &value);
//    void SetUniform(const std::string &name, const std::vector<float> &value);

    template <typename T>
    void setUniform(const std::string &name, const T &value);


private:
    static GLuint linkShaderProgram(const std::vector<GLuint>& shaders);
    static GLuint compileShader(const std::string &path, GLenum shaderType);

    GLint getUniformLocation(const std::string &name) const;
};


#endif //SHADING_H
