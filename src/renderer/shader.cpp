#include <fstream>
#include <iostream>
#include <glm/glm.hpp>
#include "shader.h"

#include "gl_misc.h"

Shader::Shader()
    : m_ShaderID(0)
{ }

Shader::Shader(const std::string &vertexShader, const std::string &fragmentShader)
    : m_vsPath(vertexShader),
      m_fsPath(fragmentShader),
      m_ShaderID(0)
{
    createShaderProgram();
}

Shader::Shader(const std::string &path, GLenum shaderType)
    : m_ShaderID(0)
{
    addShader(path, shaderType);
    createShaderProgram();
}

Shader::~Shader()
{
    if (m_ShaderID != 0)
        glDeleteProgram(m_ShaderID);
}

Shader& Shader::operator=(Shader&& other) noexcept
{
    if (this != &other)
    {
        std::swap(m_ShaderID, other.m_ShaderID);
        other.m_ShaderID = 0;
    }

    return *this;
}

void Shader::bind() const
{
    glUseProgram(m_ShaderID);
}

void Shader::unbind() const
{
    glUseProgram(0);
}

void Shader::createShaderProgram()
{
    m_ShaderID = glCreateProgram();

    std::vector<GLuint> shaders;

    if (!m_vsPath.empty())
        shaders.push_back(compileShader(m_vsPath, GL_VERTEX_SHADER));
    if (!m_fsPath.empty())
        shaders.push_back(compileShader(m_fsPath, GL_FRAGMENT_SHADER));
    if (!m_gsPath.empty())
        shaders.push_back(compileShader(m_gsPath, GL_GEOMETRY_SHADER));
    if (!m_compPath.empty())
        shaders.push_back(compileShader(m_compPath, GL_COMPUTE_SHADER));

    m_ShaderID = linkShaderProgram(shaders);
}

GLuint Shader::linkShaderProgram(const std::vector<GLuint> &shaders)
{
    GLuint program = glCreateProgram();

    for (auto shader : shaders)
        glAttachShader(program, shader);
    glLinkProgram(program);
    glValidateProgram(program);

    for (auto shader : shaders)
        glDeleteShader(shader);

    return program;
}


GLuint Shader::compileShader(const std::string &path, GLenum shaderType)
{
    std::ifstream file(path);
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();

    GLuint id = glCreateShader(shaderType);
    glShaderSource(id, 1, &source, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char *message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile " << (shaderType == GL_VERTEX_SHADER ? "vertex" : (shaderType == GL_GEOMETRY_SHADER ? "geometry" : (shaderType == GL_COMPUTE_SHADER ? "compute" : "fragment"))) << " shader\n";
        std::cout << message << "\n";
        glDeleteShader(id);
        return 0;
    }
    return id;
}


GLint Shader::getUniformLocation(const std::string &name) const
{
    if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
        return m_UniformLocationCache[name];

    GLint location = glGetUniformLocation(m_ShaderID, name.c_str());
    m_UniformLocationCache[name] = location;
    return location;
}

void Shader::addShader(const std::string &path, GLenum shaderType)
{
    switch(shaderType)
    {
        case GL_VERTEX_SHADER:
            m_vsPath = path;
            break;
        case GL_FRAGMENT_SHADER:
            m_fsPath = path;
            break;
        case GL_GEOMETRY_SHADER:
            m_gsPath = path;
            break;
        case GL_COMPUTE_SHADER:
            m_compPath = path;
            break;
        default:
            return;
    }
}

template <typename T>
void Shader::setUniform(const std::string &name, const T &value)
{}
template <>
void Shader::setUniform<glm::mat4>(const std::string &name, const glm::mat4 &value)
{
    glUniformMatrix4fv(getUniformLocation(name), 1, false, &value[0].x);
}
template <>
void Shader::setUniform<float>(const std::string &name, const float &value)
{
    glUniform1f(getUniformLocation(name), value);
}
template <>
void Shader::setUniform<GLint>(const std::string &name, const GLint &value)
{
    glUniform1i(getUniformLocation(name), value);
}
template <>
void Shader::setUniform<GLuint>(const std::string &name, const GLuint &value)
{
    glUniform1ui(getUniformLocation(name), value);
}
template <>
void Shader::setUniform<bool>(const std::string &name, const bool &value)
{
    glUniform1i(getUniformLocation(name), value ? 1 : 0);
}
template <>
void Shader::setUniform<glm::vec3>(const std::string &name, const glm::vec3 &value)
{
    glUniform3fv(getUniformLocation(name), 1, &value.x);
}

template <>
void Shader::setUniform<std::vector<glm::vec3>>(const std::string &name, const std::vector<glm::vec3> &value)
{
    glUniform3fv(getUniformLocation(name), static_cast<int>(value.size()), &value[0][0]);
}

template <>
void Shader::setUniform<std::vector<float>>(const std::string &name, const std::vector<float> &value)
{
    glUniform1fv(getUniformLocation(name), static_cast<int>(value.size()), &value[0]);
}

template <>
void Shader::setUniform<glm::vec2>(const std::string &name, const glm::vec2 &value)
{
    glUniform2fv(getUniformLocation(name), 1, &value.x);
}


template <>
void Shader::setUniform<glm::ivec3>(const std::string &name, const glm::ivec3 &value)
{
    const glm::vec3 vec = value;
    glUniform3fv(getUniformLocation(name), 1, &vec.x);
}

template <>
void Shader::setUniform<glm::ivec2>(const std::string &name, const glm::ivec2 &value)
{
    const glm::vec2 vec = value;
    glUniform3fv(getUniformLocation(name), 1, &vec.x);
}