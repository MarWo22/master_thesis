#ifndef CUDA_GL_INTEROP_MANAGER_H
#define CUDA_GL_INTEROP_MANAGER_H

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <memory>
#include <string>
#include <unordered_map>

class CudaGlInteropManager {

    struct Connection
    {
        GLuint openGlTexture;
        cudaGraphicsResource *cudaGR;
        cudaArray *cudaArr;
        size_t typeSize;
        int textureWidth;
        int textureHeight;
        GLint internalFormat;
        GLenum format;
        GLenum type;
        bool isRegistered;

        Connection(GLuint openGlTexture, size_t typeSize, int textureWidth, int textureHeight, GLint internalFormat,
                   GLenum format, GLenum type);
    };

    std::unordered_map<std::string, std::unique_ptr<Connection>> m_connections;

public:
    CudaGlInteropManager() = default;

    void addConnection(const std::string &identifier, GLuint openGlTexture, size_t typeSize, int textureWidth, int textureHeight, GLint
                       internalFormat, GLenum format, GLenum type);
    void copyConnection(const std::string &identifier, const void *cudaDeviceTexture);
    void removeConnection(const std::string &identifier);
    void toggleSubTextures(const std::string &identifier);
};



#endif //CUDA_GL_INTEROP_MANAGER_H
