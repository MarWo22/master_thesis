#ifndef CUDA_GL_INTEROP_MANAGER_H
#define CUDA_GL_INTEROP_MANAGER_H

#include <vector>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <memory>

class CudaGlInteropManager {

    struct Connection
    {
        const GLuint openGlTexture;
        const void *cudaDeviceTexture;
        cudaGraphicsResource *cudaGR;
        cudaArray *cudaArr;
        size_t typeSize;
        int textureWidth;
        int textureHeight;

        Connection(GLuint openGlTexture, const void *cudaDeviceTexture, size_t typeSize, int textureWidth,
                   int textureHeight);
    };

    std::vector<std::unique_ptr<Connection>> m_connections;

public:
    CudaGlInteropManager() = default;
    void addConnection(GLuint openGlTexture, void *cudaDeviceTexture, size_t typeSize, int textureWidth, int textureHeight);
    void copyAllConnections() const;
    void removeConnection(GLuint openGlTexture);

private:
    static void copyToOpenGL(Connection &connection);
};



#endif //CUDA_GL_INTEROP_MANAGER_H
