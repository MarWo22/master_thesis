#include "cuda_gl_interop_manager.h"

#include <cuda_runtime_api.h>


CudaGlInteropManager::Connection::Connection(const GLuint openGlTexture, const void *cudaDeviceTexture, const size_t typeSize,
                                         const int textureWidth, const int textureHeight)
    : openGlTexture(openGlTexture)
    , cudaDeviceTexture(cudaDeviceTexture)
    , cudaGR(nullptr)
    , cudaArr(nullptr)
    , typeSize(typeSize)
    , textureWidth(textureWidth)
    , textureHeight(textureHeight)
{}

void CudaGlInteropManager::addConnection(const GLuint openGlTexture, void *cudaDeviceTexture, const size_t typeSize,
                                         const int textureWidth, const int textureHeight)
{
    auto connection = std::make_unique<Connection>(openGlTexture, cudaDeviceTexture, typeSize, textureWidth, textureHeight);

    cudaGraphicsGLRegisterImage(&connection->cudaGR, connection->openGlTexture, GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaGraphicsMapResources(1, &connection->cudaGR, nullptr);
    cudaGraphicsSubResourceGetMappedArray(&connection->cudaArr, connection->cudaGR, 0, 0);
    cudaGraphicsUnmapResources(1, &connection->cudaGR, nullptr);

    m_connections.push_back(std::move(connection));
}

void CudaGlInteropManager::copyAllConnections() const
{
    for (int i = 0; i != m_connections.size(); ++i)
        copyToOpenGL(*m_connections[i]);
}

void CudaGlInteropManager::removeConnection(const GLuint openGlTexture)
{
    // This function is pretty crude, and assumes only a single instance exists, but should be sufficient for now
    int index = -1;
    for (int i = 0; i != m_connections.size(); ++i)
        if (m_connections[i]->openGlTexture == openGlTexture)
        {
            index = i;
            break;
        }

    if (index != -1)
        m_connections.erase(m_connections.begin() + index);
}

void CudaGlInteropManager::copyToOpenGL(Connection &connection)
{
    cudaGraphicsMapResources(1, &connection.cudaGR, nullptr);
    cudaMemcpy2DToArray(connection.cudaArr, 0, 0, connection.cudaDeviceTexture, connection.textureWidth * connection.typeSize,
        connection.textureWidth * connection.typeSize,connection.textureHeight, cudaMemcpyDeviceToDevice );
    cudaGraphicsUnmapResources(1, &connection.cudaGR, nullptr);
}
