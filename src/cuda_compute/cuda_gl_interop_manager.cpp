#include "cuda_gl_interop_manager.h"

#include <cuda_runtime_api.h>
#include <iostream>


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

    m_connections[openGlTexture] = std::move(connection);
}

void CudaGlInteropManager::copyAllConnections() const
{
    for (auto it = m_connections.begin(); it != m_connections.end(); ++it)
        copyToOpenGL(*it->second);
}

void CudaGlInteropManager::copyConnection(const GLuint openGLTexture)
{
    if (m_connections.contains(openGLTexture))
    {
        copyToOpenGL(*m_connections[openGLTexture]);
    }
}

void CudaGlInteropManager::removeConnection(const GLuint openGlTexture)
{
    // This function is pretty crude, and assumes only a single instance exists, but should be sufficient for now

    if (m_connections.contains(openGlTexture))
        m_connections.erase(openGlTexture);
}



void CudaGlInteropManager::copyToOpenGL(Connection &connection)
{
    cudaGraphicsMapResources(1, &connection.cudaGR, nullptr);
    const cudaError_t err = cudaMemcpy2DToArray(connection.cudaArr, 0, 0, connection.cudaDeviceTexture, connection.textureWidth * connection.typeSize,
        connection.textureWidth * connection.typeSize,connection.textureHeight, cudaMemcpyDeviceToDevice );
    if (err != cudaSuccess)
        std::cerr << "Error copying texture to openGLL: " << cudaGetErrorString(err) << std::endl;


    cudaGraphicsUnmapResources(1, &connection.cudaGR, nullptr);
}
