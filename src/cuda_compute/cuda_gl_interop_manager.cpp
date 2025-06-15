#include "cuda_gl_interop_manager.h"
#include "../renderer/renderer.h"
#include "../config.h"

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "plate_tectonic_sim.h"


class Texture;

CudaGlInteropManager::Connection::Connection(const GLuint openGlTexture, const size_t typeSize,
const int textureWidth, const int textureHeight, const GLint internalFormat, const GLenum format, const GLenum type)
    : openGlTexture(openGlTexture)
    , cudaGR(nullptr)
    , cudaArr(nullptr)
    , typeSize(typeSize)
    , textureWidth(textureWidth)
    , textureHeight(textureHeight)
    , internalFormat(internalFormat)
    , format(format)
    , type(type)
    , isRegistered(false)
{}

void CudaGlInteropManager::addConnection(const std::string &identifier,
                                         GLuint openGlTexture,
                                         size_t typeSize,
                                         int textureWidth,
                                         int textureHeight,
                                         GLint internalFormat,
                                         GLenum format,
                                         GLenum type)
{
    auto connection = std::make_unique<Connection>(openGlTexture, typeSize, textureWidth, textureHeight, internalFormat, format, type);

    m_connections[identifier] = std::move(connection);
}

void CudaGlInteropManager::copyConnection(const std::string &identifier, const void *cudaDeviceTexture)
{
    if (std::find(m_activeConnections.begin(), m_activeConnections.end(), identifier) == m_activeConnections.end()) {
        std::cerr << "Connection not active during copy: " << identifier << std::endl;
        return;
    }

    auto it = m_connections.find(identifier);
    if (it == m_connections.end()) {
        std::cerr << "Connection not found during copy: " << identifier << std::endl;
        return;
    }

    Connection *connection = it->second.get();

    if (!connection->isRegistered)
    {
        std::cout << connection->openGlTexture << std::endl;
        cudaGraphicsGLRegisterImage(&connection->cudaGR, connection->openGlTexture, GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore);
        cudaGraphicsMapResources(1, &connection->cudaGR, nullptr);
        cudaGraphicsSubResourceGetMappedArray(&connection->cudaArr, connection->cudaGR, 0, 0);
        cudaGraphicsUnmapResources(1, &connection->cudaGR, nullptr);
        connection->isRegistered = true;
    }


    cudaGraphicsMapResources(1, &connection->cudaGR, nullptr);

    const cudaError_t err = cudaMemcpy2DToArray(
        connection->cudaArr,
        0, 0,
        cudaDeviceTexture,
        connection->textureWidth * connection->typeSize,
        connection->textureWidth * connection->typeSize,
        connection->textureHeight,
        cudaMemcpyDeviceToDevice
    );

    if (err != cudaSuccess)
        std::cerr << "Error copying texture to openGLL: " << cudaGetErrorString(err) << std::endl;
    cudaGraphicsUnmapResources(1, &connection->cudaGR, nullptr);
}

void CudaGlInteropManager::removeConnection(const std::string &identifier)
{
    // This function is pretty crude, and assumes only a single instance exists, but should be sufficient for now

    if (m_connections.contains(identifier))
        m_connections.erase(identifier);
}

void CudaGlInteropManager::toggleSubTextures(const std::vector<std::string> &identifiers)
{
    auto texture_map = Renderer::getTextures();

    // Call regenerate on each texture in m_activeConnections that are not in identifiers
    for (const auto &activeId : m_activeConnections) {
        if (std::find(identifiers.begin(), identifiers.end(), activeId) == identifiers.end()) {
            if (const auto textureIt = texture_map.find(activeId); textureIt != texture_map.end()) {
                textureIt->second->regenerate();
            }
        }
    }

    if (!identifiers.empty())
    {
        for (const auto& identifier : identifiers)
            if (const auto textureIt = texture_map.find(identifier); textureIt != texture_map.end()) {
                Texture *texture = textureIt->second;
                if (const auto connectionIt = m_connections.find(identifier); connectionIt != m_connections.end())
                {
                    Connection *connection = connectionIt->second.get();
                    texture->load2DEmpty(connection->internalFormat, connection->format, connection->type, {connection->textureWidth, connection->textureHeight});
                    if (connection->isRegistered)
                        cudaGraphicsUnregisterResource(connection->cudaGR);

                    connection->openGlTexture = texture->id();
                    connection->isRegistered = false;
                }
                else
                    std::cerr << "Connection not found: " << identifier << std::endl;
            }
            else
                std::cerr << "Texture not found: " << identifier << std::endl;
    }
    m_activeConnections = identifiers;
}