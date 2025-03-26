//
// Created by marti on 21/03/2025.
//

#include "plate_tectonic_sim.h"
#include <iostream>
#include "random_texture.h"
PlateTectonicSim::PlateTectonicSim(const int width, const int height)
    : m_width(width)
    , m_height(height)
    , m_cudaGR(nullptr)
    , m_textureArr(nullptr)
    , m_heightMapDevice(nullptr)
{
    m_heightMapDevice = generateRandomTexture(height, width);
}

void PlateTectonicSim::connectToOpengl2DTexture(const GLuint texturePointer)
{
    cudaGraphicsGLRegisterImage(&m_cudaGR, texturePointer, GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaGraphicsMapResources(1, &m_cudaGR, nullptr);
    cudaGraphicsSubResourceGetMappedArray(&m_textureArr, m_cudaGR, 0, 0);
    cudaGraphicsUnmapResources(1, &m_cudaGR, nullptr);
}

void PlateTectonicSim::copyToOpenGl()
{
    cudaGraphicsMapResources(1, &m_cudaGR, nullptr);
    cudaMemcpy2DToArray(m_textureArr, 0, 0, m_heightMapDevice, m_width * sizeof(float),
        m_width * sizeof(float),m_height, cudaMemcpyDeviceToDevice );
    cudaGraphicsUnmapResources(1, &m_cudaGR, nullptr);

}
