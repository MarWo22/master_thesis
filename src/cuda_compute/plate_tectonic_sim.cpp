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
    , m_erosion(width, height)
{
    m_heightMapDevice = generateRandomTexture(height, width);
    
    
    m_erosion.mPipeCrossSectionConstant = .5;
    m_erosion.mPipeLengthConstant = .5;
    m_erosion.mCapacityConstant = 4;
    m_erosion.mDissolvingConstant = 0.01;
    m_erosion.mEvaporationConstant = 0.98;
    m_erosion.mGravityConstant = 9.81;

    m_erosion.start(m_heightMapDevice, 500, true);
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
    cudaMemcpy2DToArray(m_textureArr, 0, 0, m_erosion.mMaterialDevice, m_width * sizeof(float),
        m_width * sizeof(float),m_height, cudaMemcpyDeviceToDevice );
    cudaGraphicsUnmapResources(1, &m_cudaGR, nullptr);

}
