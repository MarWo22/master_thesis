#include "plate_tectonic_sim.h"

#include <iostream>
#include <random>

#include "random_texture.h"
#include "cuda_gl_interop_manager.h"
#include "plate_data.h"
#include "plate_tectonics_kernel.cuh"

PlateTectonicSim::PlateTectonicSim(const int width, const int height, int numStartingPlates)
    : m_width(width)
    , m_height(height)
    , m_heightMapDevice(nullptr)
    , m_plateDataLookup(nullptr)
    , m_numStartingPlates(numStartingPlates)
{
    m_heightMapDevice = generateRandomTexture(height, width);
}


void PlateTectonicSim::initialize(int seed)
{
    initializeTectonics(seed);
}

void PlateTectonicSim::initializeTectonics(const int seed)
{
    std::default_random_engine generator(seed);

    // Init plate data vector on host, copy to device
    const std::vector<PlateData> plateDataHost(m_maxPlates);  // Constructor runs on host
    cudaMalloc(&m_plateDataLookup, sizeof(PlateData) * m_maxPlates);
    cudaMemcpy(m_plateDataLookup, plateDataHost.data(), sizeof(PlateData) * m_maxPlates, cudaMemcpyHostToDevice);

    // Init voronoi vector on host, copy to local device, and free at end of function
    const auto voronoiSeedsHost = initializeVoronoiSeeds(generator);

    Vec2<float> *voronoiSeedsDevice;
    cudaMalloc(&voronoiSeedsDevice, sizeof(Vec2<float>) * m_numStartingPlates);
    cudaMemcpy(voronoiSeedsDevice, voronoiSeedsHost.data(), sizeof(Vec2<float>) * m_numStartingPlates, cudaMemcpyHostToDevice);
    m_plateIdsTexture.initialize(m_width, m_height);


    int numBlocks = (m_width * m_height + 1) / m_threadsPerBlock;
    initPlateIDs<<<numBlocks, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), voronoiSeedsDevice, static_cast<int>(voronoiSeedsHost.size()));


    auto data = new uint8_t[m_width * m_height];  // Host memory

    // Make sure getPointer() correctly returns the device pointer
    cudaError_t err = cudaMemcpy(data, static_cast<void *>(m_plateIdsTexture.getPointer()), sizeof(uint8_t) * m_width * m_height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    } else
    {
        std::cout << "Successfull copy\n";
    }

    cudaFree(voronoiSeedsDevice);
    cudaDeviceSynchronize();
    std::cout << "Texture Pointer: " << static_cast<void *>(m_plateIdsTexture.getPointer()) << std::endl;
}

std::vector<Vec2<float>> PlateTectonicSim::initializeVoronoiSeeds(std::default_random_engine &generator) const
{
    // Init vector to hold the seeds
    std::vector<Vec2<float>> seeds;
    // Init random distribution for height and width
    std::uniform_real_distribution<float> randomHeight(0, static_cast<float>(m_height));
    std::uniform_real_distribution<float> randomWidth(0, static_cast<float>(m_width));

    // Populate vector with random seeds
    for (int i = 0; i != m_numStartingPlates; ++i)
        seeds.emplace_back(randomHeight(generator), randomWidth(generator));

    return seeds;
}


void PlateTectonicSim::plateMovement()
{

}
void PlateTectonicSim::tectonicInteractions()
{

}
void PlateTectonicSim::plateUpdates()
{

}