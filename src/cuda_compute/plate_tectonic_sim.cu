#include "plate_tectonic_sim.h"

#include <iostream>
#include <random>

#include "random_texture.h"
#include "cuda_gl_interop_manager.h"
#include "plate_data.h"
#include "plate_tectonics_kernel.cuh"
#include "../generation_settings.h"

extern GenerationSettings generationSettings;
#include "flux_erosion.h"

PlateTectonicSim::PlateTectonicSim(const int width, const int height, int numStartingPlates)
    : m_width(width)
    , m_height(height)
    , m_heightMapDevice(nullptr)
    , m_plateDataLookup(nullptr)
    , m_numStartingPlates(numStartingPlates)
{
    FluxVelocityErosion erosion = FluxVelocityErosion(height, width);

    erosion.mCapacityConstant = 4;
    erosion.mDissolvingConstant = 0.1f;
    erosion.mPipeLengthConstant = 0.5f;
    erosion.mPipeCrossSectionConstant = 0.5f;
    erosion.mGravityConstant = 9.81f;
    erosion.mEvaporationConstant = 0.99;

    erosion.simulate(10);

    m_heightMapDevice = erosion.m_materialDevice.getPointer();

}


void PlateTectonicSim::initialize(int seed)
{
    initializeTectonics(seed);
    generationSettings.registerCallback("executeIterations", [this]
    {
        for (int i = 0; i < generationSettings.executionIterations; ++i)
            executeIteration();
    });
}

void PlateTectonicSim::executeIteration()
{
    int numBlocks = (m_width * m_height + 1) / m_threadsPerBlock;
    CudaTextureHost<uint8_t> writeIdsTexture;
    writeIdsTexture.initialize(m_width, m_height);
    plateMovement<<<numBlocks, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), writeIdsTexture.deviceTexture(), m_plateDataLookup);
    cudaDeviceSynchronize();

    if (const cudaError_t err = cudaMemcpy(m_plateIdsTexture.getPointer(), writeIdsTexture.getPointer(), sizeof(uint8_t) * m_width * m_height, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error copying plateIdTexture: " << cudaGetErrorString(err) << std::endl;
    // numBlocks = m_maxPlates / m_threadsPerBlock;
    // updatePlateData<<<numBlocks, m_threadsPerBlock>>>(m_plateDataLookup, Vec2(m_maxPlates, 1));

    cudaDeviceSynchronize();
    m_plateIdsTexture.update();
}

void PlateTectonicSim::initializeTectonics(const int seed)
{
    std::default_random_engine generator(seed);

    // Init plate data vector on host, copy to device

    // Init voronoi vector on host, copy to local device, and free at end of function
    const auto voronoiSeedsHost = initializeVoronoiSeeds(generator);

    Vec2<float> *voronoiSeedsDevice;
    cudaMalloc(&voronoiSeedsDevice, sizeof(Vec2<float>) * m_numStartingPlates);
    cudaMemcpy(voronoiSeedsDevice, voronoiSeedsHost.data(), sizeof(Vec2<float>) * m_numStartingPlates, cudaMemcpyHostToDevice);

    const std::vector<PlateData> plateDataHost = initializePlateData(generator, voronoiSeedsHost);
    cudaError_t err = cudaMalloc(&m_plateDataLookup, sizeof(PlateData) * m_maxPlates);
    if (err != cudaSuccess)
        std::cerr << "Error Malloc m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(m_plateDataLookup, plateDataHost.data(), sizeof(PlateData) * m_maxPlates, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cerr << "Error copy m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    m_plateIdsTexture.initialize(m_width, m_height);
    int numBlocks = (m_width * m_height + 1) / m_threadsPerBlock;
    initPlateIDs<<<numBlocks, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), voronoiSeedsDevice, static_cast<int>(voronoiSeedsHost.size()));

    cudaFree(voronoiSeedsDevice);
    cudaDeviceSynchronize();
}

std::vector<PlateData> PlateTectonicSim::initializePlateData(std::default_random_engine &generator,
    const std::vector<Vec2<float>> &voronoiSeeds) const
{
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<PlateData> plateData(m_maxPlates);

    for (int i = 0; i != m_numStartingPlates; ++i)
    {
        plateData[i].pixelCenter = {voronoiSeeds[i].x - floor(voronoiSeeds[i].x), voronoiSeeds[i].y - floor(voronoiSeeds[i].y)};
        plateData[i].velocity = (dist(generator) + 1) / 2;
        const float x_dir = dist(generator);
        const float y_dir = dist(generator);
        const float magnitude = sqrt(x_dir * x_dir + y_dir * y_dir);
        
        // plateData[i].direction = Vec2(x_dir / magnitude, y_dir / magnitude);
        //
        // plateData[i].direction = Vec2<float>(1.0,1.0);
        // plateData[i].velocity = 1;
    }

    return plateData;
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