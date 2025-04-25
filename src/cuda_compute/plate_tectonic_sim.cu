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
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

PlateTectonicSim::PlateTectonicSim(const int width, const int height, int seed, const int numStartingPlates,
                                   CudaGlInteropManager *interopManager)
    : m_width(width)
      , m_height(height)
      , m_seed(seed)
      , m_plateDataLookup(nullptr)
      , m_numStartingPlates(numStartingPlates)
      , m_interopManager(interopManager)
{
    // FluxVelocityErosion erosion = FluxVelocityErosion(height, width);
    //
    // erosion.mCapacityConstant = 4;
    // erosion.mDissolvingConstant = 0.1f;
    // erosion.mPipeLengthConstant = 0.5f;
    // erosion.mPipeCrossSectionConstant = 0.5f;
    // erosion.mGravityConstant = 9.81f;
    // erosion.mEvaporationConstant = 0.99;
    //
    // erosion.simulate(10);
    //
    // m_heightMapDevice = erosion.m_materialDevice.getPointer();
}

void PlateTectonicSim::initialize()
{
    initializeTectonics();
    generationSettings.registerCallback("executeIterations", [this]
    {
        for (int i = 0; i < generationSettings.executionIterations; ++i)
            executeIteration();
    });
    if (m_interopManager != nullptr)
        setupToggleCallbacks();
}

void PlateTectonicSim::executeIteration()
{
    const auto start{std::chrono::steady_clock::now()};
    std::cout << "Executing iteration" << std::endl;

    // Determine block numbers
    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    int numBlocksPlates = (m_maxPlates + m_threadsPerBlock - 1) / m_threadsPerBlock;

    // Allocate the pixel plate pairs as separate arrays
    CudaTextureHost<unsigned int> pixelIndicesCollisions;
    CudaTextureHost<uint8_t> plateIdsCollisions;
    pixelIndicesCollisions.initialize(m_width, m_height);
    plateIdsCollisions.initialize(m_width, m_height);
    // Execute plate movement kernel
    testingPlateMovement<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup,
                                                                 pixelIndicesCollisions.deviceTexture(),
                                                                 plateIdsCollisions.deviceTexture());

    const thrust::device_ptr<unsigned int> pixelIndicesThrust(pixelIndicesCollisions.getPointer());
    const thrust::device_ptr<uint8_t> plateIdsThrust(plateIdsCollisions.getPointer());

    thrust::sort_by_key(pixelIndicesThrust, pixelIndicesThrust + m_height * m_width, plateIdsThrust,
                        thrust::greater<unsigned int>());

    // Allocate array for the exclusive prefix sum values
    CudaTextureHost<uint8_t> exclusivePrefixSum;
    exclusivePrefixSum.initialize(m_width, m_height, 1);


    // Perform exclusive scan
    const thrust::device_ptr<uint8_t> exclusivePrefixSumThrust(exclusivePrefixSum.getPointer());
    thrust::exclusive_scan_by_key(pixelIndicesThrust, pixelIndicesThrust + m_width * m_height, exclusivePrefixSumThrust,
                                  exclusivePrefixSumThrust);


    CudaTextureHost<uint32_t> plateCollisions;
    plateCollisions.initialize(m_width, m_height, 0);

    registerPlateCollisions<<<numBlocksPixels, m_threadsPerBlock>>>(plateIdsCollisions.deviceTexture(),
                                                                    pixelIndicesCollisions.deviceTexture(),
                                                                    exclusivePrefixSum.deviceTexture(),
                                                                    plateCollisions.deviceTexture());

    // Free memory since we no longer need them
    plateIdsCollisions.free();
    pixelIndicesCollisions.free();
    exclusivePrefixSum.free();

    // std::vector<uint32_t> plateCollisionsHost(m_width * m_height);
    // if (const cudaError_t err = cudaMemcpy(plateCollisionsHost.data(), plateCollisions.getPointer(),
    //                                        m_width * m_height * sizeof(uint32_t),
    //                                        cudaMemcpyDeviceToHost); err != cudaSuccess)
    //     std::cerr << "Error memcpy plateCollisions: " << cudaGetErrorString(err) << std::endl;
    //
    // std::vector<int> counts(5);
    //
    // for (const auto value: plateCollisionsHost)
    // {
    //     if (value == 0)
    //         ++counts[0];
    //     else if (((value >> 24) & 0xFF) != 0)
    //         ++counts[4];
    //     else if (((value >> 16) & 0xFF) != 0)
    //         ++counts[3];
    //     else if (((value >> 8) & 0xFF) != 0)
    //         ++counts[2];
    //     else
    //         ++counts[1];
    // }
    //
    // std::cout << "Counts 0: " << counts[0] << std::endl;
    // std::cout << "Counts 1: " << counts[1] << std::endl;
    // std::cout << "Counts 2: " << counts[2] << std::endl;
    // std::cout << "Counts 3: " << counts[3] << std::endl;
    // std::cout << "Counts 4: " << counts[4] << std::endl;

    CudaTextureHost<float> heightMapTextureWrite;
    CudaTextureHost<uint8_t> plateIdsTextureWrite;
    CudaTextureHost<float> uplift;

    heightMapTextureWrite.initialize(m_width, m_height);
    plateIdsTextureWrite.initialize(m_width, m_height);
    uplift.initialize(m_width, m_height, 0);


    processCollisions<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(),
                                                              m_heightMapTexture.deviceTexture(),
                                                              plateCollisions.deviceTexture(), m_plateDataLookup,
                                                              plateIdsTextureWrite.deviceTexture(),
                                                              heightMapTextureWrite.deviceTexture(),
                                                              uplift.deviceTexture());

    processUplift << <numBlocksPixels, m_threadsPerBlock >> > (uplift.deviceTexture(), heightMapTextureWrite.deviceTexture(), 10, 0.1f, 20.f, m_seed);

    // TODO: Add when heightmap stuff is implemented
    if (const cudaError_t err = cudaMemcpy(m_heightMapTexture.getPointer(), heightMapTextureWrite.getPointer(), sizeof(float) * m_height * m_width, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy heightmap: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(m_plateIdsTexture.getPointer(), plateIdsTextureWrite.getPointer(), sizeof(uint8_t) * m_height * m_width, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy plateIds: " << cudaGetErrorString(err) << std::endl;

    heightMapTextureWrite.free();
    plateIdsTextureWrite.free();
    uplift.free();

    updatePlateData<<<numBlocksPlates, m_threadsPerBlock>>>(m_plateDataLookup, Vec2(m_maxPlates, 1));

    cudaDeviceSynchronize();

    if (m_interopManager)
    {
        if (generationSettings.renderMode == GenerationSettings::RenderMode::SHOW_COLLISION_AREAS)
        {
            CudaTextureHost<uint8_t> glTexture;
            glTexture.initialize(m_width, m_height);
            convertCollisionMapForGL<<<numBlocksPixels, m_threadsPerBlock>>>(
                plateCollisions.deviceTexture(), glTexture.deviceTexture());

            m_interopManager->copyConnection("collisionMap", glTexture.getPointer());
        }
        if (generationSettings.renderMode == GenerationSettings::RenderMode::SHOW_PLATES)
        {
            m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture.getPointer());
        }
        m_interopManager->copyConnection("heightMap", m_heightMapTexture.getPointer());
    }

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "Iteration duration: " << elapsed_seconds.count() << std::endl;
}

void PlateTectonicSim::copyPlateIdsGL() const
{
    m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture.getPointer());
}

void PlateTectonicSim::copyDirectionGL() const
{
    CudaTextureHost<float2> glTexture;
    glTexture.initialize(m_width, m_height);
    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    createDirectionTexture<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup,
                                                                   glTexture.deviceTexture());

    m_interopManager->copyConnection("directionTexture", glTexture.getPointer());
}

void PlateTectonicSim::copyVelocitiesGL() const
{
    CudaTextureHost<float> glTexture;
    glTexture.initialize(m_width, m_height);
    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    createVelocityTexture<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup,
                                                                  glTexture.deviceTexture());

    m_interopManager->copyConnection("velocityTexture", glTexture.getPointer());
}

void PlateTectonicSim::initializeTectonics()
{
    std::default_random_engine generator(m_seed);

    // Init plate data vector on host, copy to device

    // Init voronoi vector on host, copy to local device, and free at end of function
    const auto voronoiSeedsHost = initializeVoronoiSeeds(generator);

    Vec2<float> *voronoiSeedsDevice;
    if (const cudaError_t err = cudaMalloc(&voronoiSeedsDevice, sizeof(Vec2<float>) * m_numStartingPlates);
        err != cudaSuccess)
        std::cerr << "Error Malloc voronoiSeedsDevice: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(voronoiSeedsDevice, voronoiSeedsHost.data(),
                                           sizeof(Vec2<float>) * m_numStartingPlates,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy voronoiSeedsDevice: " << cudaGetErrorString(err) << std::endl;

    const std::vector<PlateData> plateDataHost = initializePlateData(generator, voronoiSeedsHost);
    if (const cudaError_t err = cudaMalloc(&m_plateDataLookup, sizeof(PlateData) * m_maxPlates); err != cudaSuccess)
        std::cerr << "Error Malloc m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(m_plateDataLookup, plateDataHost.data(), sizeof(PlateData) * m_maxPlates,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    m_plateIdsTexture.initialize(m_width, m_height);
    m_heightMapTexture.initialize(m_width, m_height);
    int numBlocks = (m_width * m_height + 1) / m_threadsPerBlock;
    initPlateIDs<<<numBlocks, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), voronoiSeedsDevice,
                                                   static_cast<int>(voronoiSeedsHost.size()));
    initHeightmap << <numBlocks, m_threadsPerBlock >> > (m_heightMapTexture.deviceTexture(), m_seed, 5);
    cudaFree(voronoiSeedsDevice);
    cudaDeviceSynchronize();
}

std::vector<PlateData> PlateTectonicSim::initializePlateData(std::default_random_engine &generator,
                                                             const std::vector<Vec2<float> > &voronoiSeeds) const
{
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<PlateData> plateData(m_maxPlates);

    for (int i = 0; i != m_numStartingPlates; ++i)
    {
        plateData[i].pixelCenter = {
            voronoiSeeds[i].x - floor(voronoiSeeds[i].x), voronoiSeeds[i].y - floor(voronoiSeeds[i].y)
        };
        plateData[i].velocity = (dist(generator) + 1) / 2;
        const float x_dir = dist(generator);
        const float y_dir = dist(generator);
        const float magnitude = sqrt(x_dir * x_dir + y_dir * y_dir);

        plateData[i].direction = Vec2(x_dir / magnitude, y_dir / magnitude);
    }

    return plateData;
}

void PlateTectonicSim::setupToggleCallbacks() const
{
    generationSettings.registerCallback("toggleDefaultMode", [this]
    {
        // Empty string identifier to trigger a memory free of the previous texture
        m_interopManager->toggleSubTextures("");
    });

    generationSettings.registerCallback("togglePlateMode", [this]
    {
        std::cout << "Plate mode!" << std::endl;
        m_interopManager->toggleSubTextures("cudaPlateTexture");
        copyPlateIdsGL();
    });

    generationSettings.registerCallback("toggleCollisionMode", [this]
    {
        m_interopManager->toggleSubTextures("collisionMap");
    });

    generationSettings.registerCallback("toggleDirectionMode", [this]
    {
        m_interopManager->toggleSubTextures("directionTexture");
        copyDirectionGL();
    });

    generationSettings.registerCallback("toggleVelocityMode", [this]
    {
        m_interopManager->toggleSubTextures("velocityTexture");
        copyVelocitiesGL();
    });

    generationSettings.registerCallback("toggleUpliftMode", [this]
    {
        m_interopManager->toggleSubTextures("upliftTexture");
    });
}

std::vector<Vec2<float> > PlateTectonicSim::initializeVoronoiSeeds(std::default_random_engine &generator) const
{
    // Init vector to hold the seeds
    std::vector<Vec2<float> > seeds;
    // Init random distribution for height and width
    std::uniform_real_distribution<float> randomHeight(0, static_cast<float>(m_height));
    std::uniform_real_distribution<float> randomWidth(0, static_cast<float>(m_width));

    // Populate vector with random seeds
    for (int i = 0; i != m_numStartingPlates; ++i)
        seeds.emplace_back(randomHeight(generator), randomWidth(generator));

    return seeds;
}
