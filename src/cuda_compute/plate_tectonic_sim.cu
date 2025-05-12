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
    , m_randStatesPlates(nullptr)
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
        if (!generationSettings.isExecutingRealtime)
            for (int i = 0; i < generationSettings.executionIterations; ++i)
                executeIteration();
        else
        {
            if (generationSettings.executionIterations > 0)
            {
                executeIteration();
                generationSettings.executionIterations -= 1;
            }
            if (generationSettings.executionIterations <= 0)
                generationSettings.isExecutingRealtime = false;

        }
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

    sort_by_key(pixelIndicesThrust, pixelIndicesThrust + m_height * m_width, plateIdsThrust,
                        thrust::greater<unsigned int>());

    // Allocate array for the exclusive prefix sum values
    CudaTextureHost<uint8_t> exclusivePrefixSum;
    exclusivePrefixSum.initialize(m_width, m_height, 1);

    // Perform exclusive scan
    const thrust::device_ptr<uint8_t> exclusivePrefixSumThrust(exclusivePrefixSum.getPointer());
    exclusive_scan_by_key(pixelIndicesThrust, pixelIndicesThrust + m_width * m_height, exclusivePrefixSumThrust,
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

    

    updatePlateData<<<numBlocksPlates, m_threadsPerBlock>>>(m_plateDataLookup, m_randStatesPlates, Vec2(m_maxPlates, 1));

    updatePlateMass<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_heightMapTexture.deviceTexture(), m_plateDataLookup);

    Vec2<float> center = getPlateCenter(1, numBlocksPixels, m_threadsPerBlock);

    Vec2<float> pivot;
    Vec2<float> dir;
    float* output;

    cudaMalloc(&output, sizeof(float) * 10 * 10);

    intersectPlate<<<1, 10>>>(1, center, Vec2<float>(1,0), m_plateIdsTexture.deviceTexture(), output);


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

Vec2<float> PlateTectonicSim::getPlateCenter(uint8_t plateId, int numBlocksPixels, int m_threadsPerBlock) {
    float4* d_samples;
    cudaMalloc(&d_samples, sizeof(float4) * numBlocksPixels);
    findPlateCenter<<<numBlocksPixels, m_threadsPerBlock>>>(1, m_plateDataLookup, m_plateIdsTexture.deviceTexture(), d_samples);
    float4* h_samples = new float4[numBlocksPixels];
    cudaMemcpy(h_samples, d_samples, sizeof(float4) * numBlocksPixels, cudaMemcpyDeviceToHost);

    float sinX = 0, cosX = 0, sinY = 0, cosY = 0;
    for (int i = 0; i < numBlocksPixels; ++i) {
        sinX += h_samples[i].x;
        cosX += h_samples[i].y;
        sinY += h_samples[i].z;
        cosY += h_samples[i].w;
    }

    float angleX = atan2f(sinX, cosX);
    float angleY = atan2f(sinY, cosY);

    if (angleX < 0) angleX += CURAND_2PI;
    if (angleY < 0) angleY += CURAND_2PI;

    float midX = m_width * angleX / CURAND_2PI;
    float midY = m_height * angleY / CURAND_2PI;

    cudaFree(d_samples);
    delete[] h_samples;

    return Vec2<float>(midX, midY);
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
    if (const cudaError_t err = cudaMalloc(&m_randStatesPlates,  m_maxPlates*sizeof(curandState)); err != cudaSuccess)
        std::cerr << "Error Malloc m_randStatePlates:" << cudaGetErrorString(err) << std::endl;

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

    std::vector<PlateData> plateDataHost = initializePlateData(generator, voronoiSeedsHost);
    if (const cudaError_t err = cudaMalloc(&m_plateDataLookup, sizeof(PlateData) * m_maxPlates); err != cudaSuccess)
        std::cerr << "Error Malloc m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(m_plateDataLookup, plateDataHost.data(), sizeof(PlateData) * m_maxPlates,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    m_plateIdsTexture.initialize(m_width, m_height);
    m_heightMapTexture.initialize(m_width, m_height);

    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    initPlateIDs<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup, voronoiSeedsDevice,
                                                   static_cast<int>(voronoiSeedsHost.size()));
    initHeightmap << <numBlocksPixels, m_threadsPerBlock >> > (m_heightMapTexture.deviceTexture(), m_seed, 5);

    initPixelDependantPlateData<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_heightMapTexture.deviceTexture(), m_plateDataLookup);

    int numBlocksPlates = (m_maxPlates + m_threadsPerBlock - 1) / m_threadsPerBlock;
    initPlatesRngGen<<<numBlocksPlates, m_threadsPerBlock>>>(m_randStatesPlates, m_seed, Vec2<int>(m_maxPlates, 1));

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
        plateData[i].divergenceRandomPlate = dist(generator) > 0 ? 0 : 1;
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
