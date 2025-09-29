//
// Created by marti on 15/07/2025.
//

#include "debug_logging.cuh"

#include <chrono>

#include "debug_logging_kernels.cuh"
#include "../plate_tectonic_sim.h"
#include "../texture_save.cuh"
#include "../types/voronoi_seed.h"

#define DEBUG_LOG_THREAD_SIZE 256

void SaveVoronoiProcesTextures(const int width, const int height, const unsigned int seed, const int startingPlates,
                               const std::vector<int> &voronoiSeeds)
{
    std::default_random_engine generator(seed);
    int numBlocksPixels = (width * height + 1) / DEBUG_LOG_THREAD_SIZE;

    const auto plateCentersHost = PlateTectonicSim::generatePlateCenters(generator, startingPlates, width, height);

    CudaTextureHost<uint8_t> plateIdsTexture;
    plateIdsTexture.initialize(width, height);

    CudaTextureHost<uint8_t> plateCentersTexture;
    plateCentersTexture.initializeAndClear(width, height, 255);

    int numBlocks = (startingPlates + DEBUG_LOG_THREAD_SIZE - 1) / DEBUG_LOG_THREAD_SIZE;

    VoronoiSeed *voronoiSeedsDevice;

    if (const cudaError_t err = cudaMalloc(&voronoiSeedsDevice, sizeof(VoronoiSeed) * plateCentersHost.size());
        err != cudaSuccess)
        std::cerr << "Error Malloc voronoiSeedsDevice: " << cudaGetErrorString(err) << "\n";

    if (const cudaError_t err = cudaMemcpy(voronoiSeedsDevice, plateCentersHost.data(),
                                           sizeof(VoronoiSeed) * plateCentersHost.size(),
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy voronoiSeedsDevice: " << cudaGetErrorString(err) << "\n";

    std::cout << "size " << plateCentersHost.size() << std::endl;
    drawVoronoiSeedsToTexture<<<numBlocks, DEBUG_LOG_THREAD_SIZE>>>(plateCentersTexture.deviceTexture(),
                                                                    voronoiSeedsDevice,
                                                                    static_cast<int>(plateCentersHost.size()), 10);

    initPlateIDs<<<numBlocksPixels, DEBUG_LOG_THREAD_SIZE>>>(plateIdsTexture.deviceTexture(), voronoiSeedsDevice,
                                                             static_cast<int>(plateCentersHost.size()));

    cudaDeviceSynchronize();


    cudaFree(voronoiSeedsDevice);
    saveGrayscale8BitCudaTextureToDiskAsRgb("plate_centers.png", plateCentersTexture);
    saveGrayscale8BitCudaTextureToDiskAsRgb("plate_center_plate_ids.png", plateIdsTexture);

    std::vector<VoronoiSeed> voronoiSeedsHost = plateCentersHost;


    for (int i = 0; i != voronoiSeeds.size(); ++i)
    {
        const int numSeeds = voronoiSeeds[i];
        voronoiSeedsHost = PlateTectonicSim::generateVoronoiSeeds(generator, voronoiSeedsHost, numSeeds, width, height);

        if (const cudaError_t err = cudaMalloc(&voronoiSeedsDevice, sizeof(VoronoiSeed) * voronoiSeedsHost.size());
            err != cudaSuccess)
            std::cerr << "Error Malloc voronoiSeedsDevice: " << cudaGetErrorString(err) << "\n";

        if (const cudaError_t err = cudaMemcpy(voronoiSeedsDevice, voronoiSeedsHost.data(),
                                               sizeof(VoronoiSeed) * voronoiSeedsHost.size(),
                                               cudaMemcpyHostToDevice); err != cudaSuccess)
            std::cerr << "Error copy voronoiSeedsDevice: " << cudaGetErrorString(err) << "\n";


        CudaTextureHost<uint8_t> texture;
        texture.initializeAndClear(width, height, 255);
        numBlocks = (static_cast<int>(voronoiSeedsHost.size()) + DEBUG_LOG_THREAD_SIZE - 1) / DEBUG_LOG_THREAD_SIZE;
        drawVoronoiSeedsToTexture<<<numBlocks, DEBUG_LOG_THREAD_SIZE>>>(texture.deviceTexture(), voronoiSeedsDevice,
                                                                        static_cast<int>(voronoiSeedsHost.size()),
                                                                        3 - i);

        initPlateIDs<<<numBlocksPixels, DEBUG_LOG_THREAD_SIZE>>>(plateIdsTexture.deviceTexture(), voronoiSeedsDevice,
                                                                 static_cast<int>(voronoiSeedsHost.size()));

        cudaFree(voronoiSeedsDevice);
        const std::string filename = "seeds_" + std::to_string(i) + ".png";
        const std::string filename_plate_ids = "plate_ids_" + std::to_string(i) + ".png";

        saveGrayscale8BitCudaTextureToDiskAsRgb(filename.c_str(), texture);
        saveGrayscale8BitCudaTextureToDiskAsRgb(filename_plate_ids.c_str(), plateIdsTexture);
    }
    std::cout << "done\n";
}

void MallocTestNoManager()
{
    const auto start{std::chrono::steady_clock::now()};

    int m_width = 0;
    int m_height = 0;

    CudaTextureHost<unsigned int> pixelIndicesCollisions;
    CudaTextureHost<uint8_t> plateIdsCollisions;
    pixelIndicesCollisions.initialize(m_width, m_height);
    plateIdsCollisions.initialize(m_width, m_height);
    CudaTextureHost<uint8_t> exclusivePrefixSum;
    exclusivePrefixSum.initializeAndClear(m_width, m_height, 1);
    CudaTextureHost<uint32_t> plateCollisions;
    plateCollisions.initializeAndClear(m_width, m_height, 0);
    CudaTextureHost<float> heightMapTextureWrite;
    CudaTextureHost<uint8_t> plateIdsTextureWrite;
    CudaTextureHost<float> upliftBufferA;
    CudaTextureHost<float> upliftBufferB;
    CudaTextureHost<bool> upliftGrid;
    CudaTextureHost<DistanceFieldBuffer> blurBuffer;
    CudaTextureHost<uint8_t> platesHaveCollided;
    heightMapTextureWrite.initialize(m_width, m_height);
    plateIdsTextureWrite.initialize(m_width, m_height);
    upliftBufferA.initializeAndClear(m_width, m_height, 0);
    upliftBufferB.initializeAndClear(m_width, m_height, 0);
    upliftGrid.initialize(static_cast<int>(m_width * powf(0.5, 6)), static_cast<int>(m_height * powf(0.5, 6)));
    blurBuffer.initialize(m_width, m_height);
    platesHaveCollided.initializeAndClear(255, 255, 0);
    CudaTextureHost<float4> fluxBuffer;
    CudaTextureHost<float> sedimentBuffer;
    fluxBuffer.initialize(m_width, m_height);
    sedimentBuffer.initialize(m_width, m_height);
    CudaTextureHost<unsigned int> labels;
    labels.initialize(m_width, m_height);
    CudaTextureHost<uint8_t> glTexture;
    glTexture.initialize(m_width, m_height);

    cudaDeviceSynchronize();

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "non-managed duration: " << elapsed_seconds.count() << std::endl;
}

void MallocTestManager(TextureManager &textureManager)
{
    const auto start{std::chrono::steady_clock::now()};

    int m_width = 0;
    int m_height = 0;

    auto pixelIndicesCollisions = textureManager.generateTexture<unsigned int>(m_width, m_height);
    auto plateIdsCollisions = textureManager.generateTexture<uint8_t>(m_width, m_height);
    auto exclusivePrefixSum = textureManager.generateTexture<uint8_t>(m_width, m_height);
    auto plateCollisions = textureManager.generateTextureAndReset<uint32_t>(m_width, m_height, 0);
    auto heightMapTextureWrite = textureManager.generateTexture<float>(m_width, m_height);
    auto plateIdsTextureWrite = textureManager.generateTexture<uint8_t>(m_width, m_height);
    auto upliftBufferA = textureManager.generateTextureAndReset<float>(m_width, m_height, 0);
    auto upliftBufferB = textureManager.generateTextureAndReset<float>(m_width, m_height, 0);
    auto upliftGrid = textureManager.generateTexture<bool>(m_width, m_height);
    auto blurBuffer = textureManager.generateTexture<DistanceFieldBuffer>(m_width, m_height);
    auto platesHaveCollided = textureManager.generateTextureAndReset<uint8_t>(m_width, m_height, 0);
    auto fluxBuffer = textureManager.generateTexture<float4>(m_width, m_height);
    auto sedimentBuffer = textureManager.generateTexture<float>(m_width, m_height);
    auto labels = textureManager.generateTexture<unsigned int>(m_width, m_height);
    auto glTexture = textureManager.generateTexture<uint8_t>(m_width, m_height);

    cudaDeviceSynchronize();

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "managed duration: " << elapsed_seconds.count() << std::endl;
}
