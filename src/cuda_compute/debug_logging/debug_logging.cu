//
// Created by marti on 15/07/2025.
//

#include "debug_logging.cuh"

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
    plateCentersTexture.initialize(width, height, 255);

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
        texture.initialize(width, height, 255);
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
