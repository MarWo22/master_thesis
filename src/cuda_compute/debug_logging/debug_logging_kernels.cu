#include "debug_logging_kernels.cuh"

#include "../cuda_helper.cuh"

__global__ void drawVoronoiSeedsToTexture(CudaTexture<uint8_t> *idTexturePtr, const VoronoiSeed *seeds,
                                          const int numSeeds, const int seedDrawSize)
{
    CudaTexture<uint8_t> idTexture = *idTexturePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= numSeeds)
        return;

    const auto &[position, id] = seeds[invokeIndex];

    const Vec2 seedPosition = {
        static_cast<int>(position.x),
        static_cast<int>(position.y)
    };


    const int xMin = max(0, seedPosition.x - seedDrawSize - 1);
    const int xMax = min(idTexture.size().x - 1, seedPosition.x + seedDrawSize - 1);
    const int yMin = max(0, seedPosition.y - seedDrawSize - 1);
    const int yMax = min(idTexture.size().y - 1, seedPosition.y + seedDrawSize - 1);

    for (int x = xMin; x <= xMax; ++x)
        for (int y = yMin; y <= yMax; ++y)
            idTexture[{x, y}] = id;

}
