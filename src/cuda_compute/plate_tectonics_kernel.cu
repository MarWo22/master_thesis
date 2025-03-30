#include "plate_tectonics_kernel.cuh"

#include <cfloat>
#include <cstdio>

#include "cuda_helper.cuh"

__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, const Vec2<float> *seeds, const int numSeeds)
{
    CudaTexture<uint8_t> idTexture = *idTexturePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, idTexture.size()))
        return;

    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, idTexture.size());

    const Vec2 textureIndexFloat = {static_cast<float>(textureIndex.x), static_cast<float>(textureIndex.y)};


    float minDistance = FLT_MAX;
    int minIndex = 0;

    for (int i = 0 ; i != numSeeds; ++i)
    {
        const Vec2<float> seed = seeds[i];
        const float distance = textureIndexFloat.distanceSquared(seed);
        if (distance < minDistance)
        {
            minIndex = i;
            minDistance = distance;
        }
    }
    idTexture[invokeIndex] = static_cast<uint8_t>(minIndex);
}

__global__ void plateMovement(const uint8_t *idTexture, uint8_t *writeIdTexture, PlateData *plateLookup, const Vec2<int> textureSize)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, textureSize))
        return;

    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, textureSize);
    const int plateIndex = idTexture[invokeIndex];
    const PlateData plateData = plateLookup[invokeIndex];

    const Vec2<float> offset = plateData.direction * plateData.velocity;
    const Vec2<int> newTextureIndex = {static_cast<int>(plateData.center.x + offset.x),
        static_cast<int>(plateData.center.y + offset.y)};

    // writeToTexture()



}
