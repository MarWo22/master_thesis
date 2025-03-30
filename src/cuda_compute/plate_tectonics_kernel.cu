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
        if (const float distance = textureIndexFloat.distanceSquared(seed); distance < minDistance)
        {
            minIndex = i;
            minDistance = distance;
        }
    }

    idTexture[invokeIndex] = static_cast<uint8_t>(minIndex);
}

__global__ void plateMovement(CudaTexture<uint8_t> *idTexturePtr, CudaTexture<uint8_t> *writeIdTexturePtr, const PlateData *plateLookup)
{
    CudaTexture<uint8_t> idTexture = *idTexturePtr;
    CudaTexture<uint8_t> writeTexture = *writeIdTexturePtr;
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, idTexture.size()))
        return;

    const int plateIndex = idTexture[invokeIndex];
    const PlateData plateData = plateLookup[plateIndex];

    // Get current index in the texture
    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, idTexture.size());
    // Calculate the movement of the pixel in this iteration
    const Vec2<float> pixelMovement = plateData.direction * plateData.velocity;
    // Calculate the new pixel center in integers
    const Vec2<int> newPixelCenter(static_cast<int>(plateData.pixelCenter.x + pixelMovement.x),
        static_cast<int>(plateData.pixelCenter.y + pixelMovement.y));
    // Determine the new texture index
    const Vec2<int> newTextureIndex = textureIndex + newPixelCenter ;
    // printf("%f %f\n", plateData.pixelCenter.x, plateData.pixelCenter.y);
    // Write to the output texture
    writeTexture[newTextureIndex] = plateIndex;
    // TODO:
    // Fill in empties
    // Fix wrapping
    // Fix updating of center pixels
}

__global__ void updatePlateData(PlateData *plateLookup, const Vec2<int> callSize)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(callSize))
        return;


    PlateData current = plateLookup[invokeIndex];

    current.pixelCenter = current.pixelCenter + current.direction * current.velocity;

    plateLookup[invokeIndex] = current;
}