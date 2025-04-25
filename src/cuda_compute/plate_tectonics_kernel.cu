#include "plate_tectonics_kernel.cuh"

#include <cfloat>
#include <cstdio>

#include "cuda_helper.cuh"
#include "cuda_noise.cuh"

#include "math_functions.h"

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

    for (int i = 0; i != numSeeds; ++i)
    {
        const Vec2<float> seed = seeds[i];

        const float diffX = abs(textureIndexFloat.x - seed.x);
        const float diffY = abs(textureIndexFloat.y - seed.y);

        const float wrapAdjustedX = min(diffX, static_cast<float>(idTexture.size().x) - diffX);
        const float wrapAdjustedY = min(diffY, static_cast<float>(idTexture.size().y) - diffY);

        if (const float distanceSq = wrapAdjustedX * wrapAdjustedX + wrapAdjustedY * wrapAdjustedY;
            distanceSq < minDistance)
        {
            minIndex = i;
            minDistance = distanceSq;
        }
    }

    idTexture[invokeIndex] = static_cast<uint8_t>(minIndex);
}

__global__ void initHeightmap(CudaTexture<float>* w_heightMapPtr, int seed, int octaves) {
    CudaTexture<float>& r_height = *w_heightMapPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_height.size()))
        return;

    Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    float value = 0.0f;

    for (int octave = 0; octave < octaves; octave++)
    {
        
    }
}


__global__ void plateMovement(CudaTexture<uint8_t> *idTexturePtr, CudaTexture<uint8_t> *writeIdTexturePtr,
                              const PlateData *plateLookup)
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
    const Vec2<int> newPixelCenter(
        static_cast<int>(floorf(plateData.pixelCenter.x + pixelMovement.x)),
        static_cast<int>(floorf(plateData.pixelCenter.y + pixelMovement.y))
    );

    // Determine the new texture index
    const Vec2<int> newTextureIndex = textureIndex + newPixelCenter;
    // printf("%f %f\n", plateData.pixelCenter.x, plateData.pixelCenter.y);
    // Write to the output texture
    writeTexture[newTextureIndex] = plateIndex;
    // TODO:
    // Fill in empties
    // Fix wrapping
    // Fix updating of center pixels
}

__global__ void testingPlateMovement(const CudaTexture<uint8_t> *r_idTexturePtr, const PlateData *r_plateLookup,
                                     CudaTexture<unsigned int> *w_pixelIndicesPtr, CudaTexture<uint8_t> *w_plateIdsPtr)
{
    // Dereference textures
    const CudaTexture<uint8_t> &r_idTexture = *r_idTexturePtr;
    CudaTexture<unsigned int> &w_pixelIndices = *w_pixelIndicesPtr;
    CudaTexture<uint8_t> &w_plateIds = *w_plateIdsPtr;


    // Ensure within bounds
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_idTexture.size()))
        return;

    // Extract plate index from the texture
    const int plateIndex = r_idTexture[invokeIndex];
    // Use plate index to lookup plate properties
    const PlateData plateData = r_plateLookup[plateIndex];


    // Get current index in the texture
    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, r_idTexture.size());
    // Calculate the movement of the pixel in this iteration
    const Vec2<float> pixelMovement = plateData.direction * plateData.velocity;
    // Calculate the new pixel center in integers
    const Vec2 newPixelCenter(
        static_cast<int>(floorf(plateData.pixelCenter.x + pixelMovement.x)),
        static_cast<int>(floorf(plateData.pixelCenter.y + pixelMovement.y))
    );

    // Determine the new texture index
    const Vec2<int> newTextureIndex = textureIndex + newPixelCenter;

    // Convert to index with wrapping
    const unsigned int indexWrapped = r_idTexture.coordinateToIndex(newTextureIndex);

    // Write to output list
    w_pixelIndices[invokeIndex] = indexWrapped;
    w_plateIds[invokeIndex] = plateIndex;
}

__global__ void registerPlateCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr,
                                        const CudaTexture<unsigned int> *r_pixelIndicesPtr,
                                        const CudaTexture<uint8_t> *r_exclusivePrefixSumPtr,
                                        CudaTexture<uint32_t> *w_collisionsPtr)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<uint8_t> &r_exclusivePrefixSum = *r_exclusivePrefixSumPtr;
    const CudaTexture<unsigned int> &r_pixelIndices = *r_pixelIndicesPtr;
    CudaTexture<uint32_t> &w_collisions = *w_collisionsPtr;


    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIds.size()))
        return;

    // Due to the 32bit limit of the texture, we can only register 4 collisions
    // If needed, a 64bit can be used to up this to 8, but it should already be very unlikely for 4
    // collisions to happen, and the result should not be that differing if one is ignored
    if (const uint8_t prefixSumVal = r_exclusivePrefixSum[invokeIndex]; prefixSumVal < 4)
    {
        const uint8_t plateID = r_plateIds[invokeIndex];
        const unsigned int pixelIndex = r_pixelIndices[invokeIndex];
        const uint32_t packedVal = (255 - plateID) << 8 * prefixSumVal;

        atomicOr(&w_collisions[pixelIndex], packedVal);
    }
    // Just a log to see whether more than 4 collisions is common
    else
        printf("More than 4 collisions in one pixel");
}


__device__ void processDivergence(const CudaTexture<uint8_t> *r_plateIdsPtr, CudaTexture<uint8_t> *w_plateIdsPtr,
                                  CudaTexture<float> *w_heightMapPtr, const unsigned int invokeIndex)
{
    const uint8_t previousPlateId = (*r_plateIdsPtr)[invokeIndex];
    (*w_plateIdsPtr)[invokeIndex] = previousPlateId;
    CudaTexture<float>& w_height = *w_heightMapPtr;
    w_height[invokeIndex] = 0.0f;
    // TODO: add oceanic crust to the heightmap
}

__device__ void processConvergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                   const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                   CudaTexture<float> *w_heightMapPtr, CudaTexture<float>* w_convergenceMapPtr,
                                   const uint8_t plateA, const uint8_t plateB, const uint8_t plateC, 
                                   const uint8_t plateD, const unsigned int invokeIndex)
{
    (*w_convergenceMapPtr)[invokeIndex] = 1.0f;
    (*w_plateIdsPtr)[invokeIndex] = plateA;
}

__device__ void processMovement(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                CudaTexture<float> *w_heightMapPtr, const uint8_t originId,
                                const unsigned int invokeIndex)
{
    (*w_plateIdsPtr)[invokeIndex] = originId;
    const CudaTexture<float>& r_height = *r_heightMapPtr;
    const CudaTexture<uint8_t> &plateIds = *r_plateIdsPtr;
    const Vec2<int> textureIndex = getTextureIndex(plateIds.size());

    Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    CudaTexture<float>& w_height = *w_heightMapPtr;

    // Struct is small, so a copy is likely faster than referencing in Cuda
    const PlateData plateDataOrigin = r_plateLookup[originId];
    
    w_height[invokeIndex] = r_height[Vec2<float>(coord.x, coord.y) - plateDataOrigin.direction * plateDataOrigin.velocity];
}

__global__ void processCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, const PlateData *r_plateLookup,
                                  CudaTexture<uint8_t> *w_plateIdsPtr, CudaTexture<float> *w_heightMapPtr, CudaTexture<float>* w_convergenceMapPtr)
{
    // Cache dereference since used more than once
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    // Ensure within bounds
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_collisions.size()))
        return;

    if (const uint32_t collisionValue = r_collisions[invokeIndex]; collisionValue == 0)
    // Collision value is zero, indicating no plate moved onto this pixel. Thus, this pixel is a divergence zone.
        processDivergence(r_plateIdsPtr, w_plateIdsPtr, w_heightMapPtr, invokeIndex);
    else
    {
        // Extract the four packed values
        uint8_t plateA = collisionValue & 0xFF;
        uint8_t plateB = (collisionValue >> 8) & 0xFF;
        uint8_t plateC = (collisionValue >> 16) & 0xFF;
        uint8_t plateD = (collisionValue >> 24) & 0xFF;

        // If they are not zero, invert them to get the original plateID
        if (plateA != 0) plateA = 255 - plateA;
        if (plateB != 0) plateB = 255 - plateB;
        if (plateC != 0) plateC = 255 - plateC;
        if (plateD != 0) plateD = 255 - plateD;

        // If plateB is not equal to zero, it means at least two plates move into the same pixel, indicating a divergent boundary.
        if (plateB != 0)
            processConvergence(r_plateIdsPtr, r_heightMapPtr, r_plateLookup, w_plateIdsPtr, w_heightMapPtr, w_convergenceMapPtr, plateA,
                               plateB, plateC, plateD, invokeIndex);
            // Otherwise, only one plate moves into the pixel, indicating ordinary movement
        else
            processMovement(r_plateIdsPtr, r_heightMapPtr, r_plateLookup, w_plateIdsPtr, w_heightMapPtr, plateA,
                            invokeIndex);
    }
}

__global__ void processUplift(CudaTexture<float>* w_upliftMapPtr, CudaTexture<float>* w_heightMapPtr, const int size, const float noiseFrequency, const float noiseIntensity, const int seed)
{
    CudaTexture<float>& r_uplift = *w_upliftMapPtr;
    CudaTexture<float>& r_height = *w_heightMapPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_uplift.size()))
        return;

    Vec2<int> center = r_uplift.indexToCoordinate(invokeIndex);
    
    int offset = noiseIntensity + cudaNoise::simplexNoise(make_float3(center.x, center.y, 0.0f), noiseFrequency, 100) * noiseIntensity;

    int dim(size + offset);
    float size2 = dim * dim;
    if (dim % 2 == 1) {
        dim--;
    }
    dim *= 0.5;
    float dim2 = dim * dim;
    

    float value = 0.0f;
    for (int x = -dim; x <= dim; x++)
    {
        for (int y = -dim; y <= dim; y++)
        {
            Vec2<int> sample(x, y);
            float weight = clamp01(1.0f - sample.magnitude() / dim);
            if(weight > EPSILON)
                value += (r_uplift[sample + center] * weight);
        }
    }
    
    float noise = clamp(cudaNoise::simplexNoise(make_float3(center.x, center.y, 0.0f), 0.1f, 100), 1.0f, 0.1f);
    
    r_height[invokeIndex] += clamp01(value / size2);
}

__global__ void convertCollisionMapForGL(const CudaTexture<uint32_t> *r_texturePtr, CudaTexture<uint8_t> *w_texturePtr)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_texturePtr->size()))
        return;

    if (const uint32_t collisionValue = (*r_texturePtr)[invokeIndex]; collisionValue == 0)
        (*w_texturePtr)[invokeIndex] = 0;
    else if ((collisionValue >> 24) & 0xFF != 0)
        (*w_texturePtr)[invokeIndex] = 0.75 * 255;
    else if ((collisionValue >> 16) & 0xFF != 0)
        (*w_texturePtr)[invokeIndex] = 0.5 * 255;
    else if ((collisionValue >> 8) & 0xFF != 0)
        (*w_texturePtr)[invokeIndex] = 0.25 * 255;
    else
        (*w_texturePtr)[invokeIndex] = 255;
}

__global__ void updatePlateData(PlateData *plateLookup, const Vec2<int> callSize)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(callSize))
        return;
    PlateData current = plateLookup[invokeIndex];

    const Vec2 center = current.pixelCenter + current.direction * current.velocity;
    current.pixelCenter = {
        center.x - static_cast<float>(static_cast<int>(center.x)),
        center.y - static_cast<float>(static_cast<int>(center.y))
    };

    plateLookup[invokeIndex] = current;
}

__global__ void createVelocityTexture(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateData,
                                      CudaTexture<float> *w_velocityPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIdsPtr->size()))
        return;

    // Extract plate index from the texture
    const int plateIndex = (*r_plateIdsPtr)[invokeIndex];
    // Use plate index to lookup plate properties
    const PlateData plateData = r_plateData[plateIndex];
    (*w_velocityPtr)[invokeIndex] = plateData.velocity;
}

__global__ void createDirectionTexture(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateData,
                                       CudaTexture<float2> *w_velocityPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIdsPtr->size()))
        return;

    // Extract plate index from the texture
    const int plateIndex = (*r_plateIdsPtr)[invokeIndex];
    // Use plate index to lookup plate properties
    const PlateData plateData = r_plateData[plateIndex];
    (*w_velocityPtr)[invokeIndex] = {plateData.direction.x, plateData.direction.x};
}
