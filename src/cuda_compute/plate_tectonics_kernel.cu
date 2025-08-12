#include "plate_tectonics_kernel.cuh"

#include <cfloat>
#include <cstdio>
#include <curand_kernel.h>
#include <cuda/std/detail/libcxx/include/cmath>

#include "cuda_helper.cuh"
#include "cuda_noise.cuh"

#include "math_functions.h"
#include "types/iteration_statistics.h"
#include "kernel_settings.cuh"
#include "plate_tectonic_sim.h"
#include "types/blur_buffer.h"


// These should become dynamic or as input parameters:

__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr,
                             const VoronoiSeed *seeds, const int numSeeds)
{
    CudaTexture<uint8_t> idTexture = *idTexturePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, idTexture.size()))
        return;

    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, idTexture.size());

    const Vec2 textureIndexFloat = {static_cast<float>(textureIndex.x), static_cast<float>(textureIndex.y)};


    float minDistance = FLT_MAX;
    int plateID = 0;

    for (int i = 0; i != numSeeds; ++i)
    {
        const auto &[position, id] = seeds[i];
        const float diffX = abs(textureIndexFloat.x - position.x);
        const float diffY = abs(textureIndexFloat.y - position.y);

        const float wrapAdjustedX = min(diffX, static_cast<float>(idTexture.size().x) - diffX);
        const float wrapAdjustedY = min(diffY, static_cast<float>(idTexture.size().y) - diffY);

        if (const float distanceSq = wrapAdjustedX * wrapAdjustedX + wrapAdjustedY * wrapAdjustedY;
            distanceSq < minDistance)
        {
            plateID = id;
            minDistance = distanceSq;
        }
    }

    idTexture[invokeIndex] = static_cast<uint8_t>(plateID);
}

__global__ void initHeightmap(CudaTexture<float> *w_heightMapPtr, int seed, int octaves)
{
    CudaTexture<float> &r_height = *w_heightMapPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_height.size()))
        return;

    Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    float d = 0.001f;
    float value = 0;

    float amp = .1f;
    float freq = 0.01f;

    for (int octave = 0; octave < octaves; octave++)
    {
        float v = cudaNoise::simplexNoise(coord.toFloat3(), freq, seed + octave);
        float vx = cudaNoise::simplexNoise(make_float3(coord.x + d, coord.y, 0), freq, seed + octave);
        float vy = cudaNoise::simplexNoise(make_float3(coord.x, coord.y + d, 0), freq, seed + octave);

        Vec2<float> der = Vec2<float>((vx - v) / d, (vy - v) / d);

        value += remap(v / (1 + der.magnitude()), -1, 1, 0, 1) * amp;

        amp *= 0.1f;
        freq *= 10.f;
    }

    r_height[invokeIndex] = value * 2000;
}


__global__ void mergeAndCountSizeMass(CudaTexture<uint8_t> *rw_plateIdsPtr,
                                      const CudaTexture<float> *r_heightTexturePtr, const uint8_t *r_plateMergeIds,
                                      float *w_plateMass, int *w_plateSize)
{
    __shared__ float localMassSum[MAX_PLATE_COUNT];
    __shared__ int localSize[MAX_PLATE_COUNT];

    CudaTexture<uint8_t> &rw_plateIds = *rw_plateIdsPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, rw_plateIds.size()))
        return;

    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        localMassSum[threadIdx.x] = 0;
        localSize[threadIdx.x] = 0;
    }

    __syncthreads();

    // Merge if the new merged plate is different from the original plate
    uint8_t plateID = rw_plateIds[invokeIndex];

    if (const uint8_t mergedPlateID = r_plateMergeIds[plateID]; mergedPlateID != MAX_PLATE_COUNT)
    {
        rw_plateIds[invokeIndex] = mergedPlateID;
        plateID = mergedPlateID;
    }

    const float pixelHeight = (*r_heightTexturePtr)[invokeIndex];

    atomicAdd(&localMassSum[plateID], pixelHeight);
    atomicAdd(&localSize[plateID], 1);

    __syncthreads();
    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        if (localMassSum[threadIdx.x] != 0)
            atomicAdd(&w_plateMass[threadIdx.x], localMassSum[threadIdx.x]);
        if (localSize[threadIdx.x] != 0)
            atomicAdd(&w_plateSize[threadIdx.x], localSize[threadIdx.x]);
    }
}

__global__ void initPixelDependantPlateData(const CudaTexture<uint8_t> *r_idTexturePtr,
                                            const CudaTexture<float> *r_heightTexturePtr, PlateData *w_plateData)
{
    __shared__ int localSizeCounts[MAX_PLATE_COUNT];
    __shared__ float localMassSum[MAX_PLATE_COUNT];

    CudaTexture<uint8_t> r_idTexture = *r_idTexturePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_idTexture.size()))
        return;

    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        localSizeCounts[threadIdx.x] = 0;
        localMassSum[threadIdx.x] = 0;
    }

    __syncthreads();
    const int plateID = r_idTexture[invokeIndex];
    const float pixelHeight = (*r_heightTexturePtr)[invokeIndex];

    atomicAdd(&localSizeCounts[plateID], 1);
    atomicAdd(&localMassSum[plateID], pixelHeight);

    __syncthreads();
    if (threadIdx.x < MAX_PLATE_COUNT and localSizeCounts[threadIdx.x] != 0)
    {
        atomicAdd(&w_plateData[threadIdx.x].size, localSizeCounts[threadIdx.x]);
        atomicAdd(&w_plateData[threadIdx.x].mass, localMassSum[threadIdx.x]);
    }
}

__global__ void getPixelMovements(const CudaTexture<uint8_t> *r_idTexturePtr, const PlateData *r_plateLookup,
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
    const Vec2<int> newPixelCenter(
        floor(plateData.pixelCenter.x + pixelMovement.x),
        floor(plateData.pixelCenter.y + pixelMovement.y)
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
    // collisions to happen, and the result should not be that differing if one is ignored
    if (const uint8_t prefixSumVal = r_exclusivePrefixSum[invokeIndex]; prefixSumVal < 4)
    // If needed, a 64bit can be used to up this to 8, but it should already be very unlikely for 4
    {
        const uint8_t plateID = r_plateIds[invokeIndex];
        const unsigned int pixelIndex = r_pixelIndices[invokeIndex];
        const uint32_t packedVal = (0xFF - plateID) << 8 * prefixSumVal;

        atomicOr(&w_collisions[pixelIndex], packedVal);
    }
    // Just a log to see whether more than 4 collisions is common
    else
        printf("More than 4 collisions in one pixel");
}


__device__ void processDivergence(const PlateTexturesRead r_plateTextures, PlateTexturesWrite w_plateTextures,
                                  const CudaTexture<uint32_t> *r_collisionsPtr,
                                  const uint32_t *r_divergenceBitmap,
                                  uint32_t *w_hasDivergedBitmap,
                                  const unsigned int invokeIndex, const NoiseParameters noiseParameters)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateTextures.plateIdsPtr;
    // Determine the plate that was here previously
    const uint8_t previousPlateId = r_plateIds[invokeIndex];
    const PlateData &plateData = r_plateTextures.plateData[previousPlateId];

    // Get movement of said plate, and determine the post-movement sub-pixel center of the plate
    const Vec2 pixelMovementFloat = plateData.direction * plateData.velocity;

    // Use the movement to determine the pixel opposite of the current pixel
    const Vec2 pixelMovementInt = {
        static_cast<int>(floorf(plateData.pixelCenter.x + pixelMovementFloat.x)),
        static_cast<int>(floorf(plateData.pixelCenter.y + pixelMovementFloat.y))
    };
    const Vec2 textureIndex = getTextureIndex(invokeIndex, r_plateIds.size());
    const Vec2 oppositePixelIndex = textureIndex - pixelMovementInt;

    // Check which plate is in the opposite pixel, and assure there is no divergence/convergence in said pixel.
    const uint32_t collisionValue = (*r_collisionsPtr)[oppositePixelIndex];
    uint8_t newPlateId;

    const uint8_t oppositePlateId = MAX_PLATE_COUNT - collisionValue & 0xFF;
    if (collisionValue != 0 && collisionValue <= 0xFF && oppositePlateId != previousPlateId)
    {
        // If there is no convergence/divergence, we extract the sub-pixel center of the opposite pixel

        const uint8_t minPlate = previousPlateId < oppositePlateId ? previousPlateId : oppositePlateId;
        const uint8_t maxPlate = previousPlateId < oppositePlateId ? oppositePlateId : previousPlateId;

        const unsigned int bitIndex = getUpperTriangleBitmapIndex(minPlate, maxPlate, MAX_PLATE_COUNT);
        const unsigned int wordIndex = bitIndex / 32;
        const unsigned int bitOffset = bitIndex % 32;

        newPlateId = r_divergenceBitmap[wordIndex] & 1u << bitOffset ? minPlate : maxPlate;
        atomicOr(&w_hasDivergedBitmap[wordIndex], 1u << bitOffset);
    } else
    {
        newPlateId = previousPlateId;
    }
    // Lerp height to a fixed value
    // TODO: Add some noise to the divergence_height_target, to avoid fully flat crust
    const auto noisePos = float3(static_cast<float>(textureIndex.x), static_cast<float>(textureIndex.y),
                                 static_cast<float>(noiseParameters.simIndex));
    const float noise = cudaNoise::simplexNoise(noisePos, 0.01f, noiseParameters.seed);
    const float targetHeight = 1 + noise * (kernelSettings.divergence_height_target - 1);
    const float originalHeight = (*r_plateTextures.heightMapPtr)[invokeIndex];
    const float value = cuda::std::lerp(originalHeight, targetHeight,
                                        kernelSettings.divergence_interpolation_factor);

    // Assign height and plate ID to the new pixel
    (*w_plateTextures.heightMapPtr)[invokeIndex] = value;
    (*w_plateTextures.plateIdsPtr)[invokeIndex] = newPlateId;
}

__device__ Vec2<int> previousTextureIndex(const Vec2<int> currentTexIndex, const PlateData &plateData)
{
    const Vec2 offset = plateData.pixelCenter + plateData.direction * plateData.velocity;
    const Vec2 pixelOffset = {static_cast<int>(offset.x), static_cast<int>(offset.y)};

    return currentTexIndex - pixelOffset;
}

__device__ __inline__ void computePlateContribution(const PlateData &currPlateData, const float plateMass,
                                                    Vec2<float> &numerator, float &denominator)
{
    numerator += currPlateData.direction * currPlateData.velocity * plateMass;
    denominator += plateMass;
}

__device__ __inline__ void applyElasticVelocityChange(Vec2<float> *w_velocityChanges, const PlateData &currPlateData,
                                                      const Vec2<float> &finalVelocity,
                                                      const float plateMass, const uint8_t plateID)
{
    const Vec2<float> velocityDiff = (finalVelocity - currPlateData.direction * currPlateData.velocity);
    const float massDiff = (plateMass / currPlateData.mass);
    const Vec2<float> velocityChange = velocityDiff * massDiff * kernelSettings.inelasticCollisionMultiplier;

    atomicAddVec2(&w_velocityChanges[plateID], velocityChange);
}

__device__ void applyInelasticCollision(const PlateTexturesRead r_plateTextures, Vec2<float> *w_velocityChanges,
                                        const uint8_t plateA, const uint8_t plateB,
                                        const uint8_t plateC,
                                        const uint8_t plateD, const unsigned int invokeIndex)
{
    const Vec2 currentTexIndex = r_plateTextures.plateIdsPtr->indexToCoordinate(invokeIndex);
    const CudaTexture<float> &r_heightMap = *r_plateTextures.heightMapPtr;

    const bool plateCSet = plateC != MAX_PLATE_COUNT;
    const bool plateDSet = plateD != MAX_PLATE_COUNT;

    const PlateData plateAData = r_plateTextures.plateData[plateA];
    const PlateData plateBData = r_plateTextures.plateData[plateB];
    const PlateData *plateCData = nullptr;
    const PlateData *plateDData = nullptr;

    const float plateAMass = max(r_heightMap[previousTextureIndex(currentTexIndex, plateAData)], 0.001f);
    const float plateBMass = max(r_heightMap[previousTextureIndex(currentTexIndex, plateBData)], 0.001f);
    float plateCMass = 0;
    float plateDMass = 0;

    Vec2<float> numerator = {0, 0};
    float denominator = 0;


    computePlateContribution(plateAData, plateAMass, numerator, denominator);
    computePlateContribution(plateBData, plateBMass, numerator, denominator);

    if (plateCSet)
    {
        plateCData = &r_plateTextures.plateData[plateC];
        plateCMass = max(r_heightMap[previousTextureIndex(currentTexIndex, *plateCData)], 0.001f);
        computePlateContribution(*plateCData, plateCMass, numerator, denominator);
    }

    if (plateDSet)
    {
        plateDData = &r_plateTextures.plateData[plateD];
        plateDMass = max(r_heightMap[previousTextureIndex(currentTexIndex, *plateDData)], 0.001f);
        computePlateContribution(*plateDData, plateDMass, numerator, denominator);
    }

    const Vec2<float> finalVelocity = numerator / denominator;

    applyElasticVelocityChange(w_velocityChanges, plateAData, finalVelocity, plateAMass, plateA);
    applyElasticVelocityChange(w_velocityChanges, plateBData, finalVelocity, plateBMass, plateB);
    if (plateCSet)
        applyElasticVelocityChange(w_velocityChanges, *plateCData, finalVelocity, plateCMass, plateC);
    if (plateDSet)
        applyElasticVelocityChange(w_velocityChanges, *plateDData, finalVelocity, plateDMass, plateD);
}

__device__ void processConvergence(const PlateTexturesRead r_plateTextures, PlateTexturesWrite w_plateTextures,
                                   CudaTexture<float> *w_convergenceMapPtr,
                                   CudaTexture<uint8_t> *w_platesHaveCollidedPtr,
                                   const uint8_t plateA, const uint8_t plateB, const uint8_t plateC,
                                   const uint8_t plateD, const unsigned int invokeIndex)
{
    const CudaTexture<float> &r_height = *r_plateTextures.heightMapPtr;
    const CudaTexture<uint8_t> &plateIds = *r_plateTextures.plateIdsPtr;
    const Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    const PlateData plateAData = r_plateTextures.plateData[plateA];
    const float heightA = r_height[Vec2<float>(coord.x, coord.y) - plateAData.direction * plateAData.velocity];

    float value = plateAData.mass;
    float force = 0;
    uint8_t plate = plateA;


    if (plateB != MAX_PLATE_COUNT)
    {
        const PlateData plateBData = r_plateTextures.plateData[plateB];

        if (value < plateBData.mass)
        {
            value = plateBData.mass;
            plate = plateB;
        }
    }

    if (plateC != MAX_PLATE_COUNT)
    {
        const PlateData plateCData = r_plateTextures.plateData[plateC];
        if (value < plateCData.mass)
        {
            value = plateCData.mass;
            plate = plateC;
        }
    }

    if (plateD != MAX_PLATE_COUNT)
    {
        const PlateData plateDData = r_plateTextures.plateData[plateD];
        if (value < plateDData.mass)
        {
            value = plateDData.mass;
            plate = plateD;
        }
    }

    const PlateData plateData = r_plateTextures.plateData[plate];

    CudaTexture<float> &w_height = *w_plateTextures.heightMapPtr;
    w_height[invokeIndex] = r_height[Vec2<float>(coord.x, coord.y) - plateData.direction * plateData.velocity];

    (*w_convergenceMapPtr)[invokeIndex] = 1.0f;
    (*w_plateTextures.plateIdsPtr)[invokeIndex] = plate;

    // Reporting the collision to the platesHaveCollided texture
    // Only the first two plates are reported for simplicity and efficiency reasons
    const Vec2<int> accessVec(min(plateA, plateB), max(plateA, plateB));
    if (!(*w_platesHaveCollidedPtr)[accessVec])
    {
        (*w_platesHaveCollidedPtr)[accessVec] = 1;
    }
}

__device__ void processMovement(const PlateTexturesRead r_plateTextures, PlateTexturesWrite w_plateTextures,
                                const uint8_t originId, const unsigned int invokeIndex)
{
    (*w_plateTextures.plateIdsPtr)[invokeIndex] = originId;
    const CudaTexture<float> &r_height = *r_plateTextures.heightMapPtr;
    const CudaTexture<uint8_t> &plateIds = *r_plateTextures.plateIdsPtr;
    const Vec2<int> textureIndex = getTextureIndex(plateIds.size());

    CudaTexture<float> &w_height = *w_plateTextures.heightMapPtr;

    // Struct is small, so a copy is likely faster than referencing in Cuda
    const PlateData plateDataOrigin = r_plateTextures.plateData[originId];

    const Vec2<float> pixelMovement = plateDataOrigin.direction * plateDataOrigin.velocity;

    const Vec2<int> newPixelCenter(
        static_cast<int>(floorf(plateDataOrigin.pixelCenter.x + pixelMovement.x)),
        static_cast<int>(floorf(plateDataOrigin.pixelCenter.y + pixelMovement.y))
    );

    const Vec2<int> newTextureIndex = textureIndex - newPixelCenter;

    w_height[textureIndex] = r_height[newTextureIndex];
}

__global__ void processCollisions(const PlateTexturesRead r_plateTextures, PlateTexturesWrite w_plateTextures,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, const uint32_t *r_divergenceBitmap,
                                  uint32_t *w_hasDivergedBitmap, CudaTexture<float> *w_convergenceMapPtr,
                                  CudaTexture<uint8_t> *w_platesHaveCollidedPtr, Vec2<float> *w_velocityChanges,
                                  const NoiseParameters noiseParameters)
{
    // Cache dereference since used more than once
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    // Ensure within bounds
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_collisions.size()))
        return;

    if (const uint32_t collisionValue = r_collisions[invokeIndex]; collisionValue == 0)
    // Collision value is zero, indicating no plate moved onto this pixel. Thus, this pixel is a divergence zone.
        processDivergence(r_plateTextures, w_plateTextures, r_collisionsPtr, r_divergenceBitmap,
                          w_hasDivergedBitmap, invokeIndex, noiseParameters);
    else
    {
        // Extract the four packed values
        uint8_t plateA = collisionValue & 0xFF;
        uint8_t plateB = (collisionValue >> 8) & 0xFF;
        uint8_t plateC = (collisionValue >> 16) & 0xFF;
        uint8_t plateD = (collisionValue >> 24) & 0xFF;

        // Invert them to get the original plateID
        plateA = MAX_PLATE_COUNT - plateA;
        plateB = MAX_PLATE_COUNT - plateB;
        plateC = MAX_PLATE_COUNT - plateC;
        plateD = MAX_PLATE_COUNT - plateD;

        // If plateB is not undefined, it means at least two plates move into the same pixel, indicating a divergent boundary.
        if (plateB != MAX_PLATE_COUNT)
        {
            applyInelasticCollision(r_plateTextures, w_velocityChanges, plateA, plateB, plateC, plateD,
                                    invokeIndex);
            processConvergence(r_plateTextures, w_plateTextures, w_convergenceMapPtr, w_platesHaveCollidedPtr, plateA,
                               plateB, plateC, plateD, invokeIndex);
        }
        // Otherwise, only one plate moves into the pixel, indicating ordinary movement
        else
            processMovement(r_plateTextures, w_plateTextures, plateA, invokeIndex);
    }
}

__global__ void flipDivergedBitmap(uint32_t *divergenceBitmap, const uint32_t *hasDivergedBitmap)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (invokeIndex >= NUM_WORDS_TRIANGLE_SINGLE_BITS)
        return;

    divergenceBitmap[invokeIndex] ^= hasDivergedBitmap[invokeIndex];
}

__device__ float determineCollisionHeight(const CudaTexture<float> &r_heightMap, const PlateData *plateData,
                                          const uint8_t plateID, const Vec2<int> &textureIndex)
{
    const PlateData &plateAData = plateData[plateID];
    const Vec2<float> plateAPosChange = plateAData.pixelCenter + plateAData.direction * plateAData.velocity;
    const Vec2 plateAMovement = {static_cast<int>(plateAPosChange.x), static_cast<int>(plateAPosChange.y)};

    return r_heightMap[textureIndex - plateAMovement];
}

__device__ void writeCollision(const float heightA, const float heightB, const uint8_t plateA, const uint8_t plateB,
                               const uint32_t *r_collisionTypeBitmap, CudaTexture<uint64_t> &w_continentalCrustCountMatrix)
{
    const uint8_t minPlate = plateA < plateB ? plateA : plateB;
    const uint8_t maxPlate = plateA < plateB ? plateB : plateA;

    const unsigned int bitIndex = getUpperTriangleBitmapIndex(minPlate, maxPlate, MAX_PLATE_COUNT, 2);
    const unsigned int wordIndex = bitIndex / 32;
    const unsigned int bitOffset = bitIndex % 32;

    const unsigned int mask = 0b11u << bitOffset;
    const unsigned int collisionType = (r_collisionTypeBitmap[wordIndex] & mask) >> bitOffset;

    if (collisionType != 0)
        return;

    const uint64_t valueA = heightA > kernelSettings.continentalCrustThreshold ? (static_cast<uint64_t>(1) << 32) | 1 : 1;
    const uint64_t valueB = heightB > kernelSettings.continentalCrustThreshold ? (static_cast<uint64_t>(1) << 32) | 1 : 1;

    atomicAdd(&w_continentalCrustCountMatrix[Vec2<int>{plateA, plateB}], valueA);
    atomicAdd(&w_continentalCrustCountMatrix[Vec2<int>{plateB, plateA}], valueB);
}

__global__ void determineCollisionType(const PlateTexturesRead r_plateTextures,
                                       const CudaTexture<uint32_t> *r_collisionsPtr,
                                       const uint32_t *r_collisionTypeBitmap,
                                       CudaTexture<uint64_t> *w_continentalCrustCountMatrixPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();
    const CudaTexture<float> &r_heightMap = *r_plateTextures.heightMapPtr;
    if (!isWithinBounds(invokeIndex, r_heightMap.size()))
        return;

    const uint32_t collisionsPacked = (*r_collisionsPtr)[invokeIndex];

    // Exit if there is no collision between plates
    if (collisionsPacked < 0xFF)
        return;

    // Extract the four packed values
    uint8_t plateA = collisionsPacked & 0xFF;
    uint8_t plateB = (collisionsPacked >> 8) & 0xFF;
    uint8_t plateC = (collisionsPacked >> 16) & 0xFF;
    uint8_t plateD = (collisionsPacked >> 24) & 0xFF;

    // Invert them to get the original plateID
    plateA = MAX_PLATE_COUNT - plateA;
    plateB = MAX_PLATE_COUNT - plateB;
    plateC = MAX_PLATE_COUNT - plateC;
    plateD = MAX_PLATE_COUNT - plateD;

    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, r_heightMap.size());
    CudaTexture<uint64_t> &w_continentalCrustCountMatrix = *w_continentalCrustCountMatrixPtr;

    const float heightA = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateA, textureIndex);
    const float heightB = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateB, textureIndex);
    if (plateD != MAX_PLATE_COUNT)
    {
        const float heightC = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateC, textureIndex);
        const float heightD = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateD, textureIndex);
        writeCollision(heightA, heightB, plateA, plateB, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightA, heightC, plateA, plateC, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightA, heightD, plateA, plateD, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightB, heightC, plateB, plateC, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightB, heightD, plateB, plateD, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightC, heightD, plateC, plateD, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
    } else if (plateC != MAX_PLATE_COUNT)
    {
        const float heightC = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateC, textureIndex);
        writeCollision(heightA, heightB, plateA, plateB, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightA, heightC, plateA, plateC, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
        writeCollision(heightB, heightC, plateB, plateC, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
    } else
        writeCollision(heightA, heightB, plateA, plateB, r_collisionTypeBitmap, w_continentalCrustCountMatrix);
}

__global__ void createCollisionTypeMatrix(const CudaTexture<uint64_t> *r_continentalCrustCountMatrixPtr, uint32_t *w_collisionTypeBitmap)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= MAX_PLATE_COUNT * (MAX_PLATE_COUNT - 1) / 2)
        return;

    // Do some weird calculations to get plateA and plateB from the diagonal excluding upper triangle matrix
    // Had to ask chatgpt for this one
    constexpr auto a = static_cast<float>(2 * MAX_PLATE_COUNT - 1);
    const float discriminant = a * a - 8.0f * static_cast<float>(invokeIndex);
    const float root = sqrtf(discriminant);
    const auto plateA = static_cast<uint8_t>((a - root) * 0.5f);
    const unsigned int base = plateA * (2 * MAX_PLATE_COUNT - plateA - 1) / 2;
    const auto plateB = static_cast<uint8_t>(plateA + 1 + (invokeIndex - base));

    // Get word and bit index of the entry in the two-bit triangle matrix
    const unsigned int wordIndex = invokeIndex * 2 / 32;
    const unsigned int bitOffset = invokeIndex * 2 % 32;

    // Extract the bits
    const unsigned int mask = 0b11u << bitOffset;
    const unsigned int collisionType = (w_collisionTypeBitmap[wordIndex] & mask) >> bitOffset;

    // Exit if there is already a registered collision type
    if (collisionType != 0)
        return;

    const CudaTexture<uint64_t> &r_continentalCrustCountMatrix = *r_continentalCrustCountMatrixPtr;

    // Extract the count of overall colliding pixels and count of continental pixels for A
    const uint64_t plateACount = r_continentalCrustCountMatrix[Vec2<int>{plateA, plateB}];
    const auto collisionCount = static_cast<uint32_t>(plateACount & 0xFFFFFFFFULL);       // lower 32 bits
    const auto continentalCountA = static_cast<uint32_t>((plateACount >> 32) & 0xFFFFFFFF); // upper 32 bits

    // Exit if there is no collision
    if (collisionCount == 0)
        return;

    // Extract continental pixels for B as well
    const uint64_t plateBCount = r_continentalCrustCountMatrix[Vec2<int>{plateB, plateA}];
    const auto continentalCountB = static_cast<uint32_t>((plateBCount >> 32) & 0xFFFFFFFF); // upper 32 bits

    if (collisionCount != 0)
        printf("%d: (%d %d) - (%d %d)\n", collisionCount, continentalCountA, continentalCountB, plateA, plateB);

    // If both have at least 75% continental, it is considered a continental collision
    // Otherwise, the minority will subduct
    if (continentalCountA >= collisionCount * 0.75 && continentalCountB >= collisionCount * 0.75)
    {
        // 3 is continental collision
        atomicOr(&w_collisionTypeBitmap[wordIndex], 3u << bitOffset);
        printf("%d & %d are continental -(%d)-\n", plateA, plateB, collisionCount);
    }
    else if (continentalCountA < continentalCountB)
    {
        // 1 is plateA subduction
        atomicOr(&w_collisionTypeBitmap[wordIndex], 1u << bitOffset);
        printf("%d is subducting under %d -(%d)-\n", plateA, plateB, collisionCount);
    }
    else
    {
        // 2 is plateB subduction
        atomicOr(&w_collisionTypeBitmap[wordIndex], 2u << bitOffset);
        printf("%d is subducting under %d -(%d)-\n", plateB, plateA, collisionCount);

    }
}

__global__ void createUpliftGrid(const CudaTexture<float> *r_upliftMapPtr, CudaTexture<bool> *w_gridMapPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, w_gridMapPtr->size()))
        return;
    Vec2<int> coord = w_gridMapPtr->indexToCoordinate(invokeIndex);

    CudaTexture<float> r_upliftMap = *r_upliftMapPtr;
    Vec2<int> mapSize = r_upliftMapPtr->size();
    Vec2<int> gridSize = w_gridMapPtr->size();

    Vec2<int> sampleSize = Vec2<int>(mapSize.x / gridSize.x, mapSize.y / gridSize.y);

    bool value = false;
    int samples = 0;

    for (int y = 0; y < sampleSize.y && !value; y++)
    {
        for (int x = 0; x < sampleSize.x && !value; x++)
        {
            value = r_upliftMap[Vec2<int>(coord.x * sampleSize.x + x, coord.y * sampleSize.y + y)] > 0;
            samples++;
        }
    }

    (*w_gridMapPtr)[coord] = value;
}

__device__ bool collisionContains(const uint32_t collision, const uint8_t plateId)
{
    uint8_t plateA = MAX_PLATE_COUNT - collision & 0xFF;
    if (plateA == plateId) return true;
    uint8_t plateB = MAX_PLATE_COUNT - (collision >> 8) & 0xFF;
    if (plateB == plateId) return true;
    uint8_t plateC = MAX_PLATE_COUNT - (collision >> 16) & 0xFF;
    if (plateC == plateId) return true;
    uint8_t plateD = MAX_PLATE_COUNT - (collision >> 24) & 0xFF;
    if (plateD == plateId) return true;

    return false;
}


__global__ void VerticalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                             const CudaTexture<float> *r_upliftMapPtr, const CudaTexture<bool> *r_gridMapPtr,
                             CudaTexture<BlurBuffer> *w_bufferPtr, int range)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    const CudaTexture<float> &r_uplift = *r_upliftMapPtr;
    const CudaTexture<bool> &r_grid = *r_gridMapPtr;
    CudaTexture<BlurBuffer> &w_buffer = *w_bufferPtr;


    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_uplift.size()))
        return;

    const Vec2<int> center = r_uplift.indexToCoordinate(invokeIndex);
    const uint8_t plateId = r_plateIds[invokeIndex];


    /*Vec2<double> ratio = Vec2<double>(static_cast<double>(r_gridMapPtr->size().x) / r_upliftMapPtr->size().x, static_cast<double>(r_gridMapPtr->size().y) / r_upliftMapPtr->size().y);
    if ((*r_gridMapPtr)[Vec2<float>(center.x * ratio.x, center.y * ratio.y)] == 0) {
        return;
    }*/

    const float multiplier = 1.0f / range;
    float value = 0;
    int point = 0;
    for (int i = -range; i <= range; i++)
    {
        Vec2<int> sample = center + Vec2<int>(0, i);
        float x = r_uplift[sample] - (abs((float) i) * multiplier);


        if (x > value && collisionContains(r_collisions[sample], plateId))
        {
            value = x;
            point = i;
        }
    }

    w_buffer[invokeIndex] = BlurBuffer(point, value);
}

__global__ void HorizontalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                               const CudaTexture<bool> *r_gridMapPtr, const CudaTexture<BlurBuffer> *r_bufferPtr,
                               CudaTexture<float> *w_heightMapPtr, int range)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    const CudaTexture<bool> &r_grid = *r_gridMapPtr;
    const CudaTexture<BlurBuffer> &r_buffer = *r_bufferPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_buffer.size()))
        return;

    const Vec2<int> center = r_buffer.indexToCoordinate(invokeIndex);
    const uint8_t plateId = r_plateIds[invokeIndex];

    /*Vec2<double> ratio = Vec2<double>(static_cast<double>(r_gridMapPtr->size().x) / w_heightMapPtr->size().x, static_cast<double>(r_gridMapPtr->size().y) / w_heightMapPtr->size().y);
    if ((*r_gridMapPtr)[Vec2<float>(center.x * ratio.x, center.y * ratio.y)] == 0) {
        return;
    }*/

    const float multiplier = 1.0f / range;
    float value = 0;

    for (int i = -range; i <= range; i++)
    {
        Vec2<int> sample = center + Vec2<int>(i, 0);
        BlurBuffer buffer = r_buffer[sample];

        //float x = buffer.value + (abs((float)buffer.offset) * multiplier) - pythagoras[abs(buffer.offset)][abs(i)] * multiplier;
        float x = buffer.value - abs((float) i) * multiplier;

        if (x > value && collisionContains(r_collisions[sample], plateId))
        {
            value = x;
        }
    }

    (*w_heightMapPtr)[invokeIndex] += value * 0.001f;
}


__global__ void processUplift(const CudaTexture<float> *r_upliftMapPtr, const CudaTexture<bool> *r_gridMapPtr,
                              CudaTexture<float> *w_heightMapPtr, const int size,
                              const float noiseFrequency, const float noiseIntensity, const int seed)
{
    const CudaTexture<float> &r_uplift = *r_upliftMapPtr;
    CudaTexture<float> &w_height = *w_heightMapPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, w_height.size()))
        return;

    Vec2<int> center = w_height.indexToCoordinate(invokeIndex);

    Vec2<double> ratio = Vec2<double>(static_cast<double>(r_gridMapPtr->size().x) / w_heightMapPtr->size().x,
                                      static_cast<double>(r_gridMapPtr->size().y) / w_heightMapPtr->size().y);
    if ((*r_gridMapPtr)[Vec2<float>(center.x * ratio.x, center.y * ratio.y)] == 0)
    {
        return;
    }
    //w_height[invokeIndex] = (*r_gridMapPtr)[f] ? 0.11 : 0;

    int offset = noiseIntensity + cudaNoise::simplexNoise(make_float3(center.x, center.y, 0.0f), noiseFrequency, 100) *
                 noiseIntensity;

    int dim(size + offset);
    float size2 = dim * dim;
    if (dim % 2 == 1)
    {
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
            if (weight > EPSILON)
                value += (r_uplift[sample + center] * weight);
        }
    }

    float noise = clamp(cudaNoise::simplexNoise(make_float3(center.x, center.y, 0.0f), 0.1f, 100), 1.0f, 0.1f);
    w_height[invokeIndex] = clamp01(value / size2) * 0.1f;
}

__global__ void downscaleUplift(const CudaTexture<float> *r_inputMapPtr, CudaTexture<float> *w_outputMapPtr,
                                int inWidth, int inHeight, int outWidth, int outHeight)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, Vec2<int>(inWidth, inHeight)))
    {
        printf("Problem! Out of bounds");
        return;
    }

    const CudaTexture<float> &r_inputMap = *r_inputMapPtr;

    /*if (size.x % outWidth != 0 || size.y % outWidth != 0) {
        printf("Problem!");
        return;
    }*/

    Vec2<int> samples = Vec2<int>(inWidth / outWidth, inHeight / outHeight);

    Vec2<int> coord = Vec2<int>(fmodf(invokeIndex, outWidth), invokeIndex / outWidth);

    Vec2<int> sampleCoord = Vec2<int>(coord.x * samples.x, coord.y * samples.y);

    float value = 0;

    for (size_t x = 0; x < samples.x; x++)
    {
        for (size_t y = 0; y < samples.y; y++)
        {
            value += r_inputMap[sampleCoord + Vec2<int>(x, y)];
        }
    }

    (*w_outputMapPtr)[coord] = value / (samples.x * samples.y);
}

__global__ void upscaleUplift(const CudaTexture<float> *r_inputMapPtr, CudaTexture<float> *w_outputMapPtr, int inWidth,
                              int inHeight, int outWidth, int outHeight)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, Vec2<int>(outWidth, outHeight)))
    {
        printf("Problem! Out of bounds");
        return;
    }

    Vec2<int> coord = Vec2<int>(fmodf(invokeIndex, outWidth), invokeIndex / outWidth);

    if (coord.x > outWidth || coord.y > outHeight)
    {
        printf("somethign wrong with the write coord: %.i x %.i \n", coord.x, coord.y);
    }

    Vec2<float> sampleCoord = Vec2<float>(coord.x * (inWidth / outWidth), coord.y * (inHeight / outHeight));

    if (sampleCoord.x > inWidth || sampleCoord.y > inHeight)
    {
        printf("somethign wrong with the sample coord: %.1f x %.1f \n", sampleCoord.x, sampleCoord.y);
    }

    (*w_outputMapPtr)[coord] = (*r_inputMapPtr)[sampleCoord];
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

__global__ void updatePlateData(PlateData *plateLookup, const Vec2<float> *r_velocityChanges, const float *r_plateMass,
                                const int *r_plateSize)
{
    if (threadIdx.x >= MAX_PLATE_COUNT)
        return;
    PlateData current = plateLookup[threadIdx.x];

    const Vec2 center = current.pixelCenter + current.direction * current.velocity;
    current.pixelCenter = {
        center.x - floor(center.x),
        center.y - floor(center.y)
    };

    const Vec2<float> newVelocityVector = current.direction * current.velocity + r_velocityChanges[threadIdx.x];
    if (newVelocityVector.x != 0 or newVelocityVector.y != 0)
    {
        current.velocity = newVelocityVector.magnitude();
        current.direction = newVelocityVector.normalized();
        // printf("%d: %f\n", invokeIndex, current.velocity);
    }

    // printf("%d original: (%f %f) change: (%f %f) new: (%f %f)\n",invokeIndex, original.x, original.y, current.velocityChange.x, current.velocityChange.y, newVelocityVector.x, newVelocityVector.y);

    current.mass = r_plateMass[threadIdx.x];
    current.size = r_plateSize[threadIdx.x]; // Will be updated in the next pixel kernel.
    current.used = false;

    plateLookup[threadIdx.x] = current;
}

__global__ void determinePlateMerge(const CudaTexture<uint8_t> *r_platesHaveCollidedPtr, const PlateData *r_plateLookup,
                                    const Vec2<float> *r_velocityChanges, uint8_t *w_plateMergeIds)
{
    const unsigned int invokeIndex = getInvokeIndex();

    const CudaTexture<uint8_t> &r_platesHaveCollided = *r_platesHaveCollidedPtr;

    if (!isWithinBounds(invokeIndex, r_platesHaveCollided.size()))
        return;

    __shared__ PlateData sharedPlateLookup[255];
    __shared__ Vec2<float> sharedVelocityChanges[255];

    if (threadIdx.x < 255)
    {
        sharedPlateLookup[threadIdx.x] = r_plateLookup[threadIdx.x];
        sharedVelocityChanges[threadIdx.x] = r_velocityChanges[threadIdx.x];
    }
    __syncthreads();

    // Check if the current plate pair has a collision
    if (!(*r_platesHaveCollidedPtr)[invokeIndex])
        return;

    const Vec2<int> texIndex = getTextureIndex(invokeIndex, r_platesHaveCollided.size());
    const PlateData &plateDataA = sharedPlateLookup[texIndex.x];
    const PlateData &plateDataB = sharedPlateLookup[texIndex.y];
    const Vec2<float> &velocityChangesA = sharedVelocityChanges[texIndex.x];
    const Vec2<float> &velocityChangesB = sharedVelocityChanges[texIndex.y];
    const Vec2<float> velocityPlateA = plateDataA.direction * plateDataA.velocity - velocityChangesA;
    const Vec2<float> velocityPlateB = plateDataB.direction * plateDataB.velocity - velocityChangesB;
    const float velocityA = velocityPlateA.magnitude();
    const Vec2<float> directionA = velocityPlateA.normalized();
    const float velocityB = velocityPlateB.magnitude();
    const Vec2<float> directionB = velocityPlateB.normalized();

    const float dot = directionA.dot(directionB);
    const float velocity_diff = abs(velocityA - velocityB);

    // printf("dot: %.5f plateADir: (%.5f, %.5f) plateBDir: (%.5f, %.5f) plateAVel: %.5f plateBVel: %.5f  %.5f\n", dot, plateDataA.direction.x, plateDataA.direction.y, plateDataB.direction.x, plateDataB.direction.y, plateDataA.velocity, plateDataB.velocity, velocity_diff);


    if (dot >= kernelSettings.mergeDotDirectionThreshold and velocity_diff <= kernelSettings.mergeVelocityDiffThreshold)
    {
        printf(
            "MERGING (%d,%d): dot: %.5f plateADir: (%.5f, %.5f) plateBDir: (%.5f, %.5f) plateAVel: %.5f plateBVel: %.5f  %.5f\n",
            texIndex.x, texIndex.y, dot, plateDataA.direction.x, plateDataA.direction.y, plateDataB.direction.x,
            plateDataB.direction.y, plateDataA.velocity, plateDataB.velocity, velocity_diff);
        // Merge, the plate with lower mass merges into the plate with higher mass
        if (plateDataA.mass > plateDataB.mass)
            w_plateMergeIds[texIndex.y] = texIndex.x;
        else
            w_plateMergeIds[texIndex.x] = texIndex.y;
    }
}

__device__ int getNewPlateId(const unsigned int *labelsShared, const unsigned int *labelCountsShared,
                             const unsigned int label, const unsigned int numUniqueLabels)
{
    for (int i = 0; i != numUniqueLabels; ++i)
        if (labelsShared[i] == label && labelCountsShared[i] >= kernelSettings.minPlateSize)
            return i;

    return -1;
}

__global__ void assignNewPlateIds(CudaTexture<unsigned int> *r_labelsPtr, const unsigned int *r_uniqueLabels,
                                  const unsigned int *r_labelCounts,
                                  uint8_t *w_originalPlateIds, CudaTexture<uint8_t> *plateIdsPtr,
                                  unsigned int *w_unassignedIndices, int *unassignedIndicesCount,
                                  const int numUniqueLabels)
{
    const unsigned int invokeIndex = getInvokeIndex();
    CudaTexture<uint8_t> &plateIds = *plateIdsPtr;

    if (!isWithinBounds(invokeIndex, plateIds.size()))
        return;

    __shared__ unsigned int labelsShared[MAX_PLATE_COUNT];
    __shared__ unsigned int labelCountsShared[MAX_PLATE_COUNT];

    const unsigned int label = (*r_labelsPtr)[invokeIndex];

    if (threadIdx.x < numUniqueLabels)
    {
        labelsShared[threadIdx.x] = r_uniqueLabels[threadIdx.x];
        labelCountsShared[threadIdx.x] = r_labelCounts[threadIdx.x];
    }

    __syncthreads();

    if (const int newPlateId = getNewPlateId(labelsShared, labelCountsShared, label, numUniqueLabels); newPlateId != -1)
    {
        const uint8_t originalPlateId = plateIds[invokeIndex];
        plateIds[invokeIndex] = static_cast<uint8_t>(newPlateId);
        w_originalPlateIds[newPlateId] = originalPlateId;
    } else
    {
        plateIds[invokeIndex] = static_cast<uint8_t>(MAX_PLATE_COUNT);
        const unsigned int index = atomicAdd(unassignedIndicesCount, 1);
        w_unassignedIndices[index] = invokeIndex;
    }
}

__global__ void assignUnassignedIdsToNeighbor(CudaTexture<uint8_t> *plateIdsPtr,
                                              const unsigned int *unassignedIndices,
                                              const unsigned int unassignedIndicesLen,
                                              int *hasRemainingWorkFlag)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= unassignedIndicesLen)
        return;

    const unsigned int unassignedIndex = unassignedIndices[invokeIndex];

    CudaTexture<uint8_t> &plateIds = *plateIdsPtr;

    // Exit if the plate has already been assigned
    if (plateIds[unassignedIndex] != MAX_PLATE_COUNT)
        return;

    const Vec2<int> textureIndex = getTextureIndex(unassignedIndex, plateIds.size());

    Vec2<int> neighborOffsets[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    uint8_t neighbors[8];
    int assignedNeighbors = 0;

    for (auto neighborOffset: neighborOffsets)
        if (const uint8_t plateId = plateIds[textureIndex + neighborOffset]; plateId != MAX_PLATE_COUNT)
            neighbors[assignedNeighbors++] = plateId;

    if (assignedNeighbors == 0)
    {
        *hasRemainingWorkFlag = 1;
        return;
    }

    uint8_t selectedNeighbor = 0;
    int count = 0;

    for (int i = 0; i != assignedNeighbors; ++i)
    {
        int localCount = 1;
        const uint8_t neighbor = neighbors[i];
        for (int j = 0; j != assignedNeighbors; ++j)
            if (i != j && neighbors[j] == neighbor)
                localCount++;

        if (localCount > count)
        {
            count = localCount;
            selectedNeighbor = neighbor;
        }
    }

    plateIds[unassignedIndex] = selectedNeighbor;
}

__global__ void copyNewPlateIdLookup(const PlateData *r_plateData, const uint8_t *r_originalPlateIds,
                                     PlateData *w_plateData)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (invokeIndex >= MAX_PLATE_COUNT)
        return;

    if (const uint8_t originalPlateId = r_originalPlateIds[invokeIndex]; originalPlateId != MAX_PLATE_COUNT)
        w_plateData[invokeIndex] = r_plateData[originalPlateId];
    else
        w_plateData[invokeIndex] = PlateData();
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
                                       CudaTexture<float2> *w_directionPtr)
{
    CudaTexture<float2> &w_direction = *w_directionPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, w_direction.size()))
        return;

    const Vec2 textureIndex = getTextureIndex(invokeIndex, w_direction.size());

    const Vec2 lookupIndex = {textureIndex.x * 16 + 7, textureIndex.y * 16 + 7};

    const int plateIndex = (*r_plateIdsPtr)[lookupIndex];
    const Vec2 direction = r_plateData[plateIndex].direction;

    w_direction[invokeIndex] = {direction.x, direction.y};
}

__global__ void findPlateCenter(const IterationStatistics *r_stats, PlateData *plateLookup,
                                const CudaTexture<uint8_t> *r_plateIdsPtr, float4 *samples)
{
    __shared__ float4 angularCoords[256];
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIdsPtr->size()))
        return;

    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    Vec2<int> coord = r_plateIds.indexToCoordinate(invokeIndex);

    uint8_t currentId = r_plateIds[invokeIndex];
    if (coord.x % 10 == 0 && coord.y % 10 == 0 && currentId == r_stats->largestPlateId)
    {
        Vec2<float> floatCoord = Vec2<float>(coord.x, coord.y);

        float angleX = CURAND_2PI * floatCoord.x / r_plateIds.size().x;
        float angleY = CURAND_2PI * floatCoord.y / r_plateIds.size().y;

        float sinX = sinf(angleX);
        float cosX = cosf(angleX);
        float sinY = sinf(angleY);
        float cosY = cosf(angleY);

        angularCoords[threadIdx.x] = make_float4(sinX, cosX, sinY, cosY);
    } else
    {
        angularCoords[threadIdx.x] = make_float4(0, 0, 0, 0);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float4 sum = {};
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum.x += angularCoords[i].x;
            sum.y += angularCoords[i].y;
            sum.z += angularCoords[i].z;
            sum.w += angularCoords[i].w;
        }
        samples[blockIdx.x] = sum;
    }
}

__device__ Vec2<float> getDirForInvokeIndex(int index, Vec2<float> start, int n)
{
    const float radiansOffset = Deg2Rad((360.0f / n) * index);
    const float c = cos(radiansOffset);
    const float s = sin(radiansOffset);

    return Vec2<float>(
        (start.x * c) - (start.y * s),
        (start.x * s) - (start.y * c));
}

__global__ void findPlausibleSplitLine(const IterationStatistics *r_stats, const Vec2<float> r_pivot,
                                       const CudaTexture<uint8_t> *r_plateIdsPtr, Vec2<float> *output)
{
    extern __shared__ float w_l_buffer[];
    float *bufferPtr = w_l_buffer;

    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const unsigned int invokeIndex = getInvokeIndex();
    const int n = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    const Vec2<float> baseDir = Vec2<float>(1, 0);
    const float minStepsize = 2.0f;

    Vec2<float> dir = getDirForInvokeIndex(invokeIndex, baseDir, n);

    float stepsize = 10.0f;
    uint8_t foundPlate = r_stats->largestPlateId;
    Vec2<float> point = r_pivot + dir * stepsize;
    float distance = stepsize;

    int iterations = 0;
    while (iterations < 100 && stepsize > minStepsize)
    {
        foundPlate = r_plateIds[Vec2<int>(point.x, point.y)];
        if (foundPlate == r_stats->largestPlateId)
        {
            point += dir * stepsize;
            distance += stepsize;
        } else
        {
            stepsize *= 0.5f;
            point = point - (dir * stepsize);
            distance -= stepsize;
            stepsize *= 0.5f;
        }

        iterations++;
    }

    bufferPtr[invokeIndex] = distance;

    __syncthreads();

    float approcimateWeight = 0;

    int end = invokeIndex + (n / 2);
    int samples = 0;
    for (size_t i = invokeIndex + 1; i < end; i++)
    {
        approcimateWeight += bufferPtr[i % n];
        samples++;
    }

    float score = (approcimateWeight / samples) / distance;

    __syncthreads();

    bufferPtr[invokeIndex] = score;

    __syncthreads();

    if (invokeIndex == 0)
    {
        int thread = 0;
        int oppposite = n * 0.5;
        float best = bufferPtr[0] + bufferPtr[oppposite];
        for (size_t i = 1; i < n * 0.5; i++)
        {
            if (bufferPtr[i] + bufferPtr[i + oppposite] > best)
            {
                best = bufferPtr[i] + bufferPtr[i + oppposite];
                thread = i;
            }
        }

        *output = getDirForInvokeIndex(thread, baseDir, n);

        printf("PlateId: %.i score: %.2f pivot: %.2f %.2f size: %.i \n", r_stats->largestPlateId, best, output->x,
               output->y, r_stats->largestValue);
    }
}

__global__ void splitPlate(const IterationStatistics *r_stats, const uint8_t *r_newPlateId, const Vec2<float> r_pivot,
                           const Vec2<float> *r_dir, CudaTexture<uint8_t> *w_plateIdsPtr, PlateData *plateLookup)
{
    const unsigned int invokeIndex = getInvokeIndex();

    CudaTexture<uint8_t> &w_plateIds = *w_plateIdsPtr;

    if (invokeIndex == 0)
    {
        printf("splitting from %.i to %.i \n", r_stats->largestPlateId, *r_newPlateId);

        plateLookup[*r_newPlateId].direction += Vec2<float>(r_dir->y, -r_dir->x);
        plateLookup[*r_newPlateId].direction = plateLookup[*r_newPlateId].direction.normalized();

        plateLookup[*r_newPlateId].mass = plateLookup[r_stats->largestPlateId].mass * 0.5;
        plateLookup[*r_newPlateId].size = plateLookup[r_stats->largestPlateId].size * 0.5;
        plateLookup[*r_newPlateId].velocity = plateLookup[r_stats->largestPlateId].velocity * 1.1;

        plateLookup[r_stats->largestPlateId].direction += Vec2<float>(-r_dir->y, r_dir->x);
        plateLookup[r_stats->largestPlateId].direction = plateLookup[r_stats->largestPlateId].direction.normalized();

        plateLookup[r_stats->largestPlateId].mass *= 0.5;
        plateLookup[r_stats->largestPlateId].size *= 0.5;

        plateLookup[r_stats->largestPlateId].velocity *= 1.1;
    }

    if (w_plateIds[invokeIndex] == r_stats->largestPlateId)
    {
        const Vec2<int> coordinate = w_plateIds.indexToCoordinate(invokeIndex);

        float dx = coordinate.x - r_pivot.x;
        float dy = coordinate.y - r_pivot.y;

        float distanceToLine = dx * r_dir->y - dy * r_dir->x;
        float distanceOnLine = dx * r_dir->y + dy * r_dir->x;

        float halfDistance = min(w_plateIds.size().x / abs(r_dir->x), w_plateIds.size().y / abs(r_dir->y)) * sqrt(
                                 r_dir->x * r_dir->x + r_dir->y * r_dir->y);

        float breakFreq = 0.132;
        float breakScale = 4;
        float CurveFreq = 0.1;
        float CurveScale = 10.0;

        float noise = cudaNoise::perlinNoise(make_float3(distanceOnLine * 0.1, 0, 0), 1, 123124);


        float offset = round(sin(distanceOnLine * breakFreq * noise)) * breakScale + sin(distanceOnLine * CurveFreq) *
                       CurveScale;

        if ((distanceToLine > noise * 30) ^ (distanceToLine > halfDistance))
        {
            w_plateIds[invokeIndex] = *r_newPlateId;
        }
    }
}

__global__ void statisticsPass(PlateData *w_plateData, IterationStatistics *w_stats)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex == 0)
    {
        w_stats->heaviestValue = 0;
        w_stats->largestValue = 0;
        for (size_t i = 0; i < MAX_PLATE_COUNT; i++)
        {
            if (w_stats->heaviestValue < w_plateData[i].mass)
            {
                w_stats->heaviestValue = w_plateData[i].mass;
                w_stats->heaviestPlateId = i;
            }

            if (w_stats->largestValue < w_plateData[i].size)
            {
                w_stats->largestValue = w_plateData[i].size;
                w_stats->largestPlateId = i;
            }
            w_plateData[i].used = w_plateData[i].size > 0;
        }
        printf("biggest: %d heaviest: %d \n", static_cast<int>(w_stats->largestPlateId),
               static_cast<int>(w_stats->heaviestPlateId));
    }
}

__global__ void selectUnusedPlateId(PlateData *plateLookup, uint8_t *plateId)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (invokeIndex == 0)
    {
        for (size_t i = 0; i < MAX_PLATE_COUNT; i++)
        {
            if (!plateLookup[i].used)
            {
                *plateId = i;
                plateLookup[i].used = true;
                printf("Selected: %.i \n", i);
                break;
            }
        }
    }
}

// direction mapping:
//
//      y
//  x       z
//      w
//

__global__ void rain(CudaTexture<float> *w_hydrationPtr, float deltatime, unsigned int seed)
{
    CudaTexture<float> &hydration = *w_hydrationPtr;

    Vec2<int> coords = getTextureIndex(hydration.size());
    hydration[coords] += fmaxf(0, cudaNoise::discreteNoise(make_float3(coords.x, 0, coords.y), 1, seed)) * deltatime *
            kernelSettings.hydrationRainfall;
}

__device__ float fluxSubComputation(const CudaTexture<float> r_material, const CudaTexture<float> r_hydration,
                                    float flux, Vec2<int> coordinateSelf, Vec2<int> coordinateNeighbor, float deltatime)
{
    float deltaHeight = r_material[coordinateSelf] + r_hydration[coordinateSelf] - r_material[coordinateNeighbor] -
                        r_hydration[coordinateNeighbor];

    return fmaxf(
        0, flux + deltatime * kernelSettings.hydrationPipeCrossSection * (
               (kernelSettings.gravity * deltaHeight) / kernelSettings.hydrationPipeLength));
}

__global__ void flux(const CudaTexture<float> *r_materialPtr, const CudaTexture<float> *r_hydrationPtr,
                     const CudaTexture<float4> *r_fluxPtr, CudaTexture<float4> *w_fluxPtr, float deltatime)
{
    const CudaTexture<float> &material = *r_materialPtr;
    const CudaTexture<float> &hydration = *r_hydrationPtr;
    const CudaTexture<float4> &r_flux = *r_fluxPtr;
    CudaTexture<float4> &w_flux = *w_fluxPtr;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coords = getTextureIndex(r_materialPtr->size());

    float4 current_f = r_flux[idx];
    float4 outflowFlux = make_float4(0, 0, 0, 0);

    outflowFlux.x = fluxSubComputation(material, hydration, current_f.x, coords, Vec2<int>(coords.x - 1, coords.y),
                                       deltatime);
    outflowFlux.y = fluxSubComputation(material, hydration, current_f.y, coords, Vec2<int>(coords.x, coords.y + 1),
                                       deltatime);
    outflowFlux.z = fluxSubComputation(material, hydration, current_f.z, coords, Vec2<int>(coords.x + 1, coords.y),
                                       deltatime);
    outflowFlux.w = fluxSubComputation(material, hydration, current_f.w, coords, Vec2<int>(coords.x, coords.y - 1),
                                       deltatime);

    float epsilon = 1e-6f;
    float fluxTotal = outflowFlux.x + outflowFlux.y + outflowFlux.z + outflowFlux.w + epsilon;

    float K = fminf(1.0f, hydration[idx] / (fluxTotal * deltatime + 1e-6f));
    K = fmaxf(K, 0.01f);

    outflowFlux.x *= K;
    outflowFlux.y *= K;
    outflowFlux.z *= K;
    outflowFlux.w *= K;

    w_flux[idx] = outflowFlux;
}

__global__ void flow(CudaTexture<float> *w_hydrationPtr, const CudaTexture<float4> *r_fluxPtr,
                     CudaTexture<float4> *w_fluxPtr, CudaTexture<Vec2<float> > *w_velocityPtr, float deltatime)
{
    CudaTexture<float> &hydration = *w_hydrationPtr;
    const CudaTexture<float4> &r_flux = *r_fluxPtr;
    CudaTexture<float4> &w_flux = *w_fluxPtr;
    CudaTexture<Vec2<float> > &velocity = *w_velocityPtr;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coords = getTextureIndex(w_hydrationPtr->size());

    float flowIn = 0;

    flowIn += r_flux[coords + Vec2<int>(-1, 0)].z;
    flowIn += r_flux[coords + Vec2<int>(0, 1)].w;
    flowIn += r_flux[coords + Vec2<int>(1, 0)].x;
    flowIn += r_flux[coords + Vec2<int>(0, -1)].y;

    float4 localFlux = r_flux[idx];
    float flowOut = localFlux.x + localFlux.y + localFlux.z + localFlux.w;

    float deltaVolume = deltatime * (flowIn - flowOut);

    hydration[idx] = fmaxf(0.0f, hydration[idx] + deltaVolume / kernelSettings.hydrationPipeLength);

    Vec2<float> netFlow;

    netFlow.x = (r_flux[coords + Vec2<int>(-1, 0)].z - r_flux[idx].x + r_flux[idx].z - r_flux[coords + Vec2<int>(1, 0)].
                 x) * 0.5f;

    netFlow.y = (r_flux[coords + Vec2<int>(0, 1)].w - r_flux[idx].y + r_flux[idx].y - r_flux[coords + Vec2<int>(0, -1)].
                 w) * 0.5f;

    float h = fmaxf(hydration[idx], 1e-6f);
    velocity[idx] = netFlow / h;

    //hydration[idx] = max(max(r_flux[idx].x, r_flux[idx].y), max(r_flux[idx].z, r_flux[idx].w));
    w_flux[idx] = r_flux[idx];
}

__global__ void sediment(CudaTexture<float> *w_materialPtr, CudaTexture<float> *w_sedimentPtr,
                         CudaTexture<Vec2<float> > *r_velocityPtr, float deltatime)
{
    CudaTexture<float> materialTexture = *w_materialPtr;
    CudaTexture<float> sedimentTexture = *w_sedimentPtr;
    CudaTexture<Vec2<float> > velocityTexture = *r_velocityPtr;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(w_materialPtr->size());

    float threshold = 0.03;
    float C = kernelSettings.sedimentCapacity * sinf(fmaxf(materialTexture.Slope(coord), threshold)) * velocityTexture[
                  idx].magnitude();

    if (C > sedimentTexture[idx])
    {
        float s = kernelSettings.sedimentDissolving * (C - sedimentTexture[idx]);
        float temp = fmaxf(materialTexture[idx] - s, 0.0);
        float delta = sedimentTexture[idx] - temp;
        materialTexture[idx] = fmaxf(materialTexture[idx] - s, 0.0);
        sedimentTexture[idx] = fmaxf(sedimentTexture[idx] + delta, 0.0);
    } else
    {
        float s = kernelSettings.sedimentDissolving * (sedimentTexture[idx] - C);
        float temp = fmaxf(sedimentTexture[idx] - s, 0.0);
        float delta = sedimentTexture[idx] - temp;
        sedimentTexture[idx] = temp;
        materialTexture[idx] = fmaxf(materialTexture[idx] + delta, 0.0);
    }

    if (materialTexture[idx] == INFINITY || materialTexture[idx] == -INFINITY)
    {
        printf("INFINITY!!!");
    }
}

__global__ void transport(CudaTexture<float> *r_sedimentPtr, CudaTexture<float> *w_sedimentPtr,
                          CudaTexture<Vec2<float> > *r_velocityPtr, float deltatime)
{
    CudaTexture<float> r_sediment = *r_sedimentPtr;
    CudaTexture<float> w_sediment = *w_sedimentPtr;
    CudaTexture<Vec2<float> > r_velocity = *r_velocityPtr;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(r_sedimentPtr->size());

    Vec2<float> vel = r_velocity[idx];

    w_sediment[idx] = interpolate(Vec2<float>(coord.x - vel.x * deltatime, coord.y - vel.y * deltatime), r_sedimentPtr);
}

__global__ void evaporate(CudaTexture<float> *w_hydrationPtr, float deltatime)
{
    CudaTexture<float> &hydration = *w_hydrationPtr;

    unsigned int idx = getInvokeIndex();
    hydration[idx] = fmaxf(
        0.0f, hydration[idx] - (fminf(hydration[idx], 20) * kernelSettings.hydrationEvaporation * deltatime));
}
