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
#include "types/distance_field_buffer.h"
#include "plate_tectonic_sim.h"


// Helper function to get linear weight for given offset
__device__ float getLinearWeight(int offset, int range) {
    int absOffset = abs(offset);
    if (absOffset > range) return 0.0f;

    // Linear falloff: weight = 1.0 at center, 0.0 at range
    return 1.0f - (float(absOffset) / float(range));
}

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

    float amp = 100.0f;
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

    r_height[invokeIndex] = value;
}


__global__ void mergeAndCountSizeMass(CudaTexture<uint8_t> *rw_plateIdsPtr,
                                      const CudaTexture<float> *r_heightTexturePtr, const uint8_t *r_plateMergeIds,
                                      PlateData *w_plateLookup, int *w_plateSize,
                                      const CudaTexture<Vec2<float>> *r_pressureSlopePtr)
{
    __shared__ float localMassSum[MAX_PLATE_COUNT];
    __shared__ int localSize[MAX_PLATE_COUNT];
    __shared__ int localPerimeter[MAX_PLATE_COUNT];
    __shared__ Vec2<float> localPressureSlope[MAX_PLATE_COUNT];

    CudaTexture<uint8_t> &rw_plateIds = *rw_plateIdsPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, rw_plateIds.size()))
        return;

    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        localMassSum[threadIdx.x] = 0;
        localSize[threadIdx.x] = 0;
        localPerimeter[threadIdx.x] = 0;
        localPressureSlope[threadIdx.x] = {0.0f, 0.0f};
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
    const Vec2<float> pixelVelocity = (*r_pressureSlopePtr)[invokeIndex];
    const Vec2<int> coord = rw_plateIds.indexToCoordinate(invokeIndex);
    const Vec2<int> mapSize = rw_plateIds.size();

    atomicAdd(&localMassSum[plateID], pixelHeight);
    atomicAdd(&localSize[plateID], 1);
    atomicAdd(&localPressureSlope[plateID].x, pixelVelocity.x);
    atomicAdd(&localPressureSlope[plateID].y, pixelVelocity.y);

    // Check if this pixel is on the plate boundary for perimeter calculation
    bool isBoundary = false;
    
    // Check 4-connected neighbors (N, E, S, W)
    const Vec2<int> offsets[] = {
        {0, -1}, {1, 0}, {0, 1}, {-1, 0}
    };
    
    for (int i = 0; i < 4; i++)
    {
        const Vec2<int> neighborCoord = coord + offsets[i];
     
        // Check if neighbor belongs to different plate
        uint8_t neighborPlateId = rw_plateIds[neighborCoord];
        
        // Apply same merging logic to neighbor
        if (const uint8_t mergedNeighborId = r_plateMergeIds[neighborPlateId]; mergedNeighborId != MAX_PLATE_COUNT)
        {
            neighborPlateId = mergedNeighborId;
        }
        
        if (neighborPlateId != plateID)
        {
            isBoundary = true;
            break;
        }
    }
    
    // If this pixel is on the boundary, increment the perimeter count
    if (isBoundary)
    {
        atomicAdd(&localPerimeter[plateID], 1);
    }

    __syncthreads();
    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        if (localMassSum[threadIdx.x] != 0)
            atomicAdd(&w_plateLookup[threadIdx.x].mass, localMassSum[threadIdx.x]);
        if (localSize[threadIdx.x] != 0)
            atomicAdd(&w_plateLookup[threadIdx.x].size, localSize[threadIdx.x]);
        if (localPerimeter[threadIdx.x] != 0)
            atomicAdd(&w_plateLookup[threadIdx.x].perimeter, localPerimeter[threadIdx.x]);
        if (localPressureSlope[threadIdx.x].x != 0.0f || localPressureSlope[threadIdx.x].y != 0.0f)
        {
            atomicAdd(&w_plateLookup[threadIdx.x].forceFromPressure.x, localPressureSlope[threadIdx.x].x);
            atomicAdd(&w_plateLookup[threadIdx.x].forceFromPressure.y, localPressureSlope[threadIdx.x].y);
        }
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
    if (!isWithinBounds(invokeIndex, r_plateIds.size()))
        printf("Something unexpected happened 4\n");
    if (previousPlateId >= MAX_PLATE_COUNT)
        printf("Something unexpected happened 3\n");
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
    auto coord = r_collisionsPtr->coordinateToIndex(oppositePixelIndex);
    if (coord >= 1024 * 1024 || coord < 0)
        printf("Something unexpected happened 1\n");
    if (!isWithinBounds(coord, r_collisionsPtr->size()))
        printf("Something unexpected happened 5\n");
    const uint32_t collisionValue = (*r_collisionsPtr)[oppositePixelIndex];
    uint8_t newPlateId;
    const uint8_t oppositePlateId = MAX_PLATE_COUNT - collisionValue & 0xFF;
    if (collisionValue != 0 && collisionValue <= 0xFF && oppositePlateId != previousPlateId)
    {
        // If there is no convergence/divergence, we extract the sub-pixel center of the opposite pixel

        const uint8_t minPlate = previousPlateId < oppositePlateId ? previousPlateId : oppositePlateId;
        const uint8_t maxPlate = previousPlateId < oppositePlateId ? oppositePlateId : previousPlateId;

        const unsigned int bitIndex = upperTriangleIndexUnchecked(minPlate, maxPlate, MAX_PLATE_COUNT);
        const unsigned int wordIndex = bitIndex / 32;
        const unsigned int bitOffset = bitIndex % 32;
        if (wordIndex >= 1013)
            printf("Something unexpected happened 2\n");
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
    const Vec2 pixelOffset = {static_cast<int>(floor(offset.x)), static_cast<int>(floor(offset.y))};
    return currentTexIndex - pixelOffset;
}

__device__ __inline__ void applyElasticVelocityChange(CollisionVelocityChanges *w_velocityChanges,
                                                      const PlateData &currPlateData,
                                                      const Vec2<float> &finalVelocity, const float plateMass,
                                                      const float weight, const uint8_t plateID)
{
    const Vec2<float> velocityDiff = (finalVelocity - currPlateData.direction * currPlateData.velocity);
    const Vec2<float> weightedVelocityDiff = velocityDiff * weight;
    const float massDiff = (plateMass / currPlateData.mass);
    const Vec2<float> velocityChange = weightedVelocityDiff * massDiff;
    atomicAddVec2(&w_velocityChanges[plateID].inelasticDirectionalChange, velocityChange);

    // If weight is greater than full-subduction weight, there has been at least one continental pair
    // Using a branch-free method
    const float cond = weight > kernelSettings.inelasticCollisionMultiplierSubduction;
    const float coefficient = cond * kernelSettings.frictionCoefficientContinental +
                              (1.0f - cond) * kernelSettings.frictionCoefficientSubduction;

    const float friction = (currPlateData.velocity * coefficient) * massDiff;
    atomicAdd(&w_velocityChanges[plateID].frictionLoss, friction);
}


__device__ void applyInelasticCollision(const unsigned int *collisionTypes, CollisionVelocityChanges *w_velocityChanges,
                                        const uint8_t *plateIds, const float *previousHeights,
                                        const PlateData *plateIdsData)
{
    const uint8_t plateA = plateIds[0];
    const uint8_t plateB = plateIds[1];
    const uint8_t plateC = plateIds[2];
    const uint8_t plateD = plateIds[3];

    const float massA = previousHeights[0];
    const float massB = previousHeights[1];
    const float massC = previousHeights[2];
    const float massD = previousHeights[3];

    const PlateData &plateAData = plateIdsData[0];
    const PlateData &plateBData = plateIdsData[1];
    const PlateData &plateCData = plateIdsData[2];
    const PlateData &plateDData = plateIdsData[3];

    // Determine the weights
    float weights[4] = {};
    const int max_index = plateD != MAX_PLATE_COUNT ? 4 : plateC != MAX_PLATE_COUNT ? 3 : 2;

    for (int i = 0; i != max_index - 1; ++i)
        for (int j = i + 1; j != max_index; ++j)
        {
            const int index = (j * (j - 1)) / 2 + i;
            const unsigned int collisionType = collisionTypes[index];
            // Again, doing a branch-free if using a conditional mask
            const float denominator = static_cast<float>(max_index) - 1.0f;
            // True if any continental type
            const float cond = collisionType == 0 || collisionType == 3 || collisionType == 4;
            const float multiplier =
                    cond * (kernelSettings.inelasticCollisionMultiplierContinental / denominator) +
                    (1.0f - cond) * (kernelSettings.inelasticCollisionMultiplierSubduction * denominator);

            weights[i] += multiplier;
            weights[j] += multiplier;
        }

    const Vec2 numerator = plateAData.direction * (plateAData.velocity * massA * weights[0]) +
                           plateBData.direction * (plateBData.velocity * massB * weights[1]) +
                           plateCData.direction * (plateCData.velocity * massC * weights[2]) +
                           plateDData.direction * (plateDData.velocity * massD * weights[3]);

    const float denominator = weights[0] * massA + weights[2] * massB + weights[3] * massC + weights[4] * massD;

    const Vec2 finalVelocity = numerator / denominator;

    applyElasticVelocityChange(w_velocityChanges, plateAData, finalVelocity, massA, weights[0], plateA);
    applyElasticVelocityChange(w_velocityChanges, plateBData, finalVelocity, massB, weights[1], plateB);
    if (max_index > 2)
        applyElasticVelocityChange(w_velocityChanges, plateCData, finalVelocity, massC, weights[2], plateC);
    if (max_index > 3)
        applyElasticVelocityChange(w_velocityChanges, plateDData, finalVelocity, massD, weights[3], plateD);
}

__device__ void processConvergence(PlateTexturesWrite w_plateTextures, CudaTexture<float> *w_convergenceMapPtr,
                                   CudaTexture<uint8_t> *w_platesHaveCollidedPtr, const uint8_t *plateIds,
                                   const unsigned int *collisionTypes, const float *heights,
                                   const unsigned int invokeIndex)
{
    float force = 0;

    const int numCollidingPlates = plateIds[3] != MAX_PLATE_COUNT ? 4 : plateIds[2] != MAX_PLATE_COUNT ? 3 : 2;
    const int pairLen = numCollidingPlates == 4 ? 6 : numCollidingPlates == 3 ? 3 : 1;

    bool hasSubducted[4] = {false, false, false, false};
    constexpr int pairLookup[6][2] = {
        {0, 1},
        {0, 2}, {1, 2},
        {0, 3}, {1, 3}, {2, 3}
    };

    bool isMatched = false; // TODO: DEBUG REMOVE

    // Each pair registers the subducting plate in the array
    for (int accessPos = 0; accessPos != pairLen; ++accessPos)
    {
        const unsigned int collisionType = collisionTypes[accessPos];

        // TODO: DEBUG REMOVE
        if (plateIds[pairLookup[accessPos][0]] == 15 && plateIds[pairLookup[accessPos][1]] == 2)
            isMatched = true;
        // END

        if (collisionType == 1 || collisionType == 3)
            hasSubducted[pairLookup[accessPos][0]] = true;
        else if (collisionType == 2 || collisionType == 4)
            hasSubducted[pairLookup[accessPos][1]] = true;
    }

    // TODO: DEBUG REMOVE
    if (isMatched)
    {
        printf("Total %d: (%d, %d : %d), (%d, %d : %d), (%d, %d : %d)\n", invokeIndex,  plateIds[0], plateIds[1], collisionTypes[0], plateIds[0], plateIds[2], collisionTypes[1], plateIds[1], plateIds[2], collisionTypes[2]);
        printf("sub %d: %d %d %d %d\n", numCollidingPlates, hasSubducted[0], hasSubducted[1], hasSubducted[2], hasSubducted[3]);
    }

    // The first plate that has not subducted underneath another plate will gain the pixel ownership
    // In the case of no candidates, the first plate will gain owernship
    int newPlateOwnerIndex = 0;
    for (int i = 0; i != numCollidingPlates; ++i)
        if (!hasSubducted[i])
            newPlateOwnerIndex = i;

    uint8_t newPlateOwner = plateIds[newPlateOwnerIndex];
    float writeHeight = heights[newPlateOwnerIndex];
    float heightAddition = 0;

    for (int accessPos = 0; accessPos != pairLen; ++accessPos)
    {
        const auto [index_a, index_b] = pairLookup[accessPos];

        float height;
        if (index_a == newPlateOwnerIndex)
            height = heights[index_b];
        else if (index_b == newPlateOwnerIndex)
            height = heights[index_a];
        else
            continue;

        const unsigned int collisionType = collisionTypes[accessPos];

        if (height > writeHeight)
        {
            const float tmp = height;
            height = writeHeight;
            writeHeight = tmp;
        }

        heightAddition += collisionType == 0 ? 0.4f * height : 0.025f * height;

        if (index_a == newPlateOwnerIndex && collisionType == 4)
        {
            //TODO: Mark the adjacent pixel as accreted
            printf("Accretion1: %d %d %d %d\n", plateIds[newPlateOwnerIndex], plateIds[index_a], plateIds[index_b], collisionType);
        }
        else if (index_b == newPlateOwnerIndex && collisionType == 3)
        {
            printf("Accretion2: %d %d %d %d\n", plateIds[newPlateOwnerIndex], plateIds[index_a], plateIds[index_b], collisionType);
        }
    }


    (*w_plateTextures.heightMapPtr)[invokeIndex] = writeHeight;

    (*w_convergenceMapPtr)[invokeIndex] = 1;
    (*w_plateTextures.plateIdsPtr)[invokeIndex] = newPlateOwner;

    // Reporting the collision to the platesHaveCollided texture
    for (int accessPos = 0; accessPos != pairLen; ++accessPos)
    {
        const auto [index_a, index_b] = pairLookup[accessPos];
        const Vec2 accessVec = {index_a, index_b};

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

__device__ unsigned int determineCollisionType(const uint8_t *r_collisionTypeBitmap, const uint8_t plateA,
                                               const uint8_t plateB, const float heightA, const float heightB)
{
    const bool cond = plateA < plateB;

    const uint8_t minPlate = cond ? plateA : plateB;
    const uint8_t maxPlate = cond ? plateB : plateA;
    const float minHeight = cond ? heightA : heightB;
    const float maxHeight = cond ? heightB : heightA;

    const unsigned int index = upperTriangleIndexUnchecked(minPlate, maxPlate, MAX_PLATE_COUNT);

    const uint8_t collisionTypePacked = r_collisionTypeBitmap[index];
    const bool hasType = collisionTypePacked >> 0 & 1;
    const bool type = collisionTypePacked >> 1 & 1;
    const bool polarity = collisionTypePacked >> 3 & 1;

    // 0: Continental, 1: A under B, 2: B under A, 3: A under B Accretion, 4: B under A accretion
    unsigned int output = 0;
    if (hasType && type == 0)
    {
        if (polarity == 1)
        {
            if (minHeight >= kernelSettings.continentalCrustThreshold)
                output = minPlate == plateA ? 3 : 4;
            else
                output = minPlate == plateA ? 1 : 2;
        }

        if (polarity == 0)
        {
            if (maxHeight >= kernelSettings.continentalCrustThreshold)
                output = maxPlate == plateA ? 3 : 4;
            else
                output = maxPlate == plateA ? 1 : 2;
        }
    }

    if (minPlate == 2 && maxPlate == 15)
    {
        printf("Type Output: %d %d %d\n", plateA, plateB, output);
    }

    return output;
}

__device__ void extractCollisionTypes(const uint8_t *r_collisionTypeBitmap, unsigned int *collisionTypes,
                                      const uint8_t *plateIds, const float *previousHeights)
{
    const uint8_t plateA = plateIds[0];
    const uint8_t plateB = plateIds[1];
    const float heightA = previousHeights[0];
    const float heightB = previousHeights[1];

    collisionTypes[0] = determineCollisionType(r_collisionTypeBitmap, plateA, plateB, heightA, heightB);
    const uint8_t plateC = plateIds[2];
    if (plateC == MAX_PLATE_COUNT)
        return;
    const float heightC = previousHeights[2];
    collisionTypes[1] = determineCollisionType(r_collisionTypeBitmap, plateA, plateC, heightA, heightC);
    collisionTypes[2] = determineCollisionType(r_collisionTypeBitmap, plateB, plateC, heightB, heightC);
    const uint8_t plateD = plateIds[3];
    if (plateD == MAX_PLATE_COUNT)
        return;
    const float heightD = previousHeights[3];


    collisionTypes[3] = determineCollisionType(r_collisionTypeBitmap, plateA, plateD, heightA, heightD);
    collisionTypes[4] = determineCollisionType(r_collisionTypeBitmap, plateB, plateD, heightB, heightD);
    collisionTypes[5] = determineCollisionType(r_collisionTypeBitmap, plateC, plateD, heightC, heightD);
}

__global__ void processCollisions(const PlateTexturesRead r_plateTextures, const CudaTexture<uint32_t> *r_collisionsPtr,
                                  const uint32_t *r_divergenceBitmap,
                                  const uint8_t *r_collisionTypeBitmap, PlateTexturesWrite w_plateTextures,
                                  uint32_t *w_hasDivergedBitmap, CudaTexture<float> *w_convergenceMapPtr,
                                  CudaTexture<uint8_t> *w_platesHaveCollidedPtr,
                                  CollisionVelocityChanges *w_velocityChanges,
                                  const NoiseParameters noiseParameters)
{
    // Cache dereference since used more than once
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    // Ensure within bounds
    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(invokeIndex, r_collisions.size()))
        return;

    if (const uint32_t collisionValue = r_collisions[invokeIndex]; collisionValue == 0)
    {
        // Collision value is zero, indicating no plate moved onto this pixel. Thus, this pixel is a divergence zone.
        processDivergence(r_plateTextures, w_plateTextures, r_collisionsPtr, r_divergenceBitmap,
                          w_hasDivergedBitmap, invokeIndex, noiseParameters);
    } else
    {
        const Vec2 texCoords = r_collisions.indexToCoordinate(invokeIndex);

        const uint8_t plateIds[4] = {
            static_cast<uint8_t>(MAX_PLATE_COUNT - (collisionValue & 0xFF)),
            static_cast<uint8_t>(MAX_PLATE_COUNT - ((collisionValue >> 8) & 0xFF)),
            static_cast<uint8_t>(MAX_PLATE_COUNT - ((collisionValue >> 16) & 0xFF)),
            static_cast<uint8_t>(MAX_PLATE_COUNT - ((collisionValue >> 24) & 0xFF))
        };

        // If plateB is not undefined, it means at least two plates move into the same pixel, indicating a convergent boundary.
        if (plateIds[1] != MAX_PLATE_COUNT)
        {
            const PlateData plateIdsData[4] = {
                r_plateTextures.plateData[plateIds[0]],
                r_plateTextures.plateData[plateIds[1]],
                plateIds[2] != MAX_PLATE_COUNT ? r_plateTextures.plateData[plateIds[2]] : PlateData(),
                plateIds[3] != MAX_PLATE_COUNT ? r_plateTextures.plateData[plateIds[3]] : PlateData()
            };

            const CudaTexture<float> &r_heightMap = *r_plateTextures.heightMapPtr;
            const float previousHeights[4] = {
                r_heightMap[previousTextureIndex(texCoords, plateIdsData[0])],
                r_heightMap[previousTextureIndex(texCoords, plateIdsData[1])],
                plateIds[2] != MAX_PLATE_COUNT ? r_heightMap[previousTextureIndex(texCoords, plateIdsData[2])] : 0,
                plateIds[3] != MAX_PLATE_COUNT ? r_heightMap[previousTextureIndex(texCoords, plateIdsData[3])] : 0
            };

            if (previousHeights[0] == 0)
            {
                printf("Zero mass\n");
            }
            unsigned int collisionTypes[6];
            extractCollisionTypes(r_collisionTypeBitmap, collisionTypes, plateIds, previousHeights);

            applyInelasticCollision(collisionTypes, w_velocityChanges, plateIds, previousHeights, plateIdsData);
            processConvergence(w_plateTextures, w_convergenceMapPtr, w_platesHaveCollidedPtr, plateIds,
                               collisionTypes, previousHeights, invokeIndex);
        }
        // Otherwise, only one plate moves into the pixel, indicating ordinary movement
        else
        {
            processMovement(r_plateTextures, w_plateTextures, plateIds[0], invokeIndex);
        }
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
                               const uint8_t *r_collisionTypeBitmap,
                               CollisionTypeCounts *w_collisionTypeCounts)
{
    const bool cond = plateA < plateB;

    // Minimums
    const uint8_t minPlate = cond ? plateA : plateB;
    const float minPlateHeight = cond ? heightA : heightB;

    // Maximums
    const uint8_t maxPlate = cond ? plateB : plateA;
    const float maxPlateHeight = cond ? heightB : heightA;

    const unsigned int bitIndex = upperTriangleIndexUnchecked(minPlate, maxPlate, MAX_PLATE_COUNT);
    const unsigned int wordIndex = bitIndex * 2 / 32;
    const unsigned int bitOffset = bitIndex * 2 % 32;

    const uint8_t collisionTypePacked = r_collisionTypeBitmap[wordIndex];

    // Exit if a type already exists and is not to be updated yet
    if (collisionTypePacked >> 0 & 1 && collisionTypePacked >> 4 & 0x0F) // TODO: add settingKernel
        return;

    const bool minCont = minPlateHeight >= kernelSettings.continentalCrustThreshold;
    const bool maxCont = maxPlateHeight >= kernelSettings.continentalCrustThreshold;

    const float minWeight = minCont ? minPlateHeight * 5 : minPlateHeight;
    const float maxWeight = maxCont ? maxPlateHeight * 5 : maxPlateHeight;

    atomicAdd(&w_collisionTypeCounts[bitIndex].weightedHeightA, minWeight);
    atomicAdd(&w_collisionTypeCounts[bitIndex].weightedHeightB, maxWeight);

    if (minCont && maxCont)
    {
        if (minPlate == 2 && maxPlate == 15)
            printf("Continental (%d %d) %f %f\n", minPlate, maxPlate, minPlateHeight, maxPlateHeight);
        atomicAdd(&w_collisionTypeCounts[bitIndex].continental, 1);
    } else if (minCont)
    {
        if (minPlate == 2 && maxPlate == 15)
            printf("SubductionBContinental (%d %d) %f %f\n", minPlate, maxPlate, minPlateHeight, maxPlateHeight);
        atomicAdd(&w_collisionTypeCounts[bitIndex].subductionsBContinental, 1);
    } else if (maxCont)
    {
        if (minPlate == 2 && maxPlate == 15)
            printf("SubductionAContinental (%d %d) %f %f\n", minPlate, maxPlate, minPlateHeight, maxPlateHeight);
        atomicAdd(&w_collisionTypeCounts[bitIndex].subductionsAContinental, 1);
    } else if (minPlateHeight <= maxPlateHeight)
    {
        if (minPlate == 2 && maxPlate == 15)
            printf("SubductionAOceanic (%d %d) %f %f\n", minPlate, maxPlate, minPlateHeight, maxPlateHeight);
        atomicAdd(&w_collisionTypeCounts[bitIndex].subductionsAOceanic, 1);
    } else
    {
        if (minPlate == 2 && maxPlate == 15)
            printf("SubductionBOceanic (%d %d) %f %f\n", minPlate, maxPlate, minPlateHeight, maxPlateHeight);
        atomicAdd(&w_collisionTypeCounts[bitIndex].subductionsBOceanic, 1);
    }
}

__global__ void determineCollisionType(const PlateTexturesRead r_plateTextures,
                                       const CudaTexture<uint32_t> *r_collisionsPtr,
                                       const uint8_t *r_collisionTypeBitmap,
                                       CollisionTypeCounts *w_collisionTypeCounts)
{
    const unsigned int invokeIndex = getInvokeIndex();
    const CudaTexture<float> &r_heightMap = *r_plateTextures.heightMapPtr;
    if (!isWithinBounds(invokeIndex, r_heightMap.size()))
        return;

    const uint32_t collisionsPacked = (*r_collisionsPtr)[invokeIndex];

    // Exit if there is no collision between plates
    if (collisionsPacked <= 0xFF)
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

    const float heightA = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateA, textureIndex);
    const float heightB = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateB, textureIndex);
    if (plateD != MAX_PLATE_COUNT)
    {
        const float heightC = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateC, textureIndex);
        const float heightD = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateD, textureIndex);
        writeCollision(heightA, heightB, plateA, plateB, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightA, heightC, plateA, plateC, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightA, heightD, plateA, plateD, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightB, heightC, plateB, plateC, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightB, heightD, plateB, plateD, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightC, heightD, plateC, plateD, r_collisionTypeBitmap, w_collisionTypeCounts);
    } else if (plateC != MAX_PLATE_COUNT)
    {
        const float heightC = determineCollisionHeight(r_heightMap, r_plateTextures.plateData, plateC, textureIndex);
        writeCollision(heightA, heightB, plateA, plateB, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightA, heightC, plateA, plateC, r_collisionTypeBitmap, w_collisionTypeCounts);
        writeCollision(heightB, heightC, plateB, plateC, r_collisionTypeBitmap, w_collisionTypeCounts);
    } else
        writeCollision(heightA, heightB, plateA, plateB, r_collisionTypeBitmap, w_collisionTypeCounts);
}

__global__ void createCollisionTypeMatrix(const CollisionTypeCounts *r_collisionTypeCounts,
                                          uint8_t *rw_collisionTypeBitmap)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= MAX_PLATE_COUNT * (MAX_PLATE_COUNT - 1) / 2)
        return;


    // Do some weird calculations to get plateA and plateB from the diagonal excluding upper triangle matrix
    // Had to ask chatgpt for this one
    constexpr auto a = static_cast<double>(2 * MAX_PLATE_COUNT - 1);
    const double discriminant = a * a - 8.0 * static_cast<float>(invokeIndex);
    const double root = sqrt(discriminant);
    const auto plateA = static_cast<uint8_t>((a - root) * 0.5f);
    const unsigned int base = plateA * (2 * MAX_PLATE_COUNT - plateA - 1) / 2;
    const auto plateB = static_cast<uint8_t>(plateA + 1 + (invokeIndex - base));

    const unsigned int collisionTypePacked = rw_collisionTypeBitmap[invokeIndex];
    const bool hasType = (collisionTypePacked >> 0) & 1;
    const uint8_t collisionCount = collisionTypePacked >> 4 & 0x0F;

    // Extract the count of overall colliding pixels and count of continental pixels for A
    const CollisionTypeCounts collisionTypeCounts = r_collisionTypeCounts[invokeIndex];

    const uint32_t totalCollisionSize = collisionTypeCounts.subductionsAContinental + collisionTypeCounts.
                                        subductionsAOceanic + collisionTypeCounts.subductionsBContinental +
                                        collisionTypeCounts.subductionsBOceanic + collisionTypeCounts.continental;

    // Exit if there is already a registered collision type
    if (hasType and collisionCount < 4) // TODO: Change to kernelSetting
    {
        // Increment the collisionCount if a type has been assigned already
        if (totalCollisionSize > 0)
            rw_collisionTypeBitmap[invokeIndex] = collisionTypePacked & 0x0F | (collisionCount + 1 & 0x0F) << 4;
        return;
    }

    // No type is assigned if a collision is not large enough
    if (totalCollisionSize < 50) // TODO: Change to kernelSetting
    {
        return;
    }

    // Clear the counter
    uint8_t output = collisionTypePacked & 0x0F;
    const bool hasPolarity = (collisionTypePacked >> 2) & 1;
    const bool polarityASubducting = collisionTypePacked >> 3 & 1;

    const int continentalOpposingA = collisionTypeCounts.subductionsBContinental + collisionTypeCounts.continental;
    const int continentalOpposingB = collisionTypeCounts.subductionsAContinental + collisionTypeCounts.continental;
    const int subductionAPower = collisionTypeCounts.subductionsAContinental + collisionTypeCounts.subductionsAOceanic -
                                 continentalOpposingA;
    const int subductionBPower = collisionTypeCounts.subductionsBContinental + collisionTypeCounts.subductionsBOceanic -
                                 continentalOpposingB;


    // If continental->continental collisions are the most occurring ones, the collision is continental
    if (collisionTypeCounts.continental >= collisionTypeCounts.subductionsAContinental + collisionTypeCounts.
        subductionsAOceanic &&
        collisionTypeCounts.continental >= collisionTypeCounts.subductionsBContinental + collisionTypeCounts.
        subductionsBOceanic)
    {
        // HasType is set to true, and type is set to 1 (continental)
        output |= 0b00000011;
        printf("CONTINENTAL %d %d %d %d %d %d %d %d %f %f %d       %d%d%d%d%d%d%d%d\n", plateA, plateB,
               collisionTypeCounts.subductionsAContinental,
               collisionTypeCounts.subductionsAOceanic, collisionTypeCounts.subductionsBContinental,
               collisionTypeCounts.subductionsBOceanic, collisionTypeCounts.continental, totalCollisionSize,
               collisionTypeCounts.weightedHeightA, collisionTypeCounts.weightedHeightB, invokeIndex,
               (output >> 7) & 1,
               (output >> 6) & 1,
               (output >> 5) & 1,
               (output >> 4) & 1,
               (output >> 3) & 1,
               (output >> 2) & 1,
               (output >> 1) & 1,
               (output >> 0) & 1);
    } else if (subductionAPower > 0 && subductionAPower > subductionBPower && (!hasPolarity || polarityASubducting))
    {
        // HasType and HasPolarity is set to true, and type is set to 0 (subduction), and polarity set to 1 (Plate A Subducting)
        output = output & ~0b00000010 | 0b00001101;
        printf("SubductionA %d %d %d %d %d %d %d %d %f %f %d       %d%d%d%d%d%d%d%d\n", plateA, plateB,
               collisionTypeCounts.subductionsAContinental,
               collisionTypeCounts.subductionsAOceanic, collisionTypeCounts.subductionsBContinental,
               collisionTypeCounts.subductionsBOceanic, collisionTypeCounts.continental, totalCollisionSize,
               collisionTypeCounts.weightedHeightA, collisionTypeCounts.weightedHeightB, invokeIndex,
               (output >> 7) & 1,
               (output >> 6) & 1,
               (output >> 5) & 1,
               (output >> 4) & 1,
               (output >> 3) & 1,
               (output >> 2) & 1,
               (output >> 1) & 1,
               (output >> 0) & 1);
    } else if (subductionBPower > 0 && subductionBPower > subductionAPower && (!hasPolarity || !polarityASubducting))
    {
        // HasType and HasPolarity is set to true, and type is set to 1 (subduction), and polarity set to 0 (Plate B Subducting)
        output = output & ~0b00001010 | 0b00000101;
        printf("SubductionB %d %d %d %d %d %d %d %d %f %f %d       %d%d%d%d%d%d%d%d\n", plateA, plateB,
               collisionTypeCounts.subductionsAContinental,
               collisionTypeCounts.subductionsAOceanic, collisionTypeCounts.subductionsBContinental,
               collisionTypeCounts.subductionsBOceanic, collisionTypeCounts.continental, totalCollisionSize,
               collisionTypeCounts.weightedHeightA, collisionTypeCounts.weightedHeightB, invokeIndex,
               (output >> 7) & 1,
               (output >> 6) & 1,
               (output >> 5) & 1,
               (output >> 4) & 1,
               (output >> 3) & 1,
               (output >> 2) & 1,
               (output >> 1) & 1,
               (output >> 0) & 1);
    }


    rw_collisionTypeBitmap[invokeIndex] = output;
}

__global__ void copyPlateDataGuiKernel(const PlateData *r_plateData, const uint8_t *r_collisionTypeBitmap,
                                       GuiPlateData *w_plateDataGui)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= MAX_PLATE_COUNT)
        return;

    const PlateData plateData = r_plateData[invokeIndex];

    w_plateDataGui[invokeIndex].mass = plateData.mass;
    w_plateDataGui[invokeIndex].size = plateData.size;
    w_plateDataGui[invokeIndex].hasMoved = plateData.hasMoved;
    w_plateDataGui[invokeIndex].velocity = plateData.velocity;
    w_plateDataGui[invokeIndex].direction = plateData.direction;
    w_plateDataGui[invokeIndex].break_score = plateData.breakScore;
    w_plateDataGui[invokeIndex].perimeter = plateData.perimeter;
    w_plateDataGui[invokeIndex].circularity = plateData.circularity;

    int subduction_idx = 0;
    int continental_idx = 0;

    for (int i = 0; i != MAX_PLATE_COUNT; ++i)
    {
        if (i == invokeIndex)
            continue;

        const uint8_t minPlate = invokeIndex < i ? invokeIndex : i;
        const uint8_t maxPlate = invokeIndex < i ? i : invokeIndex;


        const unsigned int entryIndex = upperTriangleIndexUnchecked(minPlate, maxPlate, MAX_PLATE_COUNT);

        const uint8_t collisionTypePacked = r_collisionTypeBitmap[entryIndex];

        const bool hasType = collisionTypePacked >> 0 & 1;
        const bool typeIsContinental = collisionTypePacked >> 1 & 1;
        const bool polarity = collisionTypePacked >> 3 & 1;

        if (!hasType)
            continue;

        if (typeIsContinental)
        {
            w_plateDataGui[invokeIndex].continental[continental_idx++] = minPlate == invokeIndex ? maxPlate : minPlate;
        }
        else
        {
            if (polarity)
                w_plateDataGui[invokeIndex].subductions[subduction_idx++] = minPlate == invokeIndex
                                                                                ? maxPlate
                                                                                : -minPlate;
            else
                w_plateDataGui[invokeIndex].subductions[subduction_idx++] = minPlate == invokeIndex
                                                                                ? -maxPlate
                                                                                : minPlate;
        }
    }

    w_plateDataGui[invokeIndex].continental_len = continental_idx;
    w_plateDataGui[invokeIndex].subduction_len = subduction_idx;
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


// Generic vertical distance field function - can be used for any float texture with plate constraints
template<typename ValidatorFunc>
__device__ DistanceFieldBuffer verticalDistanceFieldPass(const CudaTexture<float> &sourceTexture, 
                                      const Vec2<int> &center, 
                                      int range, 
                                      ValidatorFunc validator)
{
    const float multiplier = 1.0f / range;
    float bestValue = 0;
    int bestOffset = 0;
    
    for (int i = -range; i <= range; i++)
    {
        Vec2<int> sample = center + Vec2<int>(0, i);
        float sampleValue = sourceTexture[sample] - (abs((float)i) * multiplier);
        
        if (sampleValue > bestValue && validator(sample, i))
        {
            bestValue = sampleValue;
            bestOffset = i;
        }
    }
    
    return DistanceFieldBuffer(bestOffset, bestValue);
}

// Generic horizontal distance field function - can be used for any blur buffer with plate constraints
template<typename ValidatorFunc>
__device__ float horizontalDistanceFieldPass(const CudaTexture<DistanceFieldBuffer> &bufferTexture,
                                   const Vec2<int> &center,
                                   int range,
                                   ValidatorFunc validator)
{
    const float multiplier = 1.0f / range;
    float bestValue = 0;
    
    for (int i = -range; i <= range; i++)
    {
        Vec2<int> sample = center + Vec2<int>(i, 0);
        DistanceFieldBuffer buffer = bufferTexture[sample];
        float adjustedValue = buffer.value - abs((float)i) * multiplier;
        
        if (adjustedValue > bestValue && validator(sample, i))
        {
            bestValue = adjustedValue;
        }
    }
    
    return bestValue;
}

// Generic vertical blur function - performs gaussian-style smoothing with plate constraints
template<typename ValidatorFunc>
__device__ float verticalBlurPass(const CudaTexture<float> &sourceTexture, 
                                  const Vec2<int> &center, 
                                  int range, 
                                  ValidatorFunc validator)
{
    float sum = 0.0f;
    float weightSum = 0.0f;
    
    for (int i = -range; i <= range; i++)
    {
        Vec2<int> sample = center + Vec2<int>(0, i);
        
        if (validator(sample, i))
        {
            // Use linear weight
            float weight = getLinearWeight(i, range);
            sum += sourceTexture[sample] * weight;
            weightSum += weight;
        }
    }
    
    return weightSum > 0.0f ? sum / weightSum : sourceTexture[center];
}

// Generic horizontal blur function - performs gaussian-style smoothing with plate constraints
template<typename ValidatorFunc>
__device__ float horizontalBlurPass(const CudaTexture<float> &sourceTexture,
                                    const Vec2<int> &center,
                                    int range,
                                    ValidatorFunc validator)
{
    float sum = 0.0f;
    float weightSum = 0.0f;
    
    for (int i = -range; i <= range; i++)
    {
        Vec2<int> sample = center + Vec2<int>(i, 0);
        
        if (validator(sample, i))
        {
            // Use linear weight
            float weight = getLinearWeight(i, range);
            sum += sourceTexture[sample] * weight;
            weightSum += weight;
        }
    }
    
    return weightSum > 0.0f ? sum / weightSum : sourceTexture[center];
}

__global__ void VerticalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                             const CudaTexture<float> *r_upliftMapPtr,
                             CudaTexture<DistanceFieldBuffer> *w_bufferPtr)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    const CudaTexture<float> &r_uplift = *r_upliftMapPtr;
    CudaTexture<DistanceFieldBuffer> &w_buffer = *w_bufferPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_uplift.size()))
        return;

    const Vec2<int> center = r_uplift.indexToCoordinate(invokeIndex);
    const uint8_t plateId = r_plateIds[invokeIndex];

    // Create validator lambda for uplift distance field
    auto validator = [&](const Vec2<int> &sample, int offset) -> bool {
        return collisionContains(r_collisions[sample], plateId);
    };

    w_buffer[invokeIndex] = verticalDistanceFieldPass(r_uplift, center, kernelSettings.upliftRange, validator);
}

__global__ void HorizontalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                               const CudaTexture<DistanceFieldBuffer> *r_bufferPtr,
                               CudaTexture<float> *w_heightMapPtr)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    const CudaTexture<DistanceFieldBuffer> &r_buffer = *r_bufferPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_buffer.size()))
        return;

    const Vec2<int> center = r_buffer.indexToCoordinate(invokeIndex);
    const uint8_t plateId = r_plateIds[invokeIndex];

    // Create validator lambda for uplift distance field
    auto validator = [&](const Vec2<int> &sample, int offset) -> bool {
        return collisionContains(r_collisions[sample], plateId);
    };

    float value = horizontalDistanceFieldPass(r_buffer, center, kernelSettings.upliftRange, validator);
    (*w_heightMapPtr)[invokeIndex] += value * kernelSettings.upliftMultiplier;
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

__global__ void applyPlateMovementChanges(PlateData *rw_plateLookup, const CollisionVelocityChanges *r_velocityChanges)
{
    if (threadIdx.x >= MAX_PLATE_COUNT)
        return;

    PlateData current = rw_plateLookup[threadIdx.x];

    const Vec2 center = current.pixelCenter + current.direction * current.velocity;
    const Vec2 newPixelCenter = {
        center.x - floor(center.x),
        center.y - floor(center.y)
    };

    current.hasMoved = newPixelCenter != center;

    current.pixelCenter = newPixelCenter;

    const auto [inelasticDirectionalChange, frictionLoss] = r_velocityChanges[threadIdx.x];

    const Vec2<float> originalVelocityVector = current.direction * current.velocity;
    const Vec2<float> newVelocityVector = originalVelocityVector + (current.forceFromPressure / max(current.mass, 1.0f) * 0.01f);
    
    float x = (current.forceFromPressure / max(current.mass, 1.0f)).magnitude();

    if(x > 0.00001f)
        printf("speed increase: %.4f \n", (current.forceFromPressure / max(current.mass, 1.0f)).magnitude());

    const auto [norm, mag] = newVelocityVector.normalizedAndMagnitudeZeroSafe();

    current.direction = norm;
    const float postCollisionVelocity = max(mag - frictionLoss, 0.f);
    current.velocity = clamp01(postCollisionVelocity * (1 - kernelSettings.environmentalDragCoefficient));

    current.mass = 0;
    current.size = 0;
    current.perimeter = 0;
    current.breakScore = 0;
    current.used = false;

    rw_plateLookup[threadIdx.x] = current;
}

__global__ void updatePlateData(PlateData *plateLookup, const CollisionVelocityChanges *r_velocityChanges,
                                const float *r_plateMass,
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

    const Vec2<float> newVelocityVector = current.direction * current.velocity + r_velocityChanges[threadIdx.x].
                                          inelasticDirectionalChange;

    const auto [norm, mag] = newVelocityVector.normalizedAndMagnitudeZeroSafe();

    current.velocity = mag * (1 - kernelSettings.environmentalDragCoefficient);
    current.direction = norm;

    current.mass = r_plateMass[threadIdx.x];
    current.size = r_plateSize[threadIdx.x]; // Will be updated in the next pixel kernel.
    current.used = false;

    plateLookup[threadIdx.x] = current;
}

__global__ void determinePlateMerge(const CudaTexture<uint8_t> *r_platesHaveCollided, const PlateData *r_plateLookup,
                                    uint8_t *w_plateMergeIds)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(invokeIndex, r_platesHaveCollided->size()))
        return;

    __shared__ PlateData sharedPlateLookup[255];

    if (threadIdx.x < 255)
        sharedPlateLookup[threadIdx.x] = r_plateLookup[threadIdx.x];

    __syncthreads();

    // Check if the current plate pair has a collision
    if (!(*r_platesHaveCollided)[invokeIndex])
        return;

    const Vec2<int> texIndex = getTextureIndex(invokeIndex, r_platesHaveCollided->size());
    const PlateData &plateDataA = sharedPlateLookup[texIndex.x];
    const PlateData &plateDataB = sharedPlateLookup[texIndex.y];

    const float dot = plateDataA.direction.dot(plateDataB.direction);
    const float velocity_diff = abs(plateDataA.velocity - plateDataB.velocity);

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

__global__ void accumulatePlateAngularCoords(const CudaTexture<uint8_t> *r_plateIdsPtr, CudaTexture<float4> *w_plateAngularSumsPtr, CudaTexture<int> *w_plateCountsPtr)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    CudaTexture<float4> &w_plateAngularSums = *w_plateAngularSumsPtr;
    CudaTexture<int> &w_plateCounts = *w_plateCountsPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIds.size()))
        return;

    // Sample every 10th pixel for efficiency (sparse sampling)
    const Vec2<int> coord = r_plateIds.indexToCoordinate(invokeIndex);
    if (coord.x % 10 == 0 && coord.y % 10 == 0)
    {
        const uint8_t plateId = r_plateIds[invokeIndex];
        const Vec2<float> floatCoord = Vec2<float>(coord.x, coord.y);

        // Convert to angular coordinates for wrapping support
        float angleX = CURAND_2PI * floatCoord.x / r_plateIds.size().x;
        float angleY = CURAND_2PI * floatCoord.y / r_plateIds.size().y;

        float sinX = sinf(angleX);
        float cosX = cosf(angleX);
        float sinY = sinf(angleY);
        float cosY = cosf(angleY);

        // Accumulate angular coordinates for this plate
        atomicAdd(&w_plateAngularSums[plateId].x, sinX);
        atomicAdd(&w_plateAngularSums[plateId].y, cosX);
        atomicAdd(&w_plateAngularSums[plateId].z, sinY);
        atomicAdd(&w_plateAngularSums[plateId].w, cosY);
        atomicAdd(&w_plateCounts[plateId], 1);
    }
}

__global__ void calculatePlateCenters(const CudaTexture<uint8_t> *r_plateIdsPtr,
                                    const CudaTexture<float4> *r_plateAngularSumsPtr, const CudaTexture<int> *r_plateCountsPtr,
                                    PlateData *w_plateData)
{
    const unsigned int plateId = getInvokeIndex();
    if (plateId >= MAX_PLATE_COUNT)
        return;

    const CudaTexture<float4> &r_plateAngularSums = *r_plateAngularSumsPtr;
    const CudaTexture<int> &r_plateCounts = *r_plateCountsPtr;
    const CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;

    // Only process plates that have samples
    if (r_plateCounts[plateId] > 0)
    {
        const float4 angularSums = r_plateAngularSums[plateId];
        const int count = r_plateCounts[plateId];

        float4 avgAngular;
        avgAngular.x = angularSums.x / count;
        avgAngular.y = angularSums.y / count;
        avgAngular.z = angularSums.z / count;
        avgAngular.w = angularSums.w / count;

        // Convert back to Cartesian coordinates
        float centerX = atan2f(avgAngular.x, avgAngular.y) * r_plateIds.size().x / CURAND_2PI;
        float centerY = atan2f(avgAngular.z, avgAngular.w) * r_plateIds.size().y / CURAND_2PI;

        // Handle negative angles
        if (centerX < 0) centerX += r_plateIds.size().x;
        if (centerY < 0) centerY += r_plateIds.size().y;

        // Store in geometricCenter field
        w_plateData[plateId].geometricCenter = Vec2<float>(centerX, centerY);
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

__global__ void findPlausibleSplitLine(const uint8_t highestStressPlateId, const Vec2<float> r_pivot,
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
    uint8_t foundPlate = highestStressPlateId;
    Vec2<float> point = r_pivot + dir * stepsize;
    float distance = stepsize;

    int iterations = 0;
    while (iterations < 100 && stepsize > minStepsize)
    {
        foundPlate = r_plateIds[Vec2<int>(point.x, point.y)];
        if (foundPlate == highestStressPlateId)
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

        printf("PlateId: %.i score: %.2f pivot: %.2f %.2f \n", highestStressPlateId, best, output->x, output->y);
    }
}

__global__ void splitPlate(const uint8_t highestStressPlateId, const uint8_t *r_newPlateId, const Vec2<float> r_pivot,
                           const Vec2<float> *r_dir, CudaTexture<uint8_t> *w_plateIdsPtr, PlateData *plateLookup)
{
    const unsigned int invokeIndex = getInvokeIndex();

    CudaTexture<uint8_t> &w_plateIds = *w_plateIdsPtr;

    if (invokeIndex == 0)
    {
        printf("splitting from %.i to %.i \n", highestStressPlateId, *r_newPlateId);

        //plateLookup[*r_newPlateId].direction += Vec2<float>(r_dir->y, -r_dir->x);
        //plateLookup[*r_newPlateId].direction = plateLookup[*r_newPlateId].direction.normalized();

        plateLookup[*r_newPlateId].mass = plateLookup[highestStressPlateId].mass * 0.5;
        plateLookup[*r_newPlateId].size = plateLookup[highestStressPlateId].size * 0.5;
        //plateLookup[*r_newPlateId].velocity = plateLookup[highestStressPlateId].velocity * 1.1;

        //plateLookup[highestStressPlateId].direction += Vec2<float>(-r_dir->y, r_dir->x);
        //plateLookup[highestStressPlateId].direction = plateLookup[highestStressPlateId].direction.normalized();

        plateLookup[highestStressPlateId].mass *= 0.5;
        plateLookup[highestStressPlateId].size *= 0.5;

        //plateLookup[highestStressPlateId].velocity *= 1.1;
    }

    if (w_plateIds[invokeIndex] == highestStressPlateId)
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

        //float noise = cudaNoise::perlinNoise(make_float3(distanceOnLine * 0.1, 0, 0), 1, 123124);


        //float offset = round(sin(distanceOnLine * breakFreq * noise)) * breakScale + sin(distanceOnLine * CurveFreq) *
        //               CurveScale;

        if ((distanceToLine > 0) ^ (distanceToLine > halfDistance))
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

__global__ void evaporate(CudaTexture<float>* w_hydrationPtr, float deltatime)
{
    CudaTexture<float>& hydration = *w_hydrationPtr;
    unsigned int idx = getInvokeIndex();
    hydration[idx] = fmaxf(0.0f, hydration[idx] - (fminf(hydration[idx], 20) * kernelSettings.hydrationEvaporation * deltatime));
}

// Step 1: Accumulate pressure and reset at fault lines (using rain-like accumulation)
__global__ void pressureAccumulation(CudaTexture<float>* r_pressurePtr, CudaTexture<float>* w_pressurePtr, CudaTexture<uint8_t>* r_plateIdsPtr, CudaTexture<float>* r_materialPtr)
{
    CudaTexture<float>& r_pressure = *r_pressurePtr;
    CudaTexture<float>& w_pressure = *w_pressurePtr;
    CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;
    CudaTexture<float>& r_material = *r_materialPtr;

    unsigned int idx = getInvokeIndex();
    if (!isWithinBounds(idx, r_pressure.size()))
        return;

    Vec2<int> coord = getTextureIndex(r_pressurePtr->size());
    uint8_t plateId = r_plateIds[idx];

    // Check if this is a fault line (plate boundary)
    bool isFault = false;
    isFault = isFault || plateId != r_plateIds[coord + Vec2<int>(1, 0)];
    isFault = isFault || plateId != r_plateIds[coord + Vec2<int>(0, 1)];
    isFault = isFault || plateId != r_plateIds[coord + Vec2<int>(-1, 0)];
    isFault = isFault || plateId != r_plateIds[coord + Vec2<int>(0, -1)];

    if (isFault)
    {
        // Pressure is released at fault lines
        w_pressure[idx] = 0;
    }
    else
    {
        // Accumulate pressure like rain accumulates hydration
        w_pressure[idx] = r_pressure[idx] + kernelSettings.pressureAccumulation * r_material[idx];
    }
}

// Step 2: Vertical blur pass for pressure propagation
__global__ void pressureVerticalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, 
                                     const CudaTexture<float> *r_pressurePtr,
                                     CudaTexture<float> *w_bufferPtr)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<float> &r_pressure = *r_pressurePtr;
    CudaTexture<float> &w_buffer = *w_bufferPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_pressure.size()))
        return;

    const Vec2<int> center = r_pressure.indexToCoordinate(invokeIndex);
    const uint8_t plateId = r_plateIds[invokeIndex];

    // Pressure propagates within the same plate only
    auto validator = [&](const Vec2<int> &sample, int offset) -> bool {
        return r_plateIds[sample] == plateId;
    };

    w_buffer[invokeIndex] = verticalBlurPass(r_pressure, center, kernelSettings.pressureBlurRange, validator);
}

// Step 3: Horizontal blur pass for pressure propagation
__global__ void pressureHorizontalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, 
                                       const CudaTexture<float> *r_bufferPtr,
                                       CudaTexture<float> *w_pressurePtr)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    const CudaTexture<float> &r_buffer = *r_bufferPtr;
    CudaTexture<float> &w_pressure = *w_pressurePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_buffer.size()))
        return;

    const Vec2<int> center = r_buffer.indexToCoordinate(invokeIndex);
    const uint8_t plateId = r_plateIds[invokeIndex];

    // Pressure propagates within the same plate only
    auto validator = [&](const Vec2<int> &sample, int offset) -> bool {
        return r_plateIds[sample] == plateId;
    };

    float processedPressure = horizontalBlurPass(r_buffer, center, kernelSettings.pressureBlurRange, validator);
    
    // Apply the processed pressure with multiplier
    w_pressure[invokeIndex] = processedPressure * kernelSettings.pressureMultiplier;
}

__global__ void stress(const CudaTexture<float>* r_pressurePtr, const CudaTexture<float>* r_MaterialPtr, CudaTexture<float>* w_stressPtr, const CudaTexture<uint8_t>* r_plateIdsPtr, const PlateData* r_plateData)
{
    const CudaTexture<float>& r_pressure = *r_pressurePtr;
    const CudaTexture<float>& r_material = *r_MaterialPtr;
    CudaTexture<float>& w_stress = *w_stressPtr;
    const CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_pressure.size()))
        return;

    const Vec2<int> coord = getTextureIndex(w_stress.size());
    const uint8_t plateId = r_plateIds[invokeIndex];
    const PlateData plateData = r_plateData[plateId];
    
    float baseStress = r_pressure[invokeIndex] / max(min(r_material[invokeIndex], 200.0f), 1.0f);

    float distanceToCenter = plateData.geometricCenter.distance(Vec2<float>(coord.x, coord.y));
    float distanceScore = 1.0f / (0.01f * distanceToCenter + 1.0f);
    w_stress[invokeIndex] = baseStress * plateData.breakScore * distanceScore;
}

__global__ void computeBreakScore(PlateData *w_plateData)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= MAX_PLATE_COUNT)
        return;
        
    PlateData &plateData = w_plateData[invokeIndex];

    if (plateData.size > 0) {
        float area = static_cast<float>(plateData.size);
        float perimeter = static_cast<float>(plateData.perimeter);

        if (invokeIndex < 5) {
            printf("Plate %d: size=%d, perimeter=%d\n", invokeIndex, plateData.size, plateData.perimeter);
        }

        plateData.circularity = 1.0f - ((2 * CURAND_2PI * area) / (perimeter * perimeter));

        float areaIncrease = inverseClampedLerp(area, kernelSettings.targetMinimumPlateArea, kernelSettings.targetMaximumPlateArea);

        plateData.breakScore = plateData.circularity + areaIncrease; // circularity formula? need to include in research. 
    } else {
        plateData.breakScore = 0.0f;
    }
}

__global__ void thermalErosionKernel(CudaTexture<float> *w_materialPtr)
{
    CudaTexture<float> &w_material = *w_materialPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, w_material.size()))
        return;

    const Vec2<int> coord = w_material.indexToCoordinate(invokeIndex);

    bool valid[8];
    float diffs[8];
    float totalDiff = 0.0f;

    const Vec2<int> offsets[] = {
        {-1,  0}, {1, 0}, {0, -1}, {0, 1},   // N, S, W, E
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}   // NW, NE, SW, SE
    };

    for (int direction = 0; direction < 8; direction++)
    {
        const Vec2<int> neighborCoord = coord + offsets[direction];
        
        const float neighborHeight = w_material[neighborCoord];
        const float diff = w_material[invokeIndex] - neighborHeight;

        if (diff > kernelSettings.thermalThresholdAngle * kernelSettings.thermalCellSize)
        {
            valid[direction] = true;
            totalDiff += diff;
            diffs[direction] = diff;
        }
        else
        {
            valid[direction] = false;
            diffs[direction] = 0;
        }
    }

    if (totalDiff > 0.0f)
    {
        for (int direction = 0; direction < 8; direction++) 
        {
            if (valid[direction]) {
                float flow = kernelSettings.thermalErosionAmplitude * (diffs[direction] / totalDiff);

                w_material[invokeIndex] -= flow * kernelSettings.thermalErosionStrength;
                w_material[coord + offsets[direction]] += flow * kernelSettings.thermalErosionStrength;
            }
        }
    }
}

__global__ void calculatePressureVelocity(const CudaTexture<float> *r_pressurePtr,
                                          const CudaTexture<uint8_t> *r_plateIdsPtr,
                                          CudaTexture<Vec2<float>>* w_pressureVelocityPtr)
{
    const CudaTexture<float> &r_pressure = *r_pressurePtr;
    const CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;
    CudaTexture<Vec2<float>>& w_pressureVelocity = *w_pressureVelocityPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_pressure.size()))
        return;

    const Vec2<int> coord = r_pressure.indexToCoordinate(invokeIndex);

    // Calculate gradient using central differences
    Vec2<float> velocity = {0.0f, 0.0f};

    // X-direction gradient
    float leftPressure = r_pressure[r_pressure.coordinateToIndex({coord.x - 1, coord.y})];
    float rightPressure = r_pressure[r_pressure.coordinateToIndex({coord.x + 1, coord.y})];
    velocity.x = -(rightPressure - leftPressure); // Negative for downward slope

    // Y-direction gradient
    float bottomPressure = r_pressure[r_pressure.coordinateToIndex({coord.x, coord.y - 1})];
    float topPressure = r_pressure[r_pressure.coordinateToIndex({coord.x, coord.y + 1})];
    velocity.y = -(topPressure - bottomPressure); // Negative for downward slope

    w_pressureVelocity[invokeIndex] = velocity;
}

__global__ void initEffortToBoundary(const CudaTexture<uint8_t> *r_plateIdsPtr, CudaTexture<float> *w_effortToBoundaryPtr, uint8_t plateId)
{
    const CudaTexture<uint8_t> &r_plateIds = *r_plateIdsPtr;
    CudaTexture<float> &w_effortToBoundary = *w_effortToBoundaryPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIds.size()))
        return;

    if (r_plateIds[invokeIndex] == plateId)
        w_effortToBoundary[invokeIndex] = INFINITY;
}

__global__ void propagateEffortToBoundary(const CudaTexture<float> *r_effortToBoundaryPtr, CudaTexture<float> *w_effortToBoundaryPtr, const CudaTexture<float> *r_costPtr, int *hasChanged)
{
    const CudaTexture<float> &r_effortToBoundary = *r_effortToBoundaryPtr;
    CudaTexture<float> &w_effortToBoundary = *w_effortToBoundaryPtr;
    const CudaTexture<float> &r_cost = *r_costPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_effortToBoundary.size()))
        return;

    const Vec2<int> coord = r_effortToBoundary.indexToCoordinate(invokeIndex);
    const float oldDist = r_effortToBoundary[invokeIndex];
    float minDist = oldDist;

    // 8-connected neighbors
    const Vec2<int> offsets[] = {
        {-1,  0}, {1, 0}, {0, -1}, {0, 1},   // 4-connected: N, S, W, E
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}   // diagonals: NW, NE, SW, SE
    };

    const float currentCost = r_cost[invokeIndex];
    constexpr float SQRT2 = 1.41421356237f;

    for (int i = 0; i < 8; i++)
    {
        const Vec2<int> neighborCoord = coord + offsets[i];

        // Diagonal neighbors (i >= 4) have longer distance
        const float distanceMultiplier = (i >= 4) ? SQRT2 : 1.0f;
        const float tentative = r_effortToBoundary[neighborCoord] + currentCost * distanceMultiplier;

        if (tentative < minDist)
            minDist = tentative;
    }

    w_effortToBoundary[invokeIndex] = minDist;

    if (minDist != oldDist)
    {
        atomicOr(hasChanged, 1);
    }
}

__global__ void BacktrackPath(const CudaTexture<float> *r_effortToBoundaryPtr, CudaTexture<uint8_t> *rw_plateIdsPtr, const Vec2<int> point, uint8_t plateId)
{
    const CudaTexture<float> &r_effortToBoundary = *r_effortToBoundaryPtr;
    CudaTexture<uint8_t> &rw_plateIds = *rw_plateIdsPtr;

    Vec2<int> current = point;
    uint8_t oldPlateId = rw_plateIds[point];

    const Vec2<int> offsets[] = {
        {-1,  0}, {1, 0}, {0, -1}, {0, 1},   
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };

    while (r_effortToBoundary[current] > 0)
    {
        rw_plateIds[current] = MAX_PLATE_COUNT;

        float best = INFINITY;
        Vec2<int> next;

        for (int i = 0; i < 8; i++)
        {
            float x = r_effortToBoundary[current + offsets[i]];
            if (x < best)
            {
                best = x;
                next = current + offsets[i];
            }
        }

        current = next;
    }

    Vec2<float> firstBoundaryDir = Vec2<float>(current.x - point.x, current.y - point.y);
    Vec2<float> oppositeDir = (firstBoundaryDir * -1.0f).normalizedZeroSafe();
    

    current = point;
    while (r_effortToBoundary[current] > 0)
    {
        rw_plateIds[current] = MAX_PLATE_COUNT;

        float best = INFINITY;
        Vec2<int> next;

        for (int i = 0; i < 8; i++)
        {
            Vec2<int> neighborPos = current + offsets[i];

            if (rw_plateIds[neighborPos] == MAX_PLATE_COUNT)
                continue;

            ////back up boundary, prevent the line from looping back.
            //Vec2<float> perpendicular = Vec2<float>(-oppositeDir.y, oppositeDir.x);
            //Vec2<float> toNeighbor = Vec2<float>(neighborPos.x - point.x, neighborPos.y - point.y);
            //float perpendicularDot = toNeighbor.dot(perpendicular);
            //if (perpendicularDot > 0) // Neighbor is on the wrong side (first loop's side)
            //    continue;

            float distValue = r_effortToBoundary[neighborPos];

            //Replace this with better solution. this kinda sucks
            Vec2<float> toNeighborDir = Vec2<float>(offsets[i].x, offsets[i].y).normalized();
            float dot = toNeighborDir.dot(oppositeDir);
            float bias = -dot * 50; // Strong negative bias encourages movement in opposite direction
            float score = distValue + bias;

            if (score < best)
            {
                best = score;
                next = neighborPos;
            }
        }

        if (best != INFINITY)
        {
            current = next;
        }
        else
        {
            printf("dead end \n");
            return;
        }
    }

    Vec2<float> perpendicular = Vec2<float>(-oppositeDir.y, oppositeDir.x);
    Vec2<int> sampleRounded = point;

    for (int i = 1; i <= 100 && rw_plateIds[sampleRounded] != oldPlateId; i++)
    {
        int x = (i % 2 == 1) ? (i + 1) / 2 : -(i / 2);
        Vec2<float> sample = Vec2<float>(point.x, point.y) + (perpendicular * float(x));
        sampleRounded = Vec2<int>(static_cast<int>(sample.x + 0.5f), static_cast<int>(sample.y + 0.5f));
    }

    if (rw_plateIds[sampleRounded] == oldPlateId)
        rw_plateIds[sampleRounded] = plateId;
    else
        printf("Warning: Could not find seed point for flood fill\n");
}

__global__ void floodFillPlate(CudaTexture<uint8_t> *rw_plateIdsPtr, uint8_t oldPlateId, uint8_t newPlateId, int *hasChanged)
{
    CudaTexture<uint8_t> &rw_plateIds = *rw_plateIdsPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, rw_plateIds.size()))
        return;

    const uint8_t currentPlateId = rw_plateIds[invokeIndex];

    if (currentPlateId != oldPlateId)
        return;

    const Vec2<int> coord = rw_plateIds.indexToCoordinate(invokeIndex);

    const Vec2<int> offsets[] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };

    bool hasNewNeighbor = false;
    for (int i = 0; i < 4; i++)
    {
        const Vec2<int> neighborCoord = coord + offsets[i];
        if (isWithinBounds(rw_plateIds.coordinateToIndex(neighborCoord), rw_plateIds.size()))
        {
            if (rw_plateIds[neighborCoord] == newPlateId)
            {
                hasNewNeighbor = true;
                break;
            }
        }
    }

    if (hasNewNeighbor)
    {
        rw_plateIds[invokeIndex] = newPlateId;
        atomicOr(hasChanged, 1);
     }
}

__global__ void resetMaxPlateCountPixels(CudaTexture<uint8_t> *rw_plateIdsPtr, uint8_t newPlateId)
{
    CudaTexture<uint8_t> &rw_plateIds = *rw_plateIdsPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, rw_plateIds.size()))
        return;

    if (rw_plateIds[invokeIndex] == MAX_PLATE_COUNT)
        rw_plateIds[invokeIndex] = newPlateId;
}