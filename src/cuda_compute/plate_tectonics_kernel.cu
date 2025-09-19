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


// Precomputed Gaussian weights for range -8 to +8 (sigma = 2.0)
__constant__ float GAUSSIAN_WEIGHTS[17] = {
    0.0003f, 0.0013f, 0.0044f, 0.0122f, 0.0273f, 0.0540f, 0.0958f, 0.1515f, 0.2120f,
    0.2595f,
    0.2120f, 0.1515f, 0.0958f, 0.0540f, 0.0273f, 0.0122f, 0.0044f
};

// Helper function to get Gaussian weight for given offset
__device__ float getGaussianWeight(int offset) {
    if (abs(offset) > 8) return 0.0f;
    return GAUSSIAN_WEIGHTS[offset + 8]; // Convert offset to array index
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

    r_height[invokeIndex] = value;
}

__global__ void finalPixelPass(CudaTexture<uint8_t> *rw_idTexturePtr,
                               const CudaTexture<float> *r_heightTexturePtr,
                               const uint8_t *r_plateMergeIds,
                               PlateData *w_plateData)
{
    __shared__ float localMassSum[MAX_PLATE_COUNT];
    __shared__ int localSize[MAX_PLATE_COUNT];
    __shared__ int localPerimeter[MAX_PLATE_COUNT];

    CudaTexture<uint8_t> &rw_idTexture = *rw_idTexturePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, rw_idTexture.size()))
        return;

    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        localMassSum[threadIdx.x] = 0;
        localSize[threadIdx.x] = 0;
        localPerimeter[threadIdx.x] = 0;
    }

    __syncthreads();

    // Merge if the new merged plate is different from the original plate
    uint8_t plateID = rw_idTexture[invokeIndex];

    if (const uint8_t mergedPlateID = r_plateMergeIds[plateID]; mergedPlateID != MAX_PLATE_COUNT)
    {
        rw_idTexture[invokeIndex] = mergedPlateID;
        plateID = mergedPlateID;
    }

    const float pixelHeight = (*r_heightTexturePtr)[invokeIndex];
    const Vec2<int> coord = rw_idTexture.indexToCoordinate(invokeIndex);
    const Vec2<int> mapSize = rw_idTexture.size();

    atomicAdd(&localMassSum[plateID], pixelHeight);
    atomicAdd(&localSize[plateID], 1);

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
        uint8_t neighborPlateId = rw_idTexture[neighborCoord];
        
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
            atomicAdd(&w_plateData[threadIdx.x].mass, localMassSum[threadIdx.x]);
        if (localSize[threadIdx.x] != 0)
            atomicAdd(&w_plateData[threadIdx.x].size, localSize[threadIdx.x]);
        if (localPerimeter[threadIdx.x] != 0)
            atomicAdd(&w_plateData[threadIdx.x].perimeter, localPerimeter[threadIdx.x]);
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


__device__ void processDivergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateLookup,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, const CudaTexture<float> *r_heightMapPtr, CudaTexture<uint8_t> *w_plateIdsPtr,
                                  CudaTexture<float> *w_heightMapPtr, const unsigned int invokeIndex)
{
    const uint8_t previousPlateId = (*r_plateIdsPtr)[invokeIndex];
    (*w_plateIdsPtr)[invokeIndex] = previousPlateId;
    CudaTexture<float> &w_height = *w_heightMapPtr;
    const float original_height = (*r_heightMapPtr)[invokeIndex];
    const float value = cuda::std::lerp(original_height, 20, 0.05f);
    w_height[invokeIndex] = value;

    // Zero indicates the new crust will be part of the plate moving away, 1 indicates it will be part of the other plate

    (*w_plateIdsPtr)[invokeIndex] = previousPlateId;

}

__device__ Vec2<int> previousTextureIndex(const Vec2<int> currentTexIndex, const PlateData &plateData)
{
    const Vec2 offset = plateData.pixelCenter + plateData.direction * plateData.velocity;
    const Vec2 pixelOffset = {static_cast<int>(offset.x), static_cast<int>(offset.y)};

    return currentTexIndex - pixelOffset;
}

__device__ void applyInelasticCollision(const CudaTexture<uint8_t> *r_plateIdsPtr,
                                        const CudaTexture<float> *r_heightMapPtr,
                                        PlateData *plateLookup, const uint8_t plateA, const uint8_t plateB,
                                        const uint8_t plateC,
                                        const uint8_t plateD, const unsigned int invokeIndex)
{
    const Vec2 currentTexIndex = r_plateIdsPtr->indexToCoordinate(invokeIndex);

    PlateData &plateAData = plateLookup[plateA];
    PlateData &plateBData = plateLookup[plateB];


    const Vec2<int> plateATexIndex = previousTextureIndex(currentTexIndex, plateAData);
    const Vec2<int> plateBTexIndex = previousTextureIndex(currentTexIndex, plateBData);

    const CudaTexture<float> &r_heightMap = *r_heightMapPtr;

    const float plateAMass = max(r_heightMap[plateATexIndex], 0.001f);
    const float plateBMass = max(r_heightMap[plateBTexIndex], 0.001f);
    float plateCMass = 0;
    float plateDMass = 0;

    const Vec2<float> plateAVelocityVector = plateAData.direction * plateAData.velocity;
    const Vec2<float> plateBVelocityVector = plateBData.direction * plateBData.velocity;


    Vec2<float> numerator = plateAVelocityVector * plateAMass +
                            plateBVelocityVector * plateBMass;

    float denominator = plateAMass + plateBMass;

    if (plateC != MAX_PLATE_COUNT)
    {
        const PlateData &plateCData = plateLookup[plateC];
        const Vec2<int> plateCTexIndex = previousTextureIndex(currentTexIndex, plateCData);
        plateCMass = max(r_heightMap[plateCTexIndex], 0.001f);
        numerator += plateCData.direction * plateCData.velocity * plateCMass;
        denominator += plateCMass;
    }

    if (plateD != MAX_PLATE_COUNT)
    {
        const PlateData &plateDData = plateLookup[plateD];
        const Vec2<int> plateDTexIndex = previousTextureIndex(currentTexIndex, plateDData);
        plateDMass = max(r_heightMap[plateDTexIndex], 0.001f);
        numerator += plateDData.direction * plateDData.velocity * plateDMass;
        denominator += plateDMass;
    }

    const Vec2<float> finalVelocity = numerator / denominator;

    const Vec2<float> finalVelocityDir = finalVelocity.normalized();
    const float vel = finalVelocity.magnitude();

    const Vec2<float> velocityChangePlateA = (finalVelocity - plateAVelocityVector) * (plateAMass / plateAData.mass) *
                                             kernelSettings.inelasticCollisionMultiplier;
    const Vec2<float> velocityChangePlateB = (finalVelocity - plateBVelocityVector) * (plateBMass / plateBData.mass) *
                                             kernelSettings.inelasticCollisionMultiplier;

    atomicAddVec2(&plateAData.velocityChange, velocityChangePlateA);

    atomicAddVec2(&plateBData.velocityChange, velocityChangePlateB);

    if (plateC != MAX_PLATE_COUNT)
    {
        PlateData &plateCData = plateLookup[plateC];
        const Vec2<float> velocityChangePlateC =
                (finalVelocity - plateCData.direction * plateCData.velocity) * (plateCMass / plateCData.mass) *
                kernelSettings.inelasticCollisionMultiplier;
        atomicAddVec2(&plateCData.velocityChange, velocityChangePlateC);
    }

    if (plateD != MAX_PLATE_COUNT)
    {
        PlateData &plateDData = plateLookup[plateD];
        const Vec2<float> velocityChangePlateD =
                (finalVelocity - plateDData.direction * plateDData.velocity) * (plateDMass / plateDData.mass) *
                kernelSettings.inelasticCollisionMultiplier;
        atomicAddVec2(&plateDData.velocityChange, velocityChangePlateD);
    }
}

__device__ void processConvergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                   const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                   CudaTexture<float> *w_heightMapPtr, CudaTexture<float> *w_convergenceMapPtr,
                                   CudaTexture<uint8_t> *w_platesHaveCollidedPtr,
                                   const uint8_t plateA, const uint8_t plateB, const uint8_t plateC,
                                   const uint8_t plateD, const unsigned int invokeIndex)
{
    const CudaTexture<float> &r_height = *r_heightMapPtr;
    const CudaTexture<uint8_t> &plateIds = *r_plateIdsPtr;
    const Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    const PlateData plateAData = r_plateLookup[plateA];
    const float heightA = r_height[Vec2<float>(coord.x, coord.y) - plateAData.direction * plateAData.velocity];

    float value = plateAData.mass;
    float force = 0;
    uint8_t plate = plateA;


    if (plateB != MAX_PLATE_COUNT)
    {
        const PlateData plateBData = r_plateLookup[plateB];

        if (value < plateBData.mass)
        {
            value = plateBData.mass;
            plate = plateB;
        }
    }

    if (plateC != MAX_PLATE_COUNT)
    {
        const PlateData plateCData = r_plateLookup[plateC];
        if (value < plateCData.mass)
        {
            value = plateCData.mass;
            plate = plateC;
        }
    }

    if (plateD != MAX_PLATE_COUNT)
    {
        const PlateData plateDData = r_plateLookup[plateD];
        if (value < plateDData.mass)
        {
            value = plateDData.mass;
            plate = plateD;
        }
    }

    const PlateData plateData = r_plateLookup[plate];

    CudaTexture<float> &w_height = *w_heightMapPtr;
    w_height[invokeIndex] = r_height[Vec2<float>(coord.x, coord.y) - plateData.direction * plateData.velocity];

    (*w_convergenceMapPtr)[invokeIndex] = 1.0f;
    (*w_plateIdsPtr)[invokeIndex] = plate;

    // Reporting the collision to the platesHaveCollided texture
    // Only the first two plates are reported for simplicity and efficiency reasons
    const Vec2<int> accessVec(min(plateA, plateB), max(plateA, plateB));
    if (!(*w_platesHaveCollidedPtr)[accessVec])
    {
        (*w_platesHaveCollidedPtr)[accessVec] = 1;
    }
}

__device__ void processMovement(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                CudaTexture<float> *w_heightMapPtr, const uint8_t originId,
                                const unsigned int invokeIndex)
{
    (*w_plateIdsPtr)[invokeIndex] = originId;
    const CudaTexture<float> &r_height = *r_heightMapPtr;
    const CudaTexture<uint8_t> &plateIds = *r_plateIdsPtr;
    const Vec2<int> textureIndex = getTextureIndex(plateIds.size());

    CudaTexture<float> &w_height = *w_heightMapPtr;

    // Struct is small, so a copy is likely faster than referencing in Cuda
    const PlateData plateDataOrigin = r_plateLookup[originId];

    const Vec2<float> pixelMovement = plateDataOrigin.direction * plateDataOrigin.velocity;

    const Vec2<int> newPixelCenter(
        static_cast<int>(floorf(plateDataOrigin.pixelCenter.x + pixelMovement.x)),
        static_cast<int>(floorf(plateDataOrigin.pixelCenter.y + pixelMovement.y))
    );

    const Vec2<int> newTextureIndex = textureIndex - newPixelCenter;

    w_height[textureIndex] = r_height[newTextureIndex];
}

__global__ void processCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, PlateData *plateLookup,
                                  CudaTexture<uint8_t> *w_plateIdsPtr, CudaTexture<float> *w_heightMapPtr,
                                  CudaTexture<float> *w_convergenceMapPtr,
                                  CudaTexture<uint8_t> *w_platesHaveCollidedPtr)
{
    // Cache dereference since used more than once
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    // Ensure within bounds
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_collisions.size()))
        return;

    if (const uint32_t collisionValue = r_collisions[invokeIndex]; collisionValue == 0)
    // Collision value is zero, indicating no plate moved onto this pixel. Thus, this pixel is a divergence zone.
        processDivergence(r_plateIdsPtr, plateLookup, r_collisionsPtr, r_heightMapPtr, w_plateIdsPtr, w_heightMapPtr, invokeIndex);
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
            applyInelasticCollision(r_plateIdsPtr, r_heightMapPtr, plateLookup, plateA, plateB, plateC, plateD,
                                    invokeIndex);
            processConvergence(r_plateIdsPtr, r_heightMapPtr, plateLookup, w_plateIdsPtr, w_heightMapPtr,
                               w_convergenceMapPtr, w_platesHaveCollidedPtr, plateA,
                               plateB, plateC, plateD, invokeIndex);
        }
        // Otherwise, only one plate moves into the pixel, indicating ordinary movement
        else
            processMovement(r_plateIdsPtr, r_heightMapPtr, plateLookup, w_plateIdsPtr, w_heightMapPtr, plateA,
                            invokeIndex);
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
            // Use precomputed Gaussian weight
            float weight = getGaussianWeight(i);
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
            // Use precomputed Gaussian weight
            float weight = getGaussianWeight(i);
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

__global__ void updatePlateData(PlateData *plateLookup, const Vec2<int> callSize)
{
    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(callSize))
        return;
    PlateData current = plateLookup[invokeIndex];

    const Vec2 center = current.pixelCenter + current.direction * current.velocity;
    current.pixelCenter = {
        center.x - floor(center.x),
        center.y - floor(center.y)
    };

    const Vec2<float> original = current.direction * current.velocity;
    const Vec2<float> newVelocityVector = current.direction * current.velocity + current.velocityChange;
    if (newVelocityVector.x != 0 or newVelocityVector.y != 0)
    {
        current.velocity = newVelocityVector.magnitude();
        current.direction = newVelocityVector.normalized();
        // printf("%d: %f\n", invokeIndex, current.velocity);
    }

    // printf("%d original: (%f %f) change: (%f %f) new: (%f %f)\n",invokeIndex, original.x, original.y, current.velocityChange.x, current.velocityChange.y, newVelocityVector.x, newVelocityVector.y);

    current.velocityChange = {0, 0};

    current.mass = 0; // Will be updated in the next pixel kernel.
    current.size = 0; // Will be updated in the next pixel kernel.
    current.perimeter = 0; // Will be updated in the next pixel kernel.
    current.used = false;

    plateLookup[invokeIndex] = current;
}

__global__ void determinePlateMerge(const CudaTexture<uint8_t> *r_platesHaveCollidedPtr, const PlateData *r_plateLookup,
                                    uint8_t *w_plateMergeIds)
{
    const unsigned int invokeIndex = getInvokeIndex();

    const CudaTexture<uint8_t> &r_platesHaveCollided = *r_platesHaveCollidedPtr;

    if (!isWithinBounds(invokeIndex, r_platesHaveCollided.size()))
        return;

    // Check if the current plate pair has a collision
    if (!(*r_platesHaveCollidedPtr)[invokeIndex])
        return;

    const Vec2<int> texIndex = getTextureIndex(invokeIndex, r_platesHaveCollided.size());
    const PlateData &plateDataA = r_plateLookup[texIndex.x];
    const PlateData &plateDataB = r_plateLookup[texIndex.y];

    float dot = plateDataA.direction.dot(plateDataB.direction);
    float velocity_diff = abs(plateDataA.velocity - plateDataB.velocity);

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

// Step 1: Accumulate pressure and reset at fault lines
__global__ void pressureAccumulation(CudaTexture<float>* r_pressurePtr, CudaTexture<float>* w_pressurePtr, CudaTexture<uint8_t>* r_plateIdsPtr)
{
    CudaTexture<float>& r_pressure = *r_pressurePtr;
    CudaTexture<float>& w_pressure = *w_pressurePtr;
    CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;
    
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
        // Accumulate pressure in plate interiors
        w_pressure[idx] = r_pressure[idx] + kernelSettings.pressureAccumulation;
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

    const uint8_t plateId = r_plateIds[invokeIndex];
    const PlateData plateData = r_plateData[plateId];
    
    float baseStress = r_pressure[invokeIndex] / max(r_material[invokeIndex] * 0.1, 1.0f);
    //w_stress[invokeIndex] = baseStress * plateData.breakScore;
    w_stress[invokeIndex] = plateData.breakScore;

}

__global__ void computePerimeterAreaRatios(PlateData *w_plateData)
{
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex >= MAX_PLATE_COUNT)
        return;
        
    PlateData &plateData = w_plateData[invokeIndex];

    if (plateData.size > 0) {
        float area = static_cast<float>(plateData.size);
        float perimeter = static_cast<float>(plateData.perimeter);
        
        float circularity = 1.0f - ((2 * CURAND_2PI * area) / (perimeter * perimeter));

        float areaIncrease = area * 0.000f;

        printf("area: %.4f \n", areaIncrease);

        plateData.breakScore = circularity + areaIncrease; // circularity formula? need to include in research. 
    } else {
        plateData.breakScore = 1.0f;
    }

    printf("score: %.4f \n", plateData.breakScore);
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