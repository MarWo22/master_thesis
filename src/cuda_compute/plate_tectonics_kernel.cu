#include "plate_tectonics_kernel.cuh"

#include <cfloat>
#include <cstdio>
#include <curand_kernel.h>

#include "cuda_helper.cuh"
#include "cuda_noise.cuh"

#include "math_functions.h"
#include "iteration_statistics.h"


// These should become dynamic or as input parameters:
#define SCALING_FACTOR 10

__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, PlateData *plateData, const Vec2<float> *seeds,
                             const int numSeeds)
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

    CudaTexture<uint8_t> &rw_idTexture = *rw_idTexturePtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, rw_idTexture.size()))
        return;

    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        localMassSum[threadIdx.x] = 0;
        localSize[threadIdx.x] = 0;
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

    atomicAdd(&localMassSum[plateID], pixelHeight);
    atomicAdd(&localSize[plateID], 1);

    __syncthreads();
    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        if (localMassSum[threadIdx.x] != 0)
            atomicAdd(&w_plateData[threadIdx.x].mass, localMassSum[threadIdx.x]);
        if (localSize[threadIdx.x] != 0)
            atomicAdd(&w_plateData[threadIdx.x].size, localSize[threadIdx.x]);
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

__global__ void initPlatesRngGen(curandState *const rngStates, const unsigned int seed, const Vec2<int> callSize)
{
    // Determine thread ID
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, callSize))
        return;

    // Initialise the RNG
    curand_init(seed, invokeIndex, 0, &rngStates[invokeIndex]);
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
    // collisions to happen, and the result should not be that differing if one is ignored
    if (const uint8_t prefixSumVal = r_exclusivePrefixSum[invokeIndex]; prefixSumVal < 4)
    // If needed, a 64bit can be used to up this to 8, but it should already be very unlikely for 4
    {
        const uint8_t plateID = r_plateIds[invokeIndex];
        const unsigned int pixelIndex = r_pixelIndices[invokeIndex];
        const uint32_t packedVal = (MAX_PLATE_COUNT - plateID) << 8 * prefixSumVal;

        atomicOr(&w_collisions[pixelIndex], packedVal);
    }
    // Just a log to see whether more than 4 collisions is common
    else
        printf("More than 4 collisions in one pixel");
}


__device__ void processDivergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateLookup,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, CudaTexture<uint8_t> *w_plateIdsPtr,
                                  CudaTexture<float> *w_heightMapPtr, const unsigned int invokeIndex,
                                  int *localSizeChange)
{
    const uint8_t previousPlateId = (*r_plateIdsPtr)[invokeIndex];
    (*w_plateIdsPtr)[invokeIndex] = previousPlateId;
    CudaTexture<float> &w_height = *w_heightMapPtr;
    w_height[invokeIndex] = 0.0f;
    const PlateData plateData = r_plateLookup[previousPlateId];

    // Zero indicates the new crust will be part of the plate moving away, 1 indicates it will be part of the other plate
    if (plateData.divergenceRandomPlate == 0)
    {
        (*w_plateIdsPtr)[invokeIndex] = previousPlateId;
        atomicAdd(&localSizeChange[previousPlateId], 1);
    } else
    {
        // Determine where the new pixel is now located
        const Vec2 offset = plateData.pixelCenter + plateData.direction * plateData.velocity;
        const Vec2 pixelOffset = {static_cast<int>(offset.x), static_cast<int>(offset.y)};

        // Find the pixel in oposite direction

        const Vec2 currentTexIndex = r_plateIdsPtr->indexToCoordinate(invokeIndex);
        const Vec2 oppositeTexIndex = currentTexIndex - pixelOffset;

        // If there are also no plates in the pixel in the other dirfection, we can assume they are both moving away
        // from each other. In that case, we always assign the new crust to the original plate to prevent plate 'islands'
        if ((*r_collisionsPtr)[oppositeTexIndex] == 0)
        {
            (*w_plateIdsPtr)[invokeIndex] = previousPlateId;
            atomicAdd(&localSizeChange[previousPlateId], 1);
        } else
        {
            const int newPlateId = (*r_plateIdsPtr)[oppositeTexIndex];
            (*w_plateIdsPtr)[invokeIndex] = newPlateId;
            atomicAdd(&localSizeChange[newPlateId], 1);
        }
    }

    // TODO: add oceanic crust to the heightmap
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
                                             SCALING_FACTOR;
    const Vec2<float> velocityChangePlateB = (finalVelocity - plateBVelocityVector) * (plateBMass / plateBData.mass) *
                                             SCALING_FACTOR;

    // printf("Plate %d: (%f %f) %f (%f %f), Plate %d: (%f %f) %f (%f %f), Plate C - %d, Plate D - %d, finalVelocity: (%f %f)\n", plateA, plateAVelocityVector.x, plateAVelocityVector.y, plateAMass, velocityChangePlateA.x, velocityChangePlateA.y, plateB, plateBVelocityVector.x, plateBVelocityVector.y, plateBMass, velocityChangePlateB.x, velocityChangePlateB.y, plateC != MAX_PLATE_COUNT, plateD != MAX_PLATE_COUNT, finalVelocity.x, finalVelocity.y);


    // printf("A: (%f %f) final: (%f %f), Mass: (%f %f)\n", velocityChangePlateA.x, velocityChangePlateA.y, finalVelocity.x, finalVelocity.y, plateAMass, plateAData.mass);

    atomicAddVec2(&plateAData.velocityChange, velocityChangePlateA);

    // printf("B: (%f %f) final: (%f %f), Mass: (%f %f)\n", velocityChangePlateB.x, velocityChangePlateB.y, finalVelocity.x, finalVelocity.y, plateBMass, plateBData.mass);

    atomicAddVec2(&plateBData.velocityChange, velocityChangePlateB);


    if (plateC != MAX_PLATE_COUNT)
    {
        PlateData &plateCData = plateLookup[plateC];
        const Vec2<float> velocityChangePlateC =
                (finalVelocity - plateCData.direction * plateCData.velocity) * (plateCMass / plateCData.mass) *
                SCALING_FACTOR;
        // printf("C: (%f %f) final: (%f %f), Mass: (%f %f)\n", velocityChangePlateC.x, velocityChangePlateC.y, finalVelocity.x, finalVelocity.y, plateCMass, plateCData.mass);

        atomicAddVec2(&plateCData.velocityChange, velocityChangePlateC);
    }

    if (plateD != MAX_PLATE_COUNT)
    {
        PlateData &plateDData = plateLookup[plateD];
        const Vec2<float> velocityChangePlateD =
                (finalVelocity - plateDData.direction * plateDData.velocity) * (plateDMass / plateDData.mass) *
                SCALING_FACTOR;
        // printf("D: (%f %f) final: (%f %f), Mass: (%f %f)\n", velocityChangePlateD.x, velocityChangePlateD.y, finalVelocity.x, finalVelocity.y, plateDMass, plateDData.mass);

        atomicAddVec2(&plateDData.velocityChange, velocityChangePlateD);
    }
}

__device__ void processConvergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                   const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                   CudaTexture<float> *w_heightMapPtr, CudaTexture<float> *w_convergenceMapPtr,
                                   CudaTexture<uint8_t> *w_platesHaveCollidedPtr,
                                   const uint8_t plateA, const uint8_t plateB, const uint8_t plateC,
                                   const uint8_t plateD, const unsigned int invokeIndex, int *localSizeChange)
{
    const CudaTexture<float> &r_height = *r_heightMapPtr;
    const CudaTexture<uint8_t> &plateIds = *r_plateIdsPtr;
    const Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    const PlateData plateAData = r_plateLookup[plateA];
    const float heightA = r_height[Vec2<float>(coord.x, coord.y) - plateAData.direction * plateAData.velocity];

    float value = plateAData.mass;
    uint8_t plate = plateA;


    if (plateB != MAX_PLATE_COUNT)
    {
        const PlateData plateBData = r_plateLookup[plateB];
        if (value < plateBData.mass) {
            value = plateBData.mass;
            plate = plateB;
        }
    }

    if (plateC != MAX_PLATE_COUNT)
    {
        const PlateData plateCData = r_plateLookup[plateC];
        if (value < plateCData.mass) {
            value = plateCData.mass;
            plate = plateC;
        }
    }

    if (plateD != MAX_PLATE_COUNT)
    {
        const PlateData plateDData = r_plateLookup[plateD];
        if (value < plateDData.mass) {
            value = plateDData.mass;
            plate = plateD;
        }
    }

    const PlateData plateData = r_plateLookup[plate];

    CudaTexture<float>& w_height = *w_heightMapPtr;
    w_height[invokeIndex] = r_height[Vec2<float>(coord.x, coord.y) - plateData.direction * plateData.velocity];

    (*w_convergenceMapPtr)[invokeIndex] = 1.0f;
    (*w_plateIdsPtr)[invokeIndex] = plate;

    atomicAdd(&localSizeChange[plateB], -1);

    // Reporting the collision to the platesHaveCollided texture
    // Only the first two plates are reported for simplicity and efficiency reasons
    const Vec2<int> accessVec(min(plateA, plateB), max(plateA, plateB));
    if (!(*w_platesHaveCollidedPtr)[accessVec])
    {
        (*w_platesHaveCollidedPtr)[accessVec] = 1;
    }

    if (plateC == MAX_PLATE_COUNT)
        return;

    atomicAdd(&localSizeChange[plateC], -1);

    if (plateD == MAX_PLATE_COUNT)
        return;

    atomicAdd(&localSizeChange[plateD], -1);
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

    Vec2<int> coord = r_height.indexToCoordinate(invokeIndex);

    CudaTexture<float> &w_height = *w_heightMapPtr;

    // Struct is small, so a copy is likely faster than referencing in Cuda
    const PlateData plateDataOrigin = r_plateLookup[originId];

    w_height[invokeIndex] = r_height[Vec2<float>(coord.x, coord.y) - plateDataOrigin.direction * plateDataOrigin.
                                     velocity];
}

__global__ void processCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, PlateData *plateLookup,
                                  CudaTexture<uint8_t> *w_plateIdsPtr, CudaTexture<float> *w_heightMapPtr,
                                  CudaTexture<float> *w_convergenceMapPtr, CudaTexture<uint8_t> *w_platesHaveCollidedPtr)
{
    __shared__ int localSizeChange[MAX_PLATE_COUNT]; // MAX_SEEDS is numSeeds
    // Cache dereference since used more than once
    const CudaTexture<uint32_t> &r_collisions = *r_collisionsPtr;
    // Ensure within bounds
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_collisions.size()))
        return;

    if (threadIdx.x < MAX_PLATE_COUNT)
    {
        localSizeChange[threadIdx.x] = 0;
    }

    __syncthreads();

    if (const uint32_t collisionValue = r_collisions[invokeIndex]; collisionValue == 0)
    // Collision value is zero, indicating no plate moved onto this pixel. Thus, this pixel is a divergence zone.
        processDivergence(r_plateIdsPtr, plateLookup, r_collisionsPtr, w_plateIdsPtr, w_heightMapPtr, invokeIndex,
                          localSizeChange);
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
                               plateB, plateC, plateD, invokeIndex, localSizeChange);
        }
        // Otherwise, only one plate moves into the pixel, indicating ordinary movement
        else
            processMovement(r_plateIdsPtr, r_heightMapPtr, plateLookup, w_plateIdsPtr, w_heightMapPtr, plateA,
                            invokeIndex);
    }

    __syncthreads();
    if (threadIdx.x < MAX_PLATE_COUNT and localSizeChange[threadIdx.x] != 0)
    {
        atomicAdd(&plateLookup[threadIdx.x].size, localSizeChange[threadIdx.x]);
    }
}

__global__ void processUplift(CudaTexture<float> *w_upliftMapPtr, CudaTexture<float> *w_heightMapPtr, const int size,
                              const float noiseFrequency, const float noiseIntensity, const int seed)
{
    CudaTexture<float> &r_uplift = *w_upliftMapPtr;
    CudaTexture<float> &r_height = *w_heightMapPtr;

    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_uplift.size()))
        return;

    Vec2<int> center = r_uplift.indexToCoordinate(invokeIndex);

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
    r_height[invokeIndex] += clamp01(value / size2) * 0.2f;
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

__global__ void updatePlateData(PlateData *plateLookup, curandState *const rngStates, const Vec2<int> callSize)
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

    current.divergenceRandomPlate = curand(&rngStates[invokeIndex]) % 2;

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


    if (dot >= .995f and velocity_diff <= 0.025)
    {
        printf("MERGING (%d,%d): dot: %.5f plateADir: (%.5f, %.5f) plateBDir: (%.5f, %.5f) plateAVel: %.5f plateBVel: %.5f  %.5f\n", texIndex.x, texIndex.y, dot, plateDataA.direction.x, plateDataA.direction.y, plateDataB.direction.x, plateDataB.direction.y, plateDataA.velocity, plateDataB.velocity, velocity_diff);
        // Merge, the plate with lower mass merges into the plate with higher mass
        if (plateDataA.mass > plateDataB.mass)
            w_plateMergeIds[texIndex.y] = texIndex.x;
        else
            w_plateMergeIds[texIndex.x] = texIndex.y;
    }
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

__global__ void findPlateCenter(const IterationStatistics* r_stats, PlateData* plateLookup, const CudaTexture<uint8_t>* r_plateIdsPtr, float4* samples) {
    __shared__ float4 angularCoords[256];
    const unsigned int invokeIndex = getInvokeIndex();
    if (!isWithinBounds(invokeIndex, r_plateIdsPtr->size()))
        return;

    const CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;
    Vec2<int> coord = r_plateIds.indexToCoordinate(invokeIndex);

    uint8_t currentId = r_plateIds[invokeIndex];
    if (coord.x % 10 == 0 && coord.y % 10 == 0 && currentId == r_stats->largestPlateId) {
        Vec2<float> floatCoord = Vec2<float>(coord.x, coord.y);

        float angleX = CURAND_2PI * floatCoord.x / r_plateIds.size().x;
        float angleY = CURAND_2PI * floatCoord.y / r_plateIds.size().y;

        float sinX = sinf(angleX);
        float cosX = cosf(angleX);
        float sinY = sinf(angleY);
        float cosY = cosf(angleY);

        angularCoords[threadIdx.x] = make_float4(sinX, cosX, sinY, cosY);
    }
    else {
        angularCoords[threadIdx.x] = make_float4(0, 0, 0, 0);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float4 sum = {};
        for (int i = 0; i < blockDim.x; ++i) {
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

__global__ void findPlausibleSplitLine(const IterationStatistics* r_stats, const Vec2<float> r_pivot, const CudaTexture<uint8_t>* r_plateIdsPtr, Vec2<float>* output) {
    extern __shared__ float w_l_buffer[];
    float* bufferPtr = w_l_buffer;
    
    const CudaTexture<uint8_t>& r_plateIds = *r_plateIdsPtr;
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
    while (iterations < 100 && stepsize > minStepsize) {
        foundPlate = r_plateIds[Vec2<int>(point.x, point.y)];
        if (foundPlate == r_stats->largestPlateId) {
            point += dir * stepsize;
            distance += stepsize;
        }
        else {
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

    if (invokeIndex == 0) {
        int thread = 0;
        int oppposite = n * 0.5;
        float best = bufferPtr[0] + bufferPtr[oppposite];
        for (size_t i = 1; i < n * 0.5; i++)
        {
            if (bufferPtr[i] + bufferPtr[i + oppposite] > best) {
                best = bufferPtr[i] + bufferPtr[i + oppposite];
                thread = i;
            }
        }

        *output = getDirForInvokeIndex(thread, baseDir, n);
        
        printf("PlateId: %.i score: %.2f pivot: %.2f %.2f size: %.i \n", r_stats->largestPlateId, best, output->x, output->y, r_stats->largestValue);
    }
}

__global__ void splitPlate(const IterationStatistics* r_stats, const uint8_t* r_newPlateId, const Vec2<float> r_pivot, const Vec2<float>* r_dir, CudaTexture<uint8_t>* w_plateIdsPtr, PlateData* plateLookup) 
{    
    const unsigned int invokeIndex = getInvokeIndex();

    CudaTexture<uint8_t>& w_plateIds = *w_plateIdsPtr;
    
    if (invokeIndex == 0) 
    {
        printf("splitting from %.i to %.i \n", r_stats->largestPlateId, *r_newPlateId);

        plateLookup[*r_newPlateId].direction += Vec2<float>(r_dir->y, -r_dir->x);
        plateLookup[*r_newPlateId].direction = plateLookup[*r_newPlateId].direction.normalized();

        plateLookup[*r_newPlateId].mass = plateLookup[r_stats->largestPlateId].mass * 0.5;
        plateLookup[*r_newPlateId].size = plateLookup[r_stats->largestPlateId].size * 0.5;
        plateLookup[*r_newPlateId].velocity = plateLookup[r_stats->largestPlateId].velocity;



        plateLookup[r_stats->largestPlateId].direction += Vec2<float>(-r_dir->y, r_dir->x);
        plateLookup[r_stats->largestPlateId].direction = plateLookup[r_stats->largestPlateId].direction.normalized();

        plateLookup[r_stats->largestPlateId].mass = plateLookup[r_stats->largestPlateId].mass * 0.5;
        plateLookup[r_stats->largestPlateId].size = plateLookup[r_stats->largestPlateId].size * 0.5;





        

    }

    if (w_plateIds[invokeIndex] == r_stats->largestPlateId) 
    {
        const Vec2<int> coordinate = w_plateIds.indexToCoordinate(invokeIndex);

        float distanceToLine = (coordinate.x - r_pivot.x) * r_dir->y - (coordinate.y - r_pivot.y) * r_dir->x;

        float halfDistance = min(w_plateIds.size().x / abs(r_dir->x), w_plateIds.size().y / abs(r_dir->y)) * sqrt(r_dir->x * r_dir->x + r_dir->y * r_dir->y);

        if ((distanceToLine > 0.0) ^ (distanceToLine > halfDistance)) {
            w_plateIds[invokeIndex] = *r_newPlateId;
        }
    }
}

__global__ void statisticsPass(PlateData* w_plateData, IterationStatistics* w_stats) {
    const unsigned int invokeIndex = getInvokeIndex();
    if (invokeIndex == 0) {
        w_stats->heaviestValue = 0;
        w_stats->largestValue = 0;
        for (size_t i = 0; i < MAX_PLATE_COUNT; i++)
        {
            if (w_stats->heaviestValue < w_plateData[i].mass) {
                w_stats->heaviestValue = w_plateData[i].mass;
                w_stats->heaviestPlateId = i;
            }

            if (w_stats->largestValue < w_plateData[i].size) {
                w_stats->largestValue = w_plateData[i].size;
                w_stats->largestPlateId = i;
            }
            w_plateData[i].used = w_plateData[i].size > 0;
        }
        printf("biggest: %d heaviest: %d \n", static_cast<int>(w_stats->largestPlateId), static_cast<int>(w_stats->heaviestPlateId));
    }
}

__global__ void selectUnusedPlateId(PlateData* plateLookup, uint8_t* plateId) {
    const unsigned int invokeIndex = getInvokeIndex();

    if (invokeIndex == 0) {
        for (size_t i = 0; i < MAX_PLATE_COUNT; i++)
        {
            if (!plateLookup[i].used) {
                *plateId = i;
                plateLookup[i].used = true;
                printf("Selected: %.i \n", i);
                break;
            }
        }
    }
}