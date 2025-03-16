#include "particle_erosion.h"

#include <curand_kernel.h>
#include <iostream>
#include <ostream>

#include "FastNoiseLite.h"
#include "vec2.cuh"

__global__ void initRandStateKernel(curandState *const rngStates, const unsigned long long seed, const int maxIdx)
{
    if (blockIdx.x * blockDim.x + threadIdx.x >= maxIdx)
        return;

    // Determine thread ID
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, threadID, 0, &rngStates[threadID]);
}

__device__ float4 neighboringHeights(const float *heightMap, const Vec2<float> &pos, const int2 &mapDimensions)
{
    const unsigned int xIndex = static_cast<int>(pos.x);
    const unsigned int yIndex = static_cast<int>(pos.y);

    const unsigned int xRightIndex = min(mapDimensions.x - 1, xIndex + 1);
    const unsigned int yBottomIndex = min(mapDimensions.y - 1, yIndex + 1);

    const float heightCurrent = heightMap[yIndex * mapDimensions.x + xIndex];
    const float heightRight = heightMap[yIndex * mapDimensions.x + xRightIndex];
    const float heightBottom = heightMap[yBottomIndex * mapDimensions.x + xIndex];
    const float heightRightBottom = heightMap[yBottomIndex * mapDimensions.x + xRightIndex];

    return {heightCurrent, heightRight, heightBottom, heightRightBottom};
}

__device__ float bilinearInterpolation(float x, float y, const float4 &values) {
    const float R1 = values.y + (values.w - values.y) * x;
    const float R2 = values.x + (values.z - values.x) * x;
    return R1 + (R2 - R1) * y;
}

__device__ void simulateParticleDebug(float *heightMap, Droplet &droplet, const int2 &mapDimensions, curandState &randState)
{
    const float INERTIA = 0.1f;
    const float MIN_SLOPE = 0.01f;
    const float CAPACITY = 2.f;
    const float DEPOSITION = 0.1f;
    const float EROSION = 0.9f;
    const float GRAVITY = 10.f;
    const float EVAPORATION = 0.05f;

    printf("position: %.4f , %.4f\n", droplet.pos.x, droplet.pos.y);
    // Get the cells in the 2x2 block of the current droplet position
    const float4 localNeighborHeights = neighboringHeights(heightMap, droplet.pos, mapDimensions);
    printf("local neighbors: %.4f , %.4f , %.4f , %.4f\n", localNeighborHeights.x, localNeighborHeights.y, localNeighborHeights.z, localNeighborHeights.w);

    // Calculate the offsets u and v inside the cell
    const float u = droplet.pos.x - floorf(droplet.pos.x);
    const float v = droplet.pos.y - floorf(droplet.pos.y);
    printf("offset: %.4f , %.4f\n", u, v);
    // Calculate the gradient at the current droplet position

    Vec2 grad = {
        (localNeighborHeights.y - localNeighborHeights.x) * (1 - v) + (localNeighborHeights.w - localNeighborHeights.z) * v,
        (localNeighborHeights.z - localNeighborHeights.x) * (1 - u) + (localNeighborHeights.w - localNeighborHeights.y) * u
    };
    printf("gradient: %.4f , %.4f\n", grad.x, grad.y);

    // Randomize in case of zero magnitude
    if (grad.magnitude() == 0)
    {
        printf("Randomize");
        grad.x = curand_uniform(&randState);
        grad.y = curand_uniform(&randState);
    }

    const Vec2 normalizedGrad = grad.normalized();

    // Use gradient to determine the new droplet direction
    Vec2<float> newDir = droplet.dir * INERTIA - normalizedGrad * (1 - INERTIA);

    printf("normalized gradient: %.4f , %.4f\n", normalizedGrad.x, normalizedGrad.y);

    printf("droplet dir: %.4f , %.4f\n", droplet.dir.x, droplet.dir.y);

    const Vec2<float> newDirNormalized = newDir.normalized();
    printf("new normalized dir: %.4f , %.4f\n", newDirNormalized.x, newDirNormalized.y);

    // Get the new position by moving one unit
    // Bit of a hacky way to prevent overflows, should be addresses cleaner
    Vec2<float> newPos = (droplet.pos + newDirNormalized);
    newPos = {min(max(newPos.x, 0.f), static_cast<float>(mapDimensions.x) - 0.001f),
        min(max(newPos.y, 0.f), static_cast<float>(mapDimensions.y) - 0.001f)};

    printf("new pos: %.4f , %.4f\n", newPos.x, newPos.y);

    // Calculate the new offsets
    const float newU = newPos.x - floorf(newPos.x);
    const float newV = newPos.y - floorf(newPos.y);
    printf("offsets pos: %.4f , %.4f\n", newU, newV);

    // Get the 2x2 block of the new cell
    const float4 newNeighborHeights = neighboringHeights(heightMap, newPos, mapDimensions);
    printf("new neighbors: %.4f , %.4f , %.4f , %.4f\n", newNeighborHeights.x, newNeighborHeights.y, newNeighborHeights.z, newNeighborHeights.w);

    // Get the heights in the old and new location through bilinear interpolation
    const float oldHeight = bilinearInterpolation(newU, newV, localNeighborHeights);
    const float newHeight = bilinearInterpolation(newU, newV, newNeighborHeights);
    printf("height changes: %4f , %4f\n", oldHeight, newHeight);
    const float heightDifference = newHeight - oldHeight;
    printf("height difference: %4f\n", heightDifference);

    // Deposit if the new location has a higher height
    if (heightDifference >= 0)
    {
        // Deposit all if we are carrying less sediment than the difference
        // Otherwise deposit the difference
        if (droplet.sediment < heightDifference)
        {
            printf("Deposing all uphill: %4f\n", droplet.sediment);
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] += droplet.sediment;
            droplet.sediment = 0;
        }
        else
        {
            printf("Deposing some uphill: %4f\n", heightDifference);
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] += heightDifference;
            droplet.sediment -= heightDifference;
        }
    }
    else
    {
        // Determine the carry capacity of the drop
        const float carryCapacity = max(-heightDifference, MIN_SLOPE) * droplet.vel * droplet.water * CAPACITY;
        printf("carry capacity: %4f\n", carryCapacity);

        // If the drop is currently carrying more sediment than it's capacity, deposit partially into the old location
        // Otherwise, pick up some sediment from the old location
        printf("total sediment: %4f\n", droplet.sediment);

        if (droplet.sediment > carryCapacity)
        {
            const float dropAmount = (droplet.sediment - carryCapacity) * DEPOSITION;
            droplet.sediment -= dropAmount;
            printf("dropping sediment: %4f\n", dropAmount);
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] += dropAmount;
        }
        else
        {
            const float pickupAmount = min((carryCapacity - droplet.sediment) * EROSION, -heightDifference);
            droplet.sediment += pickupAmount;
            printf("picking up sediment: %4f\n", pickupAmount);
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] -= pickupAmount;
        }
    }
    printf("Velocity change: %.4f -> %.4f\n", droplet.vel, sqrt(max(droplet.vel * droplet.vel + -heightDifference * GRAVITY, 0.f)));
    printf("Water change: %.4f ->  %.4f\n", droplet.water, droplet.water * (1 - EVAPORATION));

    // Update the droplet values
    // I'm clamping to zero here. The original paper doesn't do it, but when moving uphill, it could potentially drop lower than zero (I think)
    droplet.vel = sqrt(max(droplet.vel * droplet.vel + -heightDifference * GRAVITY, 0.f));
    droplet.water = droplet.water * (1 - EVAPORATION);
    droplet.dir = newDir;
    droplet.pos = newPos;


    printf("\n----------------------------------\n\n");

}

__device__ void simulateParticle(float *heightMap, Droplet &droplet, const int2 &mapDimensions, curandState &randState)
{
    const float INERTIA = 0.1f;
    const float MIN_SLOPE = 0.01f;
    const float CAPACITY = 2.f;
    const float DEPOSITION = 0.1f;
    const float EROSION = 0.9f;
    const float GRAVITY = 10.f;
    const float EVAPORATION = 0.05f;
    // Get the cells in the 2x2 block of the current droplet position
    const float4 localNeighborHeights = neighboringHeights(heightMap, droplet.pos, mapDimensions);
    // Calculate the offsets u and v inside the cell
    const float u = droplet.pos.x - floorf(droplet.pos.x);
    const float v = droplet.pos.y - floorf(droplet.pos.y);
    // Calculate the gradient at the current droplet position
    Vec2 grad = {
        (localNeighborHeights.y - localNeighborHeights.x) * (1 - v) + (localNeighborHeights.w - localNeighborHeights.z) * v,
        (localNeighborHeights.z - localNeighborHeights.x) * (1 - u) + (localNeighborHeights.w - localNeighborHeights.y) * u
    };

    // Randomize in case of zero magnitude
    if (grad.magnitude() == 0)
    {
        grad.x = curand_uniform(&randState);
        grad.y = curand_uniform(&randState);
    }

    const Vec2 normalizedGrad = grad.normalized();

    // Use gradient to determine the new droplet direction
    Vec2<float> newDir = droplet.dir * INERTIA - normalizedGrad * (1 - INERTIA);

    const Vec2<float> newDirNormalized = newDir.normalized();

    // Get the new position by moving one unit
    // Bit of a hacky way to prevent overflows, should be addresses cleaner

    Vec2<float> newPos = (droplet.pos + newDirNormalized);
    newPos = {min(max(newPos.x, 0.f), static_cast<float>(mapDimensions.x) - 0.001f),
        min(max(newPos.y, 0.f), static_cast<float>(mapDimensions.y) - 0.001f)};


    // Calculate the new offsets
    const float newU = newPos.x - floorf(newPos.x);
    const float newV = newPos.y - floorf(newPos.y);

    // Get the 2x2 block of the new cell
    const float4 newNeighborHeights = neighboringHeights(heightMap, newPos, mapDimensions);

    // Get the heights in the old and new location through bilinear interpolation
    const float oldHeight = bilinearInterpolation(u, v, localNeighborHeights);
    const float newHeight = bilinearInterpolation(newU, newV, newNeighborHeights);
    const float heightDifference = newHeight - oldHeight;

    // Deposit if the new location has a higher height
    if (heightDifference >= 0)
    {
        // Deposit all if we are carrying less sediment than the difference
        // Otherwise deposit the difference
        if (droplet.sediment < heightDifference)
        {
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] += droplet.sediment;
            droplet.sediment = 0;
        }
        else
        {
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] += heightDifference;
            droplet.sediment -= heightDifference;
        }
    }
    else
    {
        // Determine the carry capacity of the drop
        const float carryCapacity = max(-heightDifference, MIN_SLOPE) * droplet.vel * droplet.water * CAPACITY;

        // If the drop is currently carrying more sediment than it's capacity, deposit partially into the old location
        // Otherwise, pick up some sediment from the old location

        if (droplet.sediment > carryCapacity)
        {
            const float dropAmount = (droplet.sediment - carryCapacity) * DEPOSITION;
            droplet.sediment -= dropAmount;
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] += dropAmount;
        }
        else
        {
            const float pickupAmount = min((carryCapacity - droplet.sediment) * EROSION, -heightDifference);
            droplet.sediment += pickupAmount;
            heightMap[static_cast<int>(droplet.pos.y) * mapDimensions.x + static_cast<int>(droplet.pos.x)] -= pickupAmount;
        }
    }

    // Update the droplet values
    // I'm clamping to zero here. The original paper doesn't do it, but when moving uphill, it could potentially drop lower than zero (I think)
    droplet.vel = sqrt(max(droplet.vel * droplet.vel + -heightDifference * GRAVITY, 0.f));
    droplet.water = droplet.water * (1 - EVAPORATION);
    droplet.dir = newDir;
    droplet.pos = newPos;

}


__global__ void particleBasedErosionKernel(float *heightMap, curandState *randStates, int width, int height, const int maxIdx)
{
    if (blockIdx.x * blockDim.x + threadIdx.x >= maxIdx)
    {
        return;
    }

    const int DROPLET_LIFESPAN = 10;

    Droplet droplet{};
    droplet.water = 1.f;
    droplet.vel = 0;

    curandState randState = randStates[blockIdx.x * blockDim.x + threadIdx.x];

    const int2 mapDimensions = make_int2(width, height);


    droplet.pos = {curand_uniform(&randState) * static_cast<float>(mapDimensions.x),
        curand_uniform(&randState) * static_cast<float>(mapDimensions.y)};

    for (int i = 0; i != DROPLET_LIFESPAN; ++i)
    {
        simulateParticle(heightMap, droplet, mapDimensions, randState);
    }

    randStates[blockIdx.x * blockDim.x + threadIdx.x] = randState;
}




float* generateRandomTexture(int height, int width, int seed)
{
    // Allocate array on device memory
    int size = height * width;
    // Allocate on host memory
    auto hTexture = new float[size];

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetSeed(seed);
    noise.SetFrequency(0.01f);
    noise.SetFractalType(FastNoiseLite::FractalType_FBm);
    noise.SetFractalOctaves(5);

    int index = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            hTexture[index++] = (noise.GetNoise(static_cast<float>(x), static_cast<float>(y)) + 1) / 2;
        }
    }
    return hTexture;
}


ParticleErosion::ParticleErosion(int width, int height)
    : mWidth(width)
    , mHeight(height)
    , mRandStates(nullptr)
    , mHeightmapDevice(nullptr)
{}

ParticleErosion::~ParticleErosion()
{
    if (mHeightmapDevice != nullptr)
        cudaFree(mHeightmapDevice);
}

void ParticleErosion::loadHeightmap(const float *heightmap)
{

    if (mHeightmapDevice != nullptr)
        cudaFree(mHeightmapDevice);

    unsigned int size = sizeof(float) * mHeight * mWidth;
    cudaMalloc(&mHeightmapDevice, size);
    cudaMemcpy(mHeightmapDevice, heightmap, size, cudaMemcpyHostToDevice);
}

void ParticleErosion::extractHeightmap(float *heightmap) const
{
    unsigned int size = sizeof(float) * mHeight * mWidth;
    cudaMemcpy(heightmap, mHeightmapDevice, size, cudaMemcpyDeviceToHost);
}



void ParticleErosion::simulateDroplets(int nDroplets, unsigned long long seed)
{
    cudaMalloc(&mRandStates, nDroplets*sizeof(1));
    initRandStateKernel<<<1, 1>>>(mRandStates, seed, 1);
    // int gridSize = (nDroplets + 256 - 1) / 256;

    // cudaMalloc(&mRandStates, nDroplets*sizeof(curandState));
    // initRandStateKernel<<<gridSize, 256>>>(mRandStates, seed, nDroplets);


    // Right now, it's set up for sequential to make it easier to debug the logic
    for (int i = 0; i != nDroplets; ++i)
    {
        particleBasedErosionKernel<<<1, 1>>>(mHeightmapDevice, mRandStates, mWidth, mHeight, nDroplets);
        if (i % 1000 == 0)
            std::cout << i << "\n";
    }
}
