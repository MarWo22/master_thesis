#include "erosion.h"

#include <array>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace
{
    __global__ void initRandStateKernel(curandState *const rngStates, const unsigned long long seed)
    {
        // Determine thread ID
        unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
        // Initialise the RNG
        curand_init(seed, threadID, 0, &rngStates[threadID]);
    }

    __device__ void waterIncrement(ErosionGridPoint &gridPoint, curandState &randState)
    {
        float RAINDROP_THRESHOLD = 0.1f;
        float RAINDROP_AMOUNT = 1.0f;

        float value = curand_uniform(&randState);

        gridPoint.waterHeight =+ value < RAINDROP_THRESHOLD ? RAINDROP_AMOUNT : 0.f;
    }

    __device__ void flowSimulation(ErosionGridPoint &gridPoint)
    {
        float A = 1.f; // Cross-section of virtual pipe
        float g = 9.8f; // Acceleration due to gravity
        float l = 1.f; // Length of virtual pipe

    }

    __device__ void erosionDeposition(ErosionGridPoint &gridPoint)
    {}

    __device__ void sedimentTransportation(ErosionGridPoint &gridPoint)
    {}

    __device__ void evaporation(ErosionGridPoint &gridPoint)
    {}



    __device__ std::array<ErosionGridPoint, 4>  getNeighbors(const ErosionGridPoint *erosionGrid, std::pair<int, int> const &gridSize)
    {
        const unsigned int currIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int leftIndex = max(currIndex - (currIndex % gridSize.first), currIndex - 1);
        const unsigned int rightIndex = min(currIndex - (currIndex % gridSize.first) + gridSize.first - 1, currIndex + 1);

        const unsigned int topIndex = max(currIndex % gridSize.first, currIndex - gridSize.first);
        const unsigned int bottomIndex = min(gridSize.first * gridSize.second - 1, currIndex + gridSize.first);

        return {
            erosionGrid[leftIndex],
            erosionGrid[rightIndex],
            erosionGrid[topIndex],
            erosionGrid[bottomIndex]
        };
    }

    __global__ void iterationKernel(const ErosionGridPoint *erosionGrid, curandState *const randStates, std::pair<int, int> const &gridSize)
    {
        curandState randState = randStates[blockIdx.x * blockDim.x + threadIdx.x];
        ErosionGridPoint gridPoint = erosionGrid[blockIdx.x * blockDim.x + threadIdx.x];
        auto neighbors = getNeighbors(erosionGrid, gridSize);



        waterIncrement(gridPoint, randState);
        flowSimulation(gridPoint);
        erosionDeposition(gridPoint);
        sedimentTransportation(gridPoint);
        evaporation(gridPoint);

        printf("%f", gridPoint.waterHeight);
    }
}

ErosionGrid::ErosionGrid(int width, int height)
    : mWidth(width)
    , mHeight(height)
    , mGridDevice(nullptr)
    , mRandStates(nullptr)
{
    cudaMalloc(&mGridDevice, sizeof(ErosionGridPoint) * width * height);
}

ErosionGrid::~ErosionGrid()
{
    cudaFree(mGridDevice);
}

void ErosionGrid::initRandomGen(const unsigned long long seed)
{
    cudaMalloc(&mRandStates, mHeight*mWidth*sizeof(curandState));

    initRandStateKernel<<<mHeight, mWidth>>>(mRandStates, seed);
}

void ErosionGrid::executeIteration()
{
    const int len = mWidth * mHeight;
    int gridSize = (len + 256 - 1) / 256;

    iterationKernel<<<gridSize, 256>>>(mGridDevice, mRandStates, std::pair(mWidth, mHeight));
}



