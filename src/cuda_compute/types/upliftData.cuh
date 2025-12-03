#ifndef UPLIFT_DATA_CUH
#define UPLIFT_DATA_CUH
#include <cstdint>

struct UpliftData
{
    float cumulativeHeight;
    uint8_t collisionTypes;

    __host__ __device__ UpliftData()
        : cumulativeHeight(0)
        , collisionTypes(0)
    { }

    __host__ __device__ UpliftData(const float cumulativeHeight, const uint8_t collisionTypes)
        : cumulativeHeight(cumulativeHeight)
        , collisionTypes(collisionTypes)
    { }
};

struct PropagatedUpliftData
{
    float cumulativeHeight;
    uint32_t involvedPlates;
    uint8_t collisionTypes;

    __host__ __device__ PropagatedUpliftData()
        : cumulativeHeight(0)
        , involvedPlates(0)
        , collisionTypes(0)
    { }

    __host__ __device__ PropagatedUpliftData(const float cumulativeHeight, const uint32_t involvedPlates, const uint8_t collisionTypes)
        : cumulativeHeight(cumulativeHeight)
        , involvedPlates(involvedPlates)
        , collisionTypes(collisionTypes)
    { }
};

#endif //UPLIFT_DATA_CUH
