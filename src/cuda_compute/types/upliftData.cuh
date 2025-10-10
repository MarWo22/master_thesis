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

    __host__ __device__ UpliftData(const int cumulativeHeight, const uint8_t collisionTypes)
        : cumulativeHeight(cumulativeHeight)
        , collisionTypes(collisionTypes)
    { }
};

#endif //UPLIFT_DATA_CUH
