#include <cstdint>
#ifndef DISTANCE_FIELD_BUFFER_H
#define DISTANCE_FIELD_BUFFER_H

struct DistanceFieldBuffer
{
    int dist;
    unsigned int origIndex;

    __host__ __device__ DistanceFieldBuffer()
        : dist(0)
        , origIndex(0)
    { }

    __host__ __device__ DistanceFieldBuffer(int distance, unsigned int originalIndex)
        : dist(distance)
        , origIndex(originalIndex)
    { }
};

#endif //DISTANCE_FIELD_BUFFER_H