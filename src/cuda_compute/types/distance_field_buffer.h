#include <cstdint>
#ifndef DISTANCE_FIELD_BUFFER_H
#define DISTANCE_FIELD_BUFFER_H

struct DistanceFieldBuffer
{
    int offset;
    float value;

    __host__ __device__ DistanceFieldBuffer()
        : offset(0)
        , value(0)
    { }

    __host__ __device__ DistanceFieldBuffer(int offset, float value)
        : offset(offset)
        , value(value)
    { }
};

#endif //DISTANCE_FIELD_BUFFER_H