#include <cstdint>
#ifndef BLUR_BUFFER_H
#define BLUR_BUFFER_H

struct BlurBuffer
{
    int offset;
    float value;

    __host__ __device__ BlurBuffer()
        : offset(0)
        , value(0)
    { }

    __host__ __device__ BlurBuffer(int offset, float value)
        : offset(offset)
        , value(value)
    { }
};

#endif //BLUR_BUFFER_H