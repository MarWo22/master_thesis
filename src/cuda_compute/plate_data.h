#ifndef PLATE_DATA_H
#define PLATE_DATA_H
#include "vec2.cuh"

struct PlateData
{
    float velocity;
    Vec2<float> direction;
    Vec2<float> center;

    PlateData()
        : velocity(0)
    {}
};

#endif //PLATE_DATA_H
