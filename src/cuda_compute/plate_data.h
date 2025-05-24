#ifndef PLATE_DATA_H
#define PLATE_DATA_H
#include "vec2.cuh"

struct PlateData
{
    float velocity;
    Vec2<float> direction;
    Vec2<float> pixelCenter; // ranges from [0,1]
    Vec2<float> velocityChange;
    unsigned int divergenceRandomPlate; // 0 or 1, indicates to which plate the new crust will belong
    int size;
    float mass;

    __host__ __device__ PlateData()
        : velocity(0)
        , divergenceRandomPlate(0)
        , size(0)
        , mass(0)
    {}
};

#endif //PLATE_DATA_H
