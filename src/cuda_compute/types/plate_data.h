#ifndef PLATE_DATA_H
#define PLATE_DATA_H
#include "vec2.cuh"

struct PlateData
{
    float velocity;
    Vec2<float> direction;
    Vec2<float> pixelCenter; // ranges from [0,1]
    Vec2<float> velocityChange;
    Vec2<float> geometricCenter; // geometric center of the plate
    int size;
    float mass;
    int perimeter;
    float breakScore;
    float circularity;
    bool used;
    bool hasMoved;

    __host__ __device__ PlateData()
        : velocity(0)
        , size(0)
        , mass(0)
        , perimeter(0)
        , breakScore(1.0f)
        , used(false)
        , hasMoved(false)
    {}
};

#endif //PLATE_DATA_H
