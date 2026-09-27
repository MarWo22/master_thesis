#ifndef PLATE_DATA_H
#define PLATE_DATA_H
#include "vec2.cuh"

struct PlateData
{
    float velocity;
    float velocitySmoothed;
    Vec2<float> direction;
    Vec2<float> directionSmoothed;
    Vec2<float> pixelCenter; // ranges from [0,1]
    Vec2<float> velocityChange;
    Vec2<float> geometricCenter; // geometric center of the plate
    Vec2<float> asthenosphereVelocity; // velocity of the underlying asthenosphere from curl noise
    int size;
    float mass;
    int perimeter;
    float breakScore;
    float circularity;
    bool used;
    bool hasMoved;
    int mergeWaitTime;

    __host__ __device__ PlateData()
        : velocity(0)
          , velocitySmoothed(0)
          , directionSmoothed({0.0f, 0.0f})
          , size(0)
          , mass(0)
          , perimeter(0)
          , breakScore(1.0f)
          , circularity(0)
          , asthenosphereVelocity({0.0f, 0.0f})
          , used(false)
          , hasMoved(false)
          , mergeWaitTime(0)
    {}
};

#endif //PLATE_DATA_H
