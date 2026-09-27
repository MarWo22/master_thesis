#ifndef VORONOI_SEED_H
#define VORONOI_SEED_H

#include "vec2.cuh"

struct VoronoiSeed
{
    Vec2<float> position;
    uint8_t id{};
};

#endif //VORONOI_SEED_H
