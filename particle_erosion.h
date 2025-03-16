//
// Created by marti on 05/03/2025.
//

#ifndef PARTICLE_EROSION_H
#define PARTICLE_EROSION_H

#include <curand_kernel.h>

#include "vec2.cuh"

struct ParticleErosionParameters
{
    float inertia;
    float minSlope;
    float capacity;
    float deposition;
    float erosion;
    float gravity;
    float evaporation;
    unsigned int dropletLifespan;

    ParticleErosionParameters() = default;
};

struct Droplet
{
    Vec2<float> pos; // 2D position of the droplet
    Vec2<float> dir; // 2D direction of the droplet
    float vel; // 2D velocity of the droplet
    float water; // Amount of water the droplet consists of
    float sediment; // Amount of sediment the droplet is carrying

    Droplet() = default;
};

float* generateRandomTexture(int height, int width, int seed);

class ParticleErosion
{
    int mWidth;
    int mHeight;
    curandState *mRandStates;
    float *mHeightmapDevice;


public:
    ParticleErosion(int width, int height);
    ~ParticleErosion();

    void loadHeightmap(const float *heightmap);
    void extractHeightmap(float *heightmap) const;
    void simulateDroplets(int nDroplets, unsigned long long seed);

};


#endif //PARTICLE_EROSION_H
