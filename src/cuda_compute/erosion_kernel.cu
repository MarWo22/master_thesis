#include "erosion_kernel.cuh"

#include <cfloat>
#include <cstdio>

#include "cuda_helper.cuh"
#include "cuda_noise.cuh"
#include "flux_erosion.h"


__global__ void rainComputation(CudaErosionData data, float deltatime, int seed)
{
    Vec2<int> coords = getTextureIndex(data.m_hydration->size());
    CudaTexture<float> texture = *data.m_hydration;
    texture[coords] += cudaNoise::discreteNoise(make_float3(coords.x, coords.y, 0), 1, seed) * deltatime;
}

__device__ float fluxSubComputation(CudaErosionData data, float& flux, Vec2<int> coordinateSelf, Vec2<int> coordinateNeighbor, float deltatime, float gravity, float pipe_cross_section, float pipe_length) {
    CudaTexture<float> hydrationTexture = *data.m_hydration;
    CudaTexture<float> materialTexture = *data.m_material;
    float deltaE = materialTexture[coordinateSelf] + hydrationTexture[coordinateSelf] - materialTexture[coordinateNeighbor] - hydrationTexture[coordinateNeighbor];

    return fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length));
}

__global__ void fluxComputation(CudaErosionData data, float deltatime, float gravity, float pipe_cross_section, float pipe_length)
{
    CudaTexture<float> hydrationTexture = *data.m_hydration;
    CudaTexture<float> materialTexture = *data.m_material;
    CudaTexture<float4> fluxTexture = *data.m_flux;

    unsigned int idx = getInvokeIndex();    
    Vec2<int> coords = getTextureIndex(data.m_material->size());

    float4 current_f = fluxTexture[idx];

    float fluxTotal = fmaxf(current_f.x + current_f.y + current_f.z + current_f.w, 0.01f);
    float K = fminf(1, hydrationTexture[idx] / (fluxTotal * deltatime));

    float4 intermediate_flux = make_float4(0, 0, 0, 0);

    intermediate_flux.x = fluxSubComputation(data, fluxTexture[idx].x, coords, Vec2<int>(coords.x - 1, coords.y), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.y = fluxSubComputation(data, fluxTexture[idx].y, coords, Vec2<int>(coords.x, coords.y + 1), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.z = fluxSubComputation(data, fluxTexture[idx].z, coords, Vec2<int>(coords.x + 1, coords.y), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.w = fluxSubComputation(data, fluxTexture[idx].w, coords, Vec2<int>(coords.x, coords.y - 1), deltatime, gravity, pipe_cross_section, pipe_length) * K;

    fluxTexture[idx] = intermediate_flux;
}



__global__ void flowComputation(CudaErosionData data, float deltatime, float pipe_length)
{   
    CudaTexture<float> hydrationTexture = *data.m_hydration;
    CudaTexture<Vec2<float>> velocityTexture = *data.m_velocity;
    CudaTexture<float4> fluxTexture = *data.m_flux;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coords = getTextureIndex(data.m_hydration->size());

    float flowIn = 0;

    flowIn += fluxTexture[coords + Vec2<int>(1, 0)].z;
    flowIn += fluxTexture[coords + Vec2<int>(0, 1)].w;
    flowIn += fluxTexture[coords + Vec2<int>(-1, 0)].x;
    flowIn += fluxTexture[coords + Vec2<int>(0, -1)].y;

    float4 localFlux = fluxTexture[idx];
    float flowOut = localFlux.x + localFlux.y + localFlux.z + localFlux.w;

    float deltaVolume = deltatime * (flowIn - flowOut);

    hydrationTexture[idx] += deltaVolume / pipe_length;

    velocityTexture[idx].x = (fluxTexture[coords + Vec2<int>(-1,0)].z - localFlux.x + localFlux.z - fluxTexture[coords + Vec2<int>(1, 0)].x) / 2.0;
    velocityTexture[idx].y = (fluxTexture[coords + Vec2<int>(0, 1)].y - localFlux.w + localFlux.y - fluxTexture[coords + Vec2<int>(0, -1)].w) / 2.0;
}



__global__ void sedimentComputation(CudaErosionData data, float deltatime, float kc, float ks, float kd)
{
    CudaTexture<float> materialTexture = *data.m_material;
    CudaTexture<float> sedimentTexture = *data.m_sediment;
    CudaTexture<Vec2<float>> velocityTexture = *data.m_velocity;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(data.m_material->size());

    float threshold = 0.03f;
    float capacity = kc * fmaxf(materialTexture.Slope(coord), threshold) * velocityTexture[idx].magnitude();
    if (capacity > sedimentTexture[idx]) {
        float s = ks * (capacity - sedimentTexture[idx]);
        sedimentTexture[idx] = fmaxf(sedimentTexture[idx] + s, 0.0f);
        materialTexture[idx] = fminf(fmaxf(materialTexture[idx] - s, 0.0f), 16.0f);
    }
    else {
        float s = kd * (sedimentTexture[idx] - capacity);
        sedimentTexture[idx] = fmaxf(sedimentTexture[idx] - s, 0.0f);
        materialTexture[idx] = fminf(fmaxf(materialTexture[idx] + s, 0.0f), 16.0f);
    }
}

__global__ void transportComputation(CudaErosionData data, float deltatime)
{   
    CudaTexture<float> sedimentTexture = *data.m_sediment;
    CudaTexture<float> sedimentBufferTexture = *data.m_sedimentBuffer;
    CudaTexture<Vec2<float>> velocityTexture = *data.m_velocity;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(data.m_sediment->size());

    Vec2<float> vel = velocityTexture[idx];

    sedimentBufferTexture[idx] = interpolate(Vec2<float>(coord.x - vel.x * deltatime, coord.y - vel.y * deltatime), sedimentTexture);
}

__global__ void evaporateComputation(CudaErosionData data, float deltatime, float ke)
{
    CudaTexture<float> hydrationTexture = *data.m_hydration;

    unsigned int idx = getInvokeIndex();
    hydrationTexture[idx] = hydrationTexture[idx] * (1 - ke * deltatime);
}

__global__ void initMaterial(CudaErosionData data, int seed)
{
    CudaTexture<float> materialTexture = *data.m_material;

    Vec2<int> coords = getTextureIndex(data.m_material->size());
    materialTexture[coords] += cudaNoise::simplexNoise(make_float3(coords.x, coords.y, 0), 0.001, seed);
    materialTexture[coords] += cudaNoise::simplexNoise(make_float3(coords.x, coords.y, 0), 0.01, seed) * 0.1;
    materialTexture[coords] += cudaNoise::simplexNoise(make_float3(coords.x, coords.y, 0), 0.1, seed) * 0.01;
}