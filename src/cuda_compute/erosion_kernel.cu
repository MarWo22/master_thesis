#include "erosion_kernel.cuh"

#include <cfloat>
#include <cstdio>

#include "cuda_helper.cuh"


__global__ void rainComputation(CudaTexture<float>* hydration, float deltatime, int seed)
{
    CudaTexture<float> hydrationTexture = *hydration;

    Vec2<int> coords = getTextureIndex(hydration->size());

    //hydrationTexture[coords] += cudaNoise::discreteNoise(make_float3(coords.x, coords.y, 0), 1, seed) * deltatime;
}

__device__ float fluxSubComputation(CudaTexture<float> material, CudaTexture<float> hydration, float& flux, Vec2<int> coordinateSelf, Vec2<int> coordinateNeighbor, float deltatime, float gravity, float pipe_cross_section, float pipe_length) {
    float deltaE = material[coordinateSelf] + hydration[coordinateSelf] - material[coordinateNeighbor] - hydration[coordinateNeighbor];

    return fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length));
}

__global__ void fluxComputation(CudaTexture<float>* material, CudaTexture<float>* hydration, CudaTexture<float4>* flux, float deltatime, float gravity, float pipe_cross_section, float pipe_length)
{
    CudaTexture<float> materialTexture = *material;
    CudaTexture<float> hydrationTexture = *hydration;
    CudaTexture<float4> fluxTexture = *flux;

    unsigned int idx = getInvokeIndex();    
    Vec2<int> coords = getTextureIndex(material->size());

    float4 current_f = fluxTexture[idx];

    float fluxTotal = fmaxf(current_f.x + current_f.y + current_f.z + current_f.w, 0.01f);
    float K = fminf(1, hydrationTexture[idx] / (fluxTotal * deltatime));

    float4 intermediate_flux = make_float4(0, 0, 0, 0);

    intermediate_flux.x = fluxSubComputation(materialTexture, hydrationTexture, fluxTexture[idx].x, coords, Vec2<int>(coords.x - 1, coords.y), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.y = fluxSubComputation(materialTexture, hydrationTexture, fluxTexture[idx].y, coords, Vec2<int>(coords.x, coords.y + 1), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.z = fluxSubComputation(materialTexture, hydrationTexture, fluxTexture[idx].z, coords, Vec2<int>(coords.x + 1, coords.y), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.w = fluxSubComputation(materialTexture, hydrationTexture, fluxTexture[idx].w, coords, Vec2<int>(coords.x, coords.y - 1), deltatime, gravity, pipe_cross_section, pipe_length) * K;

    fluxTexture[idx] = intermediate_flux;
}



__global__ void flowComputation(CudaTexture<float>* hydration, CudaTexture<float4>* flux, CudaTexture<Vec2<float>>* velocity, float deltatime, float pipe_length)
{
    CudaTexture<float> hydrationTexture = *hydration;
    CudaTexture<float4> fluxTexture = *flux;
    CudaTexture<Vec2<float>> velocityTexture = *velocity;
    
    unsigned int idx = getInvokeIndex();
    Vec2<int> coords = getTextureIndex(hydration->size());

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



__global__ void sedimentComputation(CudaTexture<float>* material, CudaTexture<float>* sediment, CudaTexture<Vec2<float>>* velocity, float deltatime, float kc, float ks)
{
    CudaTexture<float> materialTexture = *material;
    CudaTexture<float> sedimentTexture = *sediment;
    CudaTexture<Vec2<float>> velocityTexture = *velocity;

    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(material->size());

    float threshold = 0.03;
    float C = kc * sinf(fmaxf(materialTexture.Slope(coord), threshold)) * velocityTexture[idx].magnitude();

    if (C > sedimentTexture[idx]) {
        float s = ks * (C - sedimentTexture[idx]);
        sedimentTexture[idx] = fmaxf(sedimentTexture[idx] + s, 0.0);
        materialTexture[idx] = fmaxf(materialTexture[idx] - s, 0.0);
    }
    else {
        float s = ks * (sedimentTexture[idx] - C);
        sedimentTexture[idx] = fmaxf(sedimentTexture[idx] - s, 0.0);
        materialTexture[idx] = fmaxf(materialTexture[idx] + s, 0.0);
    }

    if (materialTexture[idx] == INFINITY || materialTexture[idx] == -INFINITY) {
        printf("INFINITY!!!");
    }
}

__global__ void transportComputation(CudaTexture<float>* sediment, CudaTexture<float>* sedimentBuffer, CudaTexture<Vec2<float>>* velocity, float deltatime)
{
    CudaTexture<float> sedimentTexture = *sediment;
    CudaTexture<float> sedimentBufferTexture = *sedimentBuffer;
    CudaTexture<Vec2<float>> velocityTexture = *velocity;
    
    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(sediment->size());

    Vec2<float> vel = velocityTexture[idx];

    //issue here. Need to setup buffer.

    sedimentBufferTexture[idx] = interpolate(Vec2<float>(coord.x - vel.x * deltatime, coord.y - vel.y * deltatime), sedimentTexture);
}

__global__ void evaporateComputation(CudaTexture<float>* hydration, float deltatime, float ke)
{
    CudaTexture<float> hydrationTexture = *hydration;

    unsigned int idx = getInvokeIndex();
    hydrationTexture[idx] = hydrationTexture[idx] * (1 - ke * deltatime);
}

__global__ void initMaterial(CudaTexture<float>* material, int seed)
{
    Vec2<int> coords = getTextureIndex(material->size());
    CudaTexture<float> materialTexture = *material;
    //materialTexture[coords] += cudaNoise::simplexNoise(make_float3(coords.x, coords.y, 0), 0.001, seed);
}