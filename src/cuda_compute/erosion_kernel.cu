#include "erosion_kernel.cuh"

#include <cfloat>
#include <cstdio>

#include "cuda_helper.cuh"
#include "cuda_texture.cuh"
#include <curand_kernel.h>



__global__ void rainComputation(CudaTexture<float> hydration, curandState* const rngStates, float deltatime)
{
    unsigned int idx = getInvokeIndex();
    hydration[idx] += curand_uniform(&rngStates[idx]) * deltatime;
}

__device__ float fluxSubComputation(CudaTexture<float> material, CudaTexture<float> hydration, float& flux, Vec2<int> coordinateSelf, Vec2<int> coordinateNeighbor, float deltatime, float gravity, float pipe_cross_section, float pipe_length) {
    float deltaE = material[coordinateSelf] + hydration[coordinateSelf] - material[coordinateNeighbor] - hydration[coordinateNeighbor];
    //printf("flux: %.4f", fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length)));
    return fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length));
}

__global__ void fluxComputation(CudaTexture<float> material, CudaTexture<float> hydration, CudaTexture<float4> flux, float deltatime, float gravity, float pipe_cross_section, float pipe_length)
{
    unsigned int idx = getInvokeIndex();    
    Vec2<int> coords = getTextureIndex(material.size());

    float4 current_f = flux[idx];

    float fluxTotal = fmaxf(current_f.x + current_f.y + current_f.z + current_f.w, 0.01f);
    float K = fminf(1, hydration[idx] / (fluxTotal * deltatime));

    float4 intermediate_flux = make_float4(0, 0, 0, 0);

    intermediate_flux.x = fluxSubComputation(material, hydration, flux[idx].x, coords, Vec2<int>(coords.x - 1, coords.y), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.y = fluxSubComputation(material, hydration, flux[idx].y, coords, Vec2<int>(coords.x, coords.y + 1), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.z = fluxSubComputation(material, hydration, flux[idx].z, coords, Vec2<int>(coords.x + 1, coords.y), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.w = fluxSubComputation(material, hydration, flux[idx].w, coords, Vec2<int>(coords.x, coords.y - 1), deltatime, gravity, pipe_cross_section, pipe_length) * K;

    flux[idx] = intermediate_flux;
}



__global__ void flowComputation(CudaTexture<float> hydration, CudaTexture<float4> flux, CudaTexture<float2> velocity, float deltatime, float pipe_length)
{
    unsigned int idx = getInvokeIndex();
    Vec2<int> coords = getTextureIndex(hydration.size());

    float flowIn = 0;

    flowIn += flux[coords + Vec2<int>(1, 0)].z;
    flowIn += flux[coords + Vec2<int>(0, 1)].w;
    flowIn += flux[coords + Vec2<int>(-1, 0)].x;
    flowIn += flux[coords + Vec2<int>(0, -1)].y;

    float4 localFlux = flux[idx];
    float flowOut = localFlux.x + localFlux.y + localFlux.z + localFlux.w;

    float deltaVolume = deltatime * (flowIn - flowOut);

    hydration[idx] += deltaVolume / pipe_length;

    velocity[idx].x = (flux[coords + Vec2<int>(-1,0)].z - localFlux.x + localFlux.z - flux[coords + Vec2<int>(1, 0)].x) / 2.0;
    velocity[idx].y = (flux[coords + Vec2<int>(0, 1)].y - localFlux.w + localFlux.y - flux[coords + Vec2<int>(0, -1)].w) / 2.0;
}



__global__ void sedimentComputation(CudaTexture<float> material, CudaTexture<float> sediment, CudaTexture<Vec2<float>> velocity, float deltatime, float kc, float ks)
{
    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(material.size());

    float threshold = 0.03;
    float C = kc * sinf(fmaxf(material.Slope(coord), threshold)) * velocity[idx].magnitude();

    if (C > sediment[idx]) {
        float s = ks * (C - sediment[idx]);
        sediment[idx] = fmaxf(sediment[idx] + s, 0.0);
        material[idx] = fmaxf(material[idx] - s, 0.0);
    }
    else {
        float s = ks * (sediment[idx] - C);
        sediment[idx] = fmaxf(sediment[idx] - s, 0.0);
        material[idx] = fmaxf(material[idx] + s, 0.0);
    }

    if (material[idx] == INFINITY || material[idx] == -INFINITY) {
        printf("INFINITY!!!");
    }
}

__global__ void transportComputation(CudaTexture<float> sediment, CudaTexture<float> sedimentBuffer, CudaTexture<Vec2<float>> velocity, float deltatime)
{
    unsigned int idx = getInvokeIndex();
    Vec2<int> coord = getTextureIndex(sediment.size());

    unsigned int idx = getInvokeIndex();

    Vec2<float> vel = velocity[idx];

    //issue here. Need to setup buffer.
    sedimentBuffer[idx] = sediment[Vec2<float>(coord.x - vel.x * deltatime, coord.y - vel.y * deltatime)];
}

__global__ void evaporateComputation(CudaTexture<float> hydration, float deltatime, float ke)
{
    unsigned int idx = getInvokeIndex();
    hydration[idx] = hydration[idx] * (1 - ke * deltatime);
}