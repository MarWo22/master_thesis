#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "erosion.h"
#include <curand_kernel.h>
#include <iostream>

__device__ unsigned int toIndex(uint2 coordinates, int height, int width) {
    return  coordinates.y * width + coordinates.x;
}

__device__ unsigned int toIndex(unsigned int x, unsigned int y, int height, int width) {
    return  y * width + x;
}

__device__ uint2 toCoord(unsigned int index, int height, int width) {
    return make_uint2(fmodf(index, width), index/width);
}

__device__ float calcFlux(float3* material, float flux, unsigned int threadID, unsigned int neighbor, float deltatime) {
    float a = 1;
    float l = 1;
    float g = 9.81;
    float deltaE = material[threadID].x + material[threadID].y - material[neighbor].x - material[neighbor].y;
    return fmaxf(0, flux + deltatime * a * ((g * deltaE) / l));
}

__global__ void runIteration(float3* material, float2* velocity, float4* flux, curandState* const rngStates, int height, int width, float deltatime)
{
    unsigned int threadID = blockIdx.x * blockIdx.x + threadIdx.x;
    uint2 coords = toCoord(threadID, height, width);


    float intermediate_hydration = curand_uniform(&rngStates[threadID]);

    float4 intermediate_flux = make_float4(0, 0, 0, 0);
    intermediate_flux.x = calcFlux(material, flux[threadID].x, threadID, toIndex(coords.x - 1, coords.y, height, width), deltatime);
    intermediate_flux.y = calcFlux(material, flux[threadID].y, threadID, toIndex(coords.x, coords.y + 1, height, width), deltatime);
    intermediate_flux.z = calcFlux(material, flux[threadID].z, threadID, toIndex(coords.x + 1, coords.y, height, width), deltatime);
    intermediate_flux.w = calcFlux(material, flux[threadID].w, threadID, toIndex(coords.x, coords.y - 1, height, width), deltatime);



    float a = material[threadID].y * 1 * 1; //wip replace with constant for scale
    float4 current_f = flux[threadID];
    float b = current_f.x + current_f.y + current_f.z + current_f.w;
    float K = fminf(1, b * deltatime);

    material[threadID].x = flux[threadID].x;
}

__global__ void initKernel(curandState* const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, threadID, 0, &rngStates[threadID]);
}


float* run_erosion(int height, int width)
{
    // Allocate array on device memory
    int size = height * width;
    
    float3* d_material;
    cudaMalloc(&d_material, size * sizeof(float3));

    float2* d_velocity;
    cudaMalloc(&d_velocity, size * sizeof(float2));

    float4* d_flux;
    cudaMalloc(&d_flux, size * sizeof(float4));


    // Allocate on host memory
    auto h_result_texture = new float[size];

    curandState* randStates = nullptr;
    cudaMalloc(&randStates, size * sizeof(curandState));

    initKernel<<<height, width>>>(randStates, rand());

    for (unsigned int i = 0; i < 100; i++)
    {
        runIteration <<<height, width >> > (d_material, d_velocity, d_flux, randStates, height, width);
    }


    cudaMemcpy(h_result_texture, (float*)d_material, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_material);
    cudaFree(d_velocity);
    cudaFree(d_flux);

    return h_result_texture;
}
