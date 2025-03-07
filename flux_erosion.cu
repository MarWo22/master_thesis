#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "flux_erosion.h"
#include <curand_kernel.h>
#include <iostream>

__global__ void copyToFloat3(float* d_input, float3* d_output, int size) {
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= size) return;
    d_output[threadID] = make_float3(d_input[threadID], 0.0f, 0.0f);
}

__global__ void extractHeight(float3* d_input, float* d_output, int size) {
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= size) return;
    d_output[threadID] = d_input[threadID].x;
}

__device__ unsigned int toIndex(uint2 coordinates, int height, int width) {
    return  coordinates.y * width + coordinates.x;
}

__device__ unsigned int toIndex(unsigned int x, unsigned int y, int height, int width) {
    return  y * width + x;
}

__device__ uint2 toCoord(unsigned int index, int height, int width) {
    return make_uint2(fmodf(index, width), index/width);
}

__device__ float fluxComponentComputation(float *material, float* hydration, float &flux, unsigned int threadID, unsigned int neighbor, float deltatime) {
    float pipe_cross_section = 1;//A
    float pipe_length = 1; //l
    float gravity = 9.81;//g
    float deltaE = material[threadID] + hydration[threadID] - material[neighbor] - hydration[neighbor];
    return fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length));
}

__global__ void initErosionKernel(curandState* const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, threadID, 0, &rngStates[threadID]);
}

__global__ void rainComputation(float* hydration, curandState* const rngStates, int height, int width, float deltatime)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    hydration[threadID] += curand_uniform(&rngStates[threadID]) * deltatime;
}

__global__ void fluxComputation(float* material, float* hydration, float4* flux, int height, int width, float deltatime)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= height * width) return;

    uint2 coords = toCoord(threadID, height, width);

    float4 current_f = flux[threadID];
    float K = fminf(1, (current_f.x + current_f.y + current_f.z + current_f.w) * deltatime);

    float4 intermediate_flux = make_float4(0, 0, 0, 0);
    
    intermediate_flux.x = fluxComponentComputation(material, hydration, flux[threadID].x, threadID, toIndex(coords.x - 1, coords.y, height, width), deltatime) * K;
    intermediate_flux.y = fluxComponentComputation(material, hydration, flux[threadID].y, threadID, toIndex(coords.x, coords.y + 1, height, width), deltatime) * K;
    intermediate_flux.z = fluxComponentComputation(material, hydration, flux[threadID].z, threadID, toIndex(coords.x + 1, coords.y, height, width), deltatime) * K;
    intermediate_flux.w = fluxComponentComputation(material, hydration, flux[threadID].w, threadID, toIndex(coords.x, coords.y - 1, height, width), deltatime) * K;

    flux[threadID] = intermediate_flux;
}



__global__ void flowComputation(float* material, float* hydration, float4* flux, curandState* const rngStates, int height, int width, float deltatime)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 coords = toCoord(threadID, height, width);

    float flowIn = 0;

    flowIn += flux[toIndex(coords.x + 1, coords.y, height, width)].z;
    flowIn += flux[toIndex(coords.x, coords.y + 1, height, width)].w;
    flowIn += flux[toIndex(coords.x - 1, coords.y, height, width)].x;
    flowIn += flux[toIndex(coords.x, coords.y - 1, height, width)].y;

    float4 localFlux = flux[threadID];
    float flowOut = localFlux.x + localFlux.y + localFlux.z + localFlux.w;

    float deltaVolume = deltatime * (flowIn - flowOut);

    float pipe_length = 1; //l

    hydration[threadID] += deltaVolume / pipe_length;
}

FluxVelocityErosion::FluxVelocityErosion(int width, int height)
    : mWidth(width)
    , mHeight(height)
    , mRandStates(nullptr)
    , mMaterial(nullptr)
    , mHydration(nullptr)
    , mSediment(nullptr)
    , mFlux(nullptr)
    , mVelocity(nullptr)
{}

FluxVelocityErosion::~FluxVelocityErosion() {
    if (mMaterial != nullptr)
        cudaFree(mMaterial);
    if (mHydration != nullptr)
        cudaFree(mHydration);
    if (mSediment != nullptr)
        cudaFree(mSediment);
    if (mFlux != nullptr)
        cudaFree(mFlux);
    if (mVelocity != nullptr)
        cudaFree(mVelocity);
}

void FluxVelocityErosion::simulate(float *input, int iterations) {
    // Allocate array on device memory
    unsigned int size = mHeight * mWidth;
    dim3 blockDim(16, 16);
    dim3 gridDim((mWidth + blockDim.x - 1) / blockDim.x,
        (mHeight + blockDim.y - 1) / blockDim.y);


    if (mMaterial != nullptr)
        cudaFree(mMaterial);
    cudaMalloc(&mMaterial, size * sizeof(float));
    cudaMemcpy(mMaterial, input, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&mHydration, size * sizeof(float));
    cudaMalloc(&mSediment, size * sizeof(float));
    cudaMalloc(&mFlux, size * sizeof(float4));
    cudaMalloc(&mVelocity, size * sizeof(float2));
    cudaMalloc(&mRandStates, size * sizeof(curandState));

    initErosionKernel <<<gridDim, blockDim >>> (mRandStates, rand());



    for (unsigned int i = 0; i < iterations; i++)
    {
        rainComputation << <gridDim, blockDim >> > (mHydration, mRandStates, mHeight, mWidth, 0.02);

        fluxComputation << <gridDim, blockDim >> > (mMaterial, mHydration, mFlux, mHeight, mWidth, 0.02);

        flowComputation << <gridDim, blockDim >>> (mMaterial, mHydration, mFlux, mRandStates, mHeight, mWidth, 0.02);
        printf("Completed: %4i/%4i\n", i, iterations);

    }
}

void FluxVelocityErosion::getHydration(float* output) {
    unsigned int size = sizeof(float) * mHeight * mWidth;
    cudaMemcpy(output, mHydration, size, cudaMemcpyDeviceToHost);
}

void FluxVelocityErosion::getMaterial(float* output) {
    unsigned int size = sizeof(float) * mHeight * mWidth;
    cudaMemcpy(output, mMaterial, size, cudaMemcpyDeviceToHost);
}

void FluxVelocityErosion::getSediment(float* output) {
    unsigned int size = sizeof(float) * mHeight * mWidth;
    cudaMemcpy(output, mSediment, size, cudaMemcpyDeviceToHost);
}