#include <algorithm>
#include <curand_kernel.h>
#include <iostream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_save.h"
#include "flux_erosion.h"

__global__ void copyToFloat3(float* d_input, float3* d_output, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    d_output[idx] = make_float3(d_input[idx], 0.0f, 0.0f);
}

__global__ void extractHeight(float3* d_input, float* d_output, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    d_output[idx] = d_input[idx].x;
}

__device__ float magnitude(float2 input) {
    return sqrt(input.x * input.x + input.y * input.y);
}

__device__ float bilinearInterpolate(float Q00, float Q10, float Q01, float Q11, float tx, float ty) {
    float I0 = (1 - tx) * Q00 + tx * Q10;
    float I1 = (1 - tx) * Q01 + tx * Q11;
    return (1 - ty) * I0 + ty * I1;
}

__device__ float clamp(float value, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(value, maxVal));
}

__device__ unsigned int toIndex(int x, int y, int height, int width) {
    /*if (x >= width) x = x % width;
    if (x < 0) x = width - (x % width);

    if (y >= height) y = y % height;
    if (y < 0) y = height - (y % height);*/

    x = ((x % width) + width) % width;
    y = ((y % height) + height) % height;

    if (x < 0 || x >= width)
        printf("X is wrong: %.i \n", x);

    if (y < 0 || y >= width)
        printf("X is wrong: %.i \n", y);

    /*if (y * width + x >= width * height || y * width + x < 0)
        printf("Index out of range for %.4f, %.4f id: %.4f \n", y, x, y * width + x);*/

    return  y * width + x;
}

__device__ unsigned int toIndex(uint2 coordinates, int height, int width) {
    return toIndex(coordinates.x, coordinates.y, height, width);
}

__device__ uint2 toCoord(unsigned int index, int height, int width) {
    return make_uint2(fmodf(index, width), index/width);
}

__device__ float Slope(float* map, int x, int y, int height, int width) {
    double dzdx = (map[toIndex(x + 1, y, height, width)] - map[toIndex(x - 1, y, height, width)]) / 2.0;
    double dzdy = (map[toIndex(x, y + 1, height, width)] - map[toIndex(x, y - 1, height, width)]) / 2.0;

    return std::sqrt(dzdx * dzdx + dzdy * dzdy);
}

__device__ float fluxComponentComputation(float *material, float* hydration, float &flux, unsigned int idx, unsigned int neighbor, float deltatime, float gravity, float pipe_cross_section, float pipe_length) {
    float deltaE = material[idx] + hydration[idx] - material[neighbor] - hydration[neighbor];
    //printf("flux: %.4f", fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length)));
    return fmaxf(0, flux + deltatime * pipe_cross_section * ((gravity * deltaE) / pipe_length));
}

__global__ void initErosionKernel(curandState* const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, idx, 0, &rngStates[idx]);
}

__global__ void rainComputation(float* hydration, curandState* const rngStates, int height, int width, float deltatime)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hydration[idx] += curand_uniform(&rngStates[idx]) * deltatime;
}

__global__ void fluxComputation(float* material, float* hydration, float4* flux, int height, int width, float deltatime, float gravity, float pipe_cross_section, float pipe_length)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx >= height * width) return;

    uint2 coords = toCoord(idx, height, width);

    float4 current_f = flux[idx];

    float K = fminf(1, hydration[idx] / ((current_f.x + current_f.y + current_f.z + current_f.w) * deltatime));

    float4 intermediate_flux = make_float4(0, 0, 0, 0);
    
    intermediate_flux.x = fluxComponentComputation(material, hydration, flux[idx].x, idx, toIndex(coords.x - 1, coords.y, height, width), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.y = fluxComponentComputation(material, hydration, flux[idx].y, idx, toIndex(coords.x, coords.y + 1, height, width), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.z = fluxComponentComputation(material, hydration, flux[idx].z, idx, toIndex(coords.x + 1, coords.y, height, width), deltatime, gravity, pipe_cross_section, pipe_length) * K;
    intermediate_flux.w = fluxComponentComputation(material, hydration, flux[idx].w, idx, toIndex(coords.x, coords.y - 1, height, width), deltatime, gravity, pipe_cross_section, pipe_length) * K;


    //printf("flux: %.4f : %.4f : %.4f : %.4f\n", intermediate_flux.x, intermediate_flux.y, intermediate_flux.z, intermediate_flux.w);

    //printf("%.4f, %.4f, %.4f, %.4f", intermediate_flux.x, intermediate_flux.y, intermediate_flux.z, intermediate_flux.w);

    flux[idx] = intermediate_flux;
}



__global__ void flowComputation(float* hydration, float4* flux, float2* velocity, int height, int width, float deltatime, float pipe_length)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 coords = toCoord(idx, height, width);

    float flowIn = 0;

    flowIn += flux[toIndex(coords.x + 1, coords.y, height, width)].z;
    flowIn += flux[toIndex(coords.x, coords.y + 1, height, width)].w;
    flowIn += flux[toIndex(coords.x - 1, coords.y, height, width)].x;
    flowIn += flux[toIndex(coords.x, coords.y - 1, height, width)].y;

    float4 localFlux = flux[idx];
    float flowOut = localFlux.x + localFlux.y + localFlux.z + localFlux.w;

    float deltaVolume = deltatime * (flowIn - flowOut);

    hydration[idx] += deltaVolume / pipe_length;
    //printf("Indexes: %.4f - %.4f - %.4f - %.4f", toIndex(coords.x - 1, coords.y, height, width), toIndex(coords.x + 1, coords.y, height, width), toIndex(coords.x, coords.y + 1, height, width), toIndex(coords.x, coords.y - 1, height, width));

    velocity[idx].x = (flux[toIndex(coords.x - 1, coords.y, height, width)].z - localFlux.x + localFlux.z - flux[toIndex(coords.x + 1, coords.y, height, width)].x) / 2.0;
    velocity[idx].y = (flux[toIndex(coords.x, coords.y + 1, height, width)].y - localFlux.w + localFlux.y - flux[toIndex(coords.x, coords.y - 1, height, width)].w) / 2.0;

    if (velocity[idx].x == -INFINITY) {
        printf("vel: %.4f", velocity[idx].x);
    }

    //printf("velocity: %.4f : %.4f hydration: %.4f deltaVol: %.4f flowIn: %.4f flowOut: %.4f\n", velocity[idx].x, velocity[idx].y, hydration[idx], deltaVolume, flowIn, flowOut);
}



__global__ void sedimentComputation(float* material, float* sediment, float2* velocity, int height, int width, float deltatime, float kc, float ks)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint2 coord = toCoord(idx, height, width);

    float slope = Slope(material, coord.x, coord.y, height, width);

    float threshold = 0.03;
    float C = kc * sinf(fmaxf(slope, threshold)) * magnitude(velocity[idx]); // replace 1 with local slope

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
}

__global__ void transportComputation(float* sediment, float2* velocity, int height, int width, float deltatime)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 ucoords = toCoord(idx, height, width);
    int2 coords = make_int2(ucoords.x, ucoords.y);

    float2 vel = velocity[idx];

    /*if (vel.x == -INFINITY) {
        printf("vel: %.4f", vel.x);
    }*/

    float2 sample_coords = make_float2(coords.x - vel.x * deltatime, coords.y - vel.y * deltatime);

    /*if (sample_coords.x == -INFINITY) {
        printf("Sample coords infite!");
    }*/

    int x0 = floorf(sample_coords.x);
    int x1 = x0 + 1;
    int y0 = floorf(sample_coords.y);
    int y1 = y0 + 1;

    float weightX = sample_coords.x - x0;
    float weightY = sample_coords.y - y0;

    float corner00 = sediment[toIndex(x0, y0, height, width)];
    float corner10 = sediment[toIndex(x1, y0, height, width)];
    float corner01 = sediment[toIndex(x0, y1, height, width)];
    float corner11 = sediment[toIndex(x1, y1, height, width)];

    sediment[idx] = bilinearInterpolate(corner00, corner10, corner01, corner11, weightX, weightY);
}

__global__ void evaporateComputation(float* hydration, int height, int width, float deltatime, float ke)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    hydration[threadID] = hydration[threadID] * (1 - ke * deltatime);
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
    , completed(0)
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

void FluxVelocityErosion::start(float *input, int iterations) {
    // Allocate array on device memory
    unsigned int size = mHeight * mWidth;
    
    /*dim3 blockDim(16, 16);
    dim3 gridDim((mWidth + blockDim.x - 1) / blockDim.x,
        (mHeight + blockDim.y - 1) / blockDim.y);*/


    if (mMaterial != nullptr)
        cudaFree(mMaterial);
    cudaMalloc(&mMaterial, size * sizeof(float));
    cudaMemcpy(mMaterial, input, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&mHydration, size * sizeof(float));
    cudaMalloc(&mSediment, size * sizeof(float));
    cudaMalloc(&mFlux, size * sizeof(float4));
    cudaMalloc(&mVelocity, size * sizeof(float2));
    cudaMalloc(&mRandStates, size * sizeof(curandState));

    initErosionKernel <<<mWidth, mHeight >>> (mRandStates, rand());

    simulate(iterations);
}

void FluxVelocityErosion::resume(int iterations) {
    simulate(iterations);
}

void FluxVelocityErosion::simulate(int iterations) {
    for (unsigned int i = 0; i < iterations; i++)
    {
        rainComputation << <mWidth, mHeight >> > (mHydration, mRandStates, mHeight, mWidth, 0.02);

        fluxComputation << <mWidth, mHeight >> > (mMaterial, mHydration, mFlux, mHeight, mWidth, 0.02, mGravityConstant, mPipeCrossSectionConstant, mPipeLengthConstant);

        flowComputation << <mWidth, mHeight >> > (mHydration, mFlux, mVelocity, mHeight, mWidth, 0.02, mPipeLengthConstant);
        //cudaDeviceSynchronize();

        sedimentComputation << <mWidth, mHeight >> > (mMaterial, mSediment, mVelocity, mHeight, mWidth, 0.02, mCapacityConstant, mDissolvingConstant);

        transportComputation << <mWidth, mHeight >> > (mSediment, mVelocity, mHeight, mWidth, 0.02);

        evaporateComputation << <mWidth, mHeight >> > (mHydration, mHeight, mWidth, 0.02, mEvaporationConstant);
        cudaThreadSynchronize();

        completed++;
        printf("Batch: %4i/%4i Total: %.i \n", i + 1, iterations, completed);
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

