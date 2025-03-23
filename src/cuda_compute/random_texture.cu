#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_noise.cuh"

#include "random_texture.h"
#include <curand_kernel.h>
#include <iostream>

__global__ void initKernel(curandState *const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, threadID, 0, &rngStates[threadID]);
}

__global__ void generateRandomTextureKernel(float *texture, int width, int height)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    uint2 coord = make_uint2(fmodf(threadID, width), threadID / width);

    texture[threadID] = cudaNoise::simplexNoise(make_float3(coord.x, coord.y, 0), 0.01, 1212);
}

__global__ void convertTextureTo16BitKernel(const float *input_texture, uint16_t *output_texture)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    float value = input_texture[threadID];
    auto convertedValue = static_cast<uint16_t>(value * 65535);
    output_texture[threadID] = convertedValue;
}

__global__ void convertTextureTo8BitKernel(const float *input_texture, uint8_t *output_texture)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    float value = input_texture[threadID];
    auto convertedValue = static_cast<uint8_t>(value * 255);
    output_texture[threadID] = convertedValue;
}

float* generateRandomTexture(int height, int width)
{
    // Allocate array on device memory
    int size = height * width;
    float* d_texture;
    cudaMalloc(&d_texture, size*sizeof(float));

    // Allocate on host memory
    auto h_texture = new float[size];

    curandState *randStates = nullptr;
    cudaMalloc(&randStates, size*sizeof(curandState));

    initKernel<<<height, width>>>(randStates, rand());

    generateRandomTextureKernel<<<1024, 1024>>>(d_texture, width, height);

    return d_texture;
}

uint16_t* convertFloatTextureTo16Bit(float *texture, int height, int width)
{
    int size = height * width;
    float* d_input_texture = nullptr;
    uint16_t* d_output_texture;
    cudaMalloc(&d_input_texture, size*sizeof(float));
    cudaMalloc(&d_output_texture, size*sizeof(uint16_t));
    cudaMemcpy(d_input_texture, texture, size * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate on host memory
    auto h_texture = new uint16_t[size];

    convertTextureTo16BitKernel<<<height, width>>>(d_input_texture, d_output_texture);
    cudaMemcpy(h_texture, d_output_texture, size * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input_texture);
    cudaFree(d_output_texture);
    return h_texture;
}

uint8_t* convertFloatTextureTo8Bit(float *texture, int height, int width)
{
    int size = height * width;
    float* d_input_texture;
    uint8_t* d_output_texture;
    cudaMalloc(&d_input_texture, size*sizeof(float));
    cudaMalloc(&d_output_texture, size*sizeof(uint8_t));
    cudaMemcpy(d_input_texture, texture, size * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate on host memory
    auto h_texture = new uint8_t[size];

    convertTextureTo8BitKernel<<<height, width>>>(d_input_texture, d_output_texture);
    cudaMemcpy(h_texture, d_output_texture, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input_texture);
    cudaFree(d_output_texture);
    return h_texture;
}
