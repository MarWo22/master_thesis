#ifndef EROSION_KERNEL_CUH
#define EROSION_KERNEL_CUH
#include "cuda_texture.cuh"
#include "plate_data.h"
#include "vec2.cuh"
#include <curand_kernel.h>


__global__ void rainComputation(CudaTexture<float> hydration, curandState* const rngStates, float deltatime);

__global__ void fluxComputation(CudaTexture<float> material, CudaTexture<float> hydration, CudaTexture<float4> flux, float deltatime, float gravity, float pipe_cross_section, float pipe_length);

__global__ void flowComputation(CudaTexture<float> hydration, CudaTexture<float4> flux, CudaTexture<float2> velocity, float deltatime, float pipe_length);

__global__ void sedimentComputation(CudaTexture<float> material, CudaTexture<float> sediment, CudaTexture<Vec2<float>> velocity, float deltatime, float kc, float ks);

__global__ void transportComputation(CudaTexture<float> sediment, CudaTexture<float> sedimentBuffer, CudaTexture<Vec2<float>> velocity, float deltatime);

__global__ void evaporateComputation(float* hydration, float deltatime, float ke);


#endif //EROSION_KERNEL_CUH