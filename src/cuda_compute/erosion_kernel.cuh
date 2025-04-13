#ifndef EROSION_KERNEL_CUH
#define EROSION_KERNEL_CUH
#include "cuda_texture.cuh"
#include "vec2.cuh"
#include "flux_erosion.h"


__global__ void rainComputation(CudaErosionData data, float deltatime, int seed);

__global__ void fluxComputation(CudaErosionData data, float deltatime, float gravity, float pipe_cross_section, float pipe_length);

__global__ void flowComputation(CudaErosionData data, float deltatime, float pipe_length);

__global__ void sedimentComputation(CudaErosionData data, float deltatime, float kc, float ks, float kd);

__global__ void transportComputation(CudaErosionData data, float deltatime);

__global__ void evaporateComputation(CudaErosionData data, float deltatime, float ke);

__global__ void initMaterial(CudaErosionData data, int seed);

#endif //EROSION_KERNEL_CUH