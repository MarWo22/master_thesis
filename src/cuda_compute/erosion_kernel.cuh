#ifndef EROSION_KERNEL_CUH
#define EROSION_KERNEL_CUH
#include "cuda_texture.cuh"
#include "vec2.cuh"


//__global__ void rainComputation(CudaTexture<float>* hydration, float deltatime, int seed);
//
//__global__ void fluxComputation(CudaTexture<float>* material, CudaTexture<float>* hydration, CudaTexture<float4>* flux, float deltatime, float gravity, float pipe_cross_section, float pipe_length);
//
//__global__ void flowComputation(CudaTexture<float>* hydration, CudaTexture<float4>* flux, CudaTexture<Vec2<float>>* velocity, float deltatime, float pipe_length);
//
//__global__ void sedimentComputation(CudaTexture<float>* material, CudaTexture<float>* sediment, CudaTexture<Vec2<float>>* velocity, float deltatime, float kc, float ks);
//
//__global__ void transportComputation(CudaTexture<float>* sediment, CudaTexture<float>* sedimentBuffer, CudaTexture<Vec2<float>>* velocity, float deltatime);
//
//__global__ void evaporateComputation(CudaTexture<float>* hydration, float deltatime, float ke);
//
//__global__ void initMaterial(CudaTexture<float>* material, int seed);

#endif //EROSION_KERNEL_CUH