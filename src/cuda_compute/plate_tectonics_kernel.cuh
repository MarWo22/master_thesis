#ifndef PLATE_TECTONICS_KERNEL_CUH
#define PLATE_TECTONICS_KERNEL_CUH
#include "cuda_texture.cuh"
#include "plate_data.h"
#include "vec2.cuh"

__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, const Vec2<float> *seeds, int numSeeds);

__global__ void plateMovement(uint8_t *idTexture, float *crustTexture, PlateData *plateLookup, Vec2<int> textureSize);

#endif //PLATE_TECTONICS_KERNEL_CUH
