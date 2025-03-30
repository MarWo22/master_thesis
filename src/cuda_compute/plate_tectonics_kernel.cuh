#ifndef PLATE_TECTONICS_KERNEL_CUH
#define PLATE_TECTONICS_KERNEL_CUH
#include "cuda_texture.cuh"
#include "plate_data.h"
#include "vec2.cuh"

__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, const Vec2<float> *seeds, int numSeeds);

__global__ void plateMovement(CudaTexture<uint8_t> *idTexturePtr, CudaTexture<uint8_t> *writeIdTexturePtr, const PlateData *plateLookup);

__global__ void updatePlateData(PlateData *plateLookup, Vec2<int> callSize);


#endif //PLATE_TECTONICS_KERNEL_CUH
