//
// Created by marti on 15/07/2025.
//

#ifndef DEBUG_KERNELS_CUH
#define DEBUG_KERNELS_CUH
#include "../types/cuda_texture.cuh"
#include "../types/voronoi_seed.h"

__global__ void drawVoronoiSeedsToTexture(CudaTexture<uint8_t> *idTexturePtr, const VoronoiSeed *seeds,
                                          int numSeeds, int seedDrawSize);


#endif //DEBUG_KERNELS_CUH
