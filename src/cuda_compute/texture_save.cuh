#ifndef TEXTURE_SAVE_H
#define TEXTURE_SAVE_H
#include <iostream>

#include "types/cuda_texture.cuh"

#include <vector>
#include <cuda_runtime.h>
#include <png.h>


void saveGrayscale8BitCudaTextureToDiskAsRgb(const char *fileName, const CudaTextureHost<uint8_t> &texture);

void saveFloatCudaTextureToDiskAsGray16(
    const char* fileName,
    const CudaTextureHost<float>& texture,
    float minVal,
    float maxVal);



#endif //TEXTURE_SAVE_H
