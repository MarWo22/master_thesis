#ifndef CCL_CUH
#define CCL_CUH
#include "cuda_texture.cuh"

__global__ void initializeCCL(const CudaTexture<uint8_t> *idsPtr, CudaTexture<unsigned int> *labelsPtr);

__global__ void analysisCCL(const CudaTexture<uint8_t> *idsPtr, CudaTexture<unsigned int> *labelsPtr);

__global__ void labelReductionCCL(const CudaTexture<uint8_t> *idsPtr, CudaTexture<unsigned int> *labelsPtr);

__global__ void copyToUint8Texture(const CudaTexture<unsigned int> *labelsPtr, CudaTexture<uint8_t> *writePtr,
                                   const unsigned int *labelIndices,
                                   int len);

__device__ void reduction(const CudaTexture<uint8_t> &ids, CudaTexture<unsigned int> &labels,
                          unsigned int invokeIndex, const Vec2<int> &neighborIndex);

#endif //CCL_CUH
