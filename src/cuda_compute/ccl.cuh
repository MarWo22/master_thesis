#ifndef CCL_CUH
#define CCL_CUH
#include "cuda_texture.cuh"


__global__ void copyToUint8Texture(const CudaTexture<unsigned int> *labelsPtr, CudaTexture<uint8_t> *writePtr,
                                   const unsigned int *labelIndices,
                                   int len);


__device__ unsigned findClamped(CudaTexture<unsigned int> &labels, unsigned int index);
__device__ unsigned findUnclamped(CudaTexture<unsigned int> &labels, unsigned int index);
__device__ void unionStep(CudaTexture<unsigned int> &labels, unsigned int index_a, unsigned index_b);
__global__ void init(const CudaTexture<uint8_t> *r_inputPtr, CudaTexture<unsigned int> *w_labelsPtr);
__global__ void analyzeClamped(CudaTexture<unsigned int> *w_labelsPtr);
__global__ void analyzeUnclamped(CudaTexture<unsigned int> *w_labelsPtr);

__global__ void reduce(const CudaTexture<uint8_t> *r_inputPtr, CudaTexture<unsigned int> *w_labelsPtr);

#endif //CCL_CUH
