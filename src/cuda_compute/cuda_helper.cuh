#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include "vec2.cuh"


__device__ __inline__ unsigned int getInvokeIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __inline__ Vec2<int> getTextureIndex(const unsigned int invokeIndex, const Vec2<int> &textureSize)
{
    const int idx = static_cast<int>(invokeIndex);
    return {idx % textureSize.x,
        idx / textureSize.x};
}

__device__ __inline__ Vec2<int> getTextureIndex(const Vec2<int> &textureSize)
{
    const unsigned int invokeIndex = getInvokeIndex();
    return getTextureIndex(invokeIndex, textureSize);
}

__device__ __inline__ bool isWithinBounds(const unsigned int invokeIndex, const Vec2<int> &textureSize)
{
    return invokeIndex < textureSize.x * textureSize.y;
}

__device__ __inline__ bool isWithinBounds(const Vec2<int> &textureSize)
{
    const unsigned int invokeIndex = getInvokeIndex();
    return isWithinBounds(invokeIndex, textureSize);
}

#endif //CUDA_HELPER_CUH
