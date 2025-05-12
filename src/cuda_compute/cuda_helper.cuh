#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include "vec2.cuh"
#include <curand_kernel.h>

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

__device__ __inline__ float Deg2Rad(const float degrees)
{
    return degrees * CURAND_PI_DOUBLE / 180.0;
}

__device__ __inline__ float rad2deg(const float radians) {
    return radians * 180.0 / CURAND_PI_DOUBLE;
}

template<typename T>
__device__ __inline__ T interpolate(const Vec2<float>& textureCoordinate, const CudaTexture<T>* texture)
{
    const CudaTexture<T>& r_texture = *texture;


    int x0 = floorf(textureCoordinate.x);
    int x1 = x0 + 1;
    int y0 = floorf(textureCoordinate.y);
    int y1 = y0 + 1;

    float weightX = textureCoordinate.x - x0;
    float weightY = textureCoordinate.y - y0;

    T I0 = (1 - weightX) * r_texture[(Vec2<int>(x0, y0))] + weightX * r_texture[(Vec2<int>(x1, y0))];
    T I1 = (1 - weightX) * r_texture[(Vec2<int>(x0, y1))] + weightX * r_texture[(Vec2<int>(x1, y1))];
    return (1 - weightY) * I0 + weightY * I1;         //Keep in mind. I0 and I1 might need to be swapped !!!!!!!
}

__device__ __inline__ float clamp(const float value, const float min, const float max) {
    return fmaxf(fminf(value, max), min);
}

__device__ __inline__ float clamp01(const float value) {
    return fmaxf(fminf(value, 1.0f), 0.0f);
}

__device__ __inline__ float remap(const float value, const float inStart, const float inStop, const float outStart, const float outStop) {
    return outStart + (outStop - outStart) * ((value - inStart) / (inStop - inStart));
}

#endif //CUDA_HELPER_CUH
