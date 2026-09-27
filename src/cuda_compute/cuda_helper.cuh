#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include "types/vec2.cuh"
#include <curand_kernel.h>

__device__ __inline__ unsigned int getInvokeIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __inline__ Vec2<int> getTextureIndex(const unsigned int invokeIndex, const Vec2<int> &textureSize)
{
    const int idx = static_cast<int>(invokeIndex);
    return {
        idx % textureSize.x,
        idx / textureSize.x
    };
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

__device__ __inline__ float rad2deg(const float radians)
{
    return radians * 180.0 / CURAND_PI_DOUBLE;
}

template<typename T>
__device__ __inline__ T interpolate(const Vec2<float> &textureCoordinate, const CudaTexture<T> *texture)
{
    const CudaTexture<T> &r_texture = *texture;


    int x0 = floorf(textureCoordinate.x);
    int x1 = x0 + 1;
    int y0 = floorf(textureCoordinate.y);
    int y1 = y0 + 1;

    float weightX = textureCoordinate.x - x0;
    float weightY = textureCoordinate.y - y0;

    T I0 = (1 - weightX) * r_texture[(Vec2<int>(x0, y0))] + weightX * r_texture[(Vec2<int>(x1, y0))];
    T I1 = (1 - weightX) * r_texture[(Vec2<int>(x0, y1))] + weightX * r_texture[(Vec2<int>(x1, y1))];
    return (1 - weightY) * I0 + weightY * I1; //Keep in mind. I0 and I1 might need to be swapped !!!!!!!
}

__device__ __inline__ float clamp(const float value, const float min, const float max)
{
    return fmaxf(fminf(value, max), min);
}

__device__ __inline__ float clamp01(const float value)
{
    return fmaxf(fminf(value, 1.0f), 0.0f);
}

__device__ __inline__ float remap(const float value, const float inStart, const float inStop, const float outStart,
                                  const float outStop)
{
    return outStart + (outStop - outStart) * ((value - inStart) / (inStop - inStart));
}

__device__ __inline__ float inverseClampedLerp(float value, float min, float max)
{
    return clamp01((value - min) / (max - min));
}

__device__ __inline__ unsigned int upperTriangleIndex(const unsigned int i, const unsigned int j,
                                                               const int matrixSize,
                                                               const unsigned int bitEntrySize = 1)
{
    const unsigned int minVal = (i < j) ? i : j;
    const unsigned int maxVal = (i < j) ? j : i;
    const unsigned int base = minVal * (2 * matrixSize - minVal - 1) / 2;
    return bitEntrySize * (base + (maxVal - minVal - 1));
}

__device__ __inline__ unsigned int upperTriangleIndexUnchecked(const unsigned int i, const unsigned int j,
                                                               const int matrixSize,
                                                               const unsigned int bitEntrySize = 1)
{
    const unsigned int base = i * (2 * matrixSize - i - 1) / 2;
    return bitEntrySize * (base + (j - i - 1));
}


template<typename T>
__device__ __inline__ void swap(T &x, T &y) noexcept
{
    T tmp = x;
    x = y;
    y = tmp;
}

template<typename T>
__device__ __inline__ void sort2(T &a, T &b)
{
    if (a > b) swap(a, b);
}

template<typename T>
__device__ __inline__ void sort3(T &a, T &b, T &c)
{
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
}

template<typename T>
__device__ __inline__ void sort4(T &a, T &b, T &c, T &d)
{
    if (a > b) swap(a, b);
    if (c > d) swap(c, d);
    if (a > c) swap(a, c);
    if (b > d) swap(b, d);
    if (b > c) swap(b, c);
}

#endif //CUDA_HELPER_CUH
