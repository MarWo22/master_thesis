#ifndef VEC2_CUH
#define VEC2_CUH

#include <cuda_runtime.h>
#include <cmath>

template <typename T>
struct Vec2 {
    T x, y;

    __device__ Vec2()
        : x(0)
        , y(0)
    {}

    __device__ Vec2(T _x, T _y)
        : x(_x)
        , y(_y)
    {}

    __device__ Vec2<T> operator+(const Vec2<T>& other) const
    {
        return Vec2<T>(x + other.x, y + other.y);
    }

    __device__ Vec2<T> operator-(const Vec2<T>& other) const
    {
        return Vec2<T>(x - other.x, y - other.y);
    }

    __device__ Vec2<T> operator+(const T& other) const
    {
        return Vec2<T>(x + other, y + other);
    }

    __device__ Vec2<T> operator-(const T& other) const
    {
        return Vec2<T>(x - other, y - other);
    }

    __device__ Vec2<T> operator*(T scalar) const
    {
        return Vec2<T>(x * scalar, y * scalar);
    }

    __device__ T dot(const Vec2<T>& other) const
    {
        return x * other.x + y * other.y;
    }

    __device__ float magnitude() const
    {
        return sqrt(x*x + y*y);
    }

    __device__ Vec2<T> normalized() const
    {
        float mag = magnitude();
        return Vec2<T>(x / mag, y / mag);
    }
};

#endif //VEC2_CUH
