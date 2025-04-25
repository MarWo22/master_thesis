#ifndef VEC2_CUH
#define VEC2_CUH

#include <cuda_runtime.h>
#include <cmath>

template <typename T>
struct Vec2 {
    T x, y;

    __device__ __host__ Vec2()
        : x(0)
        , y(0)
    {}

    __device__ __host__ Vec2(T _x, T _y)
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

    __device__ [[nodiscard]] float magnitude() const
    {
        return sqrtf(x*x + y*y);
    }

    __device__ [[nodiscard]] float magnitudeSquared() const
    {
        return x*x + y*y;
    }

    __device__ [[nodiscard]] float distance(const Vec2 &other) const
    {
        return sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
    }

    __device__ [[nodiscard]] float distanceSquared(const Vec2 &other) const
    {
        return (other.x - x) * (other.x - x) + (other.y - y) * (other.y - y);
    }

    __device__ Vec2<T> normalized() const
    {
        float mag = magnitude();
        return Vec2<T>(x / mag, y / mag);
    }

    __device__ float2 toFloat2() const
    {
        return make_float2(x, y);
    }

    __device__ float3 toFloat3() const
    {
        return make_float3(x, y, 0);
    }
};

#endif //VEC2_CUH
